import logging
import numpy as np
import numba as nb
import random
import json
from pathlib import Path
import h5py
import argparse

from june.interaction import Interaction
from june.epidemiology.infection import (
    Infection,
    InfectionSelector,
    InfectionSelectors,
    HealthIndexGenerator,
    SymptomTag,
    ImmunitySetter,
    Covid19,
    B16172,
)
from june.groups import Hospitals, Schools, Companies, Households, CareHomes, Cemeteries
from june.groups.travel import Travel
from june.groups.leisure import Cinemas, Pubs, Groceries, generate_leisure_for_config
from june.simulator import Simulator
from june.epidemiology.epidemiology import Epidemiology
from june.epidemiology.infection_seed import (
    InfectionSeed,
    Observed2Cases,
    InfectionSeeds,
)
from june.policy import Policies
from june.event import Events
from june import paths
from june.records import Record
from june.records.records_writer import combine_records
from june.domains import Domain, DomainSplitter
from june.mpi_setup import mpi_comm, mpi_rank, mpi_size

from june.tracker.tracker import Tracker
from june.tracker.tracker_merger import MergerClass

import pandas as pd
from june.records import RecordReader

import multiprocessing as mp

# ------------------------------------------------ #
#           Auxiliary functions                    #
# ------------------------------------------------ #

def set_random_seed(seed=999):
    """
    Sets global random seeds for numpy, Python's random module, and numba-compiled
    functions. Call once before the simulation loop to ensure reproducibility.
    """

    @nb.njit(cache=True)
    def set_seed_numba(seed):
        random.seed(seed)
        return np.random.seed(seed)

    np.random.seed(seed)
    set_seed_numba(seed)
    random.seed(seed)
    return

def keys_to_int(x):
    """Convert JSON-loaded dict keys from strings back to integers."""
    return {int(k): v for k, v in x.items()}

# Suppress verbose JUNE logging on all MPI ranks except rank 0
if mpi_rank > 0:
    logging.disable(logging.CRITICAL)

# ------------------------------------------------ #
#                   Main function                  #
# ------------------------------------------------ #

def run_iteration(iii, sims_metadata_df, sims_seeds_df, sims_local_susceptibility_df, args):
    """
    Run a single JUNE simulation (index iii) end-to-end:
      1. Extract per-simulation parameters from the metadata DataFrames.
      2. Set up the output folder.
      3. Build the JUNE simulator (domain, epidemiology, interaction, policies, etc.).
      4. Run the simulator.
      5. Combine MPI records into a single file (rank 0 only. I never used it).
      6. Export all output tables to .feather files.
      7. Delete the heavy .h5 record files to save storage.

    Parameters
    ----------
    iii : int
        Simulation index (matches the 'sim' column in the metadata CSVs).
    sims_metadata_df : pd.DataFrame
        Global simulation parameters (from settings_[]_sims.csv).
    sims_seeds_df : pd.DataFrame
        Seeding parameters (from settings_[]_seeds.csv).
    sims_local_susceptibility_df : pd.DataFrame or None
        MSOA-level local susceptibility and rate factors (from settings_[]_betas.csv).
        None if --betas_metadata_file was not provided.
    args : argparse.Namespace
        Parsed command-line arguments including paths and flags.
    """

    # -- Extract per-simulation parameters from metadata --
    waning_factor_i  = sims_metadata_df[sims_metadata_df['sim'] == iii]['susceptibility'].tolist()[0]
    beta_factor_i    = sims_metadata_df[sims_metadata_df['sim'] == iii]['beta_factor'].tolist()[0]
    seeding_dates_i  = sims_seeds_df[sims_seeds_df['sim'] == iii]['date'].tolist()
    seeding_MSOA_i   = sims_seeds_df[sims_seeds_df['sim'] == iii]['MSOA'].tolist()
    seeds_strength_i = (sims_seeds_df[sims_seeds_df['sim'] == iii]['cases_per_capita'] / 100).tolist()
    start_date_i     = sims_metadata_df[sims_metadata_df['sim'] == iii]['start_date'].tolist()[0]
    policy_file      = paths.configs_path / "defaults/policy/policy.yaml"

    # ------------------------------------------------ #
    #                  Set up output folder            #
    # ------------------------------------------------ #

    # Build the base output path for this simulation folder
    args.save_path = args.local_folder2 + args.output_path + f"{args.fol_index:02d}"
    args.save_path = Path(args.save_path)

    if mpi_rank == 0:
        # Append the simulation index to the folder name (e.g. results_hub/prefix02_003)
        OG_save_path = args.save_path
        args.save_path = Path(f"{str(OG_save_path)}_{iii:03d}")
        # If a folder with this name already exists, append a counter suffix to avoid overwriting
        counter = 1
        OG_save_path = args.save_path
        while args.save_path.is_dir() is True:
            args.save_path = Path(f"{str(OG_save_path)}_{counter:02d}")
            counter += 1
        args.save_path.mkdir(parents=True, exist_ok=False)

    # Broadcast the resolved save path from rank 0 to all MPI ranks
    mpi_comm.Barrier()
    args.save_path = mpi_comm.bcast(args.save_path, root=0)
    mpi_comm.Barrier()

    # Normalise the tracker flag from string to bool (argparse limitation)
    if args.tracker == "True":
        args.tracker = True
    else:
        args.tracker = False

    if mpi_rank == 0:
        print("Parameters path set to: {}".format(args.parameters))
        print("World path set to: {}".format(args.world_path))
        print("Save path set to: {}".format(args.save_path))
        print("\n", args.__dict__, "\n")

    # ------------------------------------------------ #
    #              Build simulator                     #
    # ------------------------------------------------ #

    CONFIG_PATH = args.config

    def generate_simulator():

        # ------------------------------------------------ #
        #              Set up recorder                     #
        # ------------------------------------------------ #

        # The Record object captures all infection events during the simulation
        # and writes them to a .h5 file, which is later parsed into feather tables.
        record = Record(
            record_path=args.save_path, record_static_data=True, mpi_rank=mpi_rank
        )

        # Rank 0 reads the geography from the .hdf5 world file and splits super areas
        # across MPI domains. The mapping is written to JSON so other ranks can load it.
        if mpi_rank == 0:
            with h5py.File(args.world_path, "r") as f:
                super_area_ids   = f["geography"]["super_area_id"]
                super_area_names = f["geography"]["super_area_name"]
                super_area_name_to_id = {
                    name.decode(): id for name, id in zip(super_area_names, super_area_ids)
                }
            super_areas_per_domain, score_per_domain = DomainSplitter.generate_world_split(
                number_of_domains=mpi_size, world_path=args.world_path
            )
            # Build name->domain and id->domain lookup dicts
            super_area_names_to_domain_dict = {}
            super_area_ids_to_domain_dict   = {}
            for domain, super_areas in super_areas_per_domain.items():
                for super_area in super_areas:
                    super_area_names_to_domain_dict[super_area] = domain
                    super_area_ids_to_domain_dict[
                        int(super_area_name_to_id[super_area])
                    ] = domain
            # Write to JSON so non-zero ranks can read them after the barrier
            with open("super_area_ids_to_domain.json", "w") as f:
                json.dump(super_area_ids_to_domain_dict, f)
            with open("super_area_names_to_domain.json", "w") as f:
                json.dump(super_area_names_to_domain_dict, f)

        print(f"mpi_rank {mpi_rank} waiting")
        mpi_comm.Barrier()

        # Non-zero ranks load the domain mapping written by rank 0
        if mpi_rank > 0:
            with open("super_area_ids_to_domain.json", "r") as f:
                super_area_ids_to_domain_dict = json.load(f, object_hook=keys_to_int)
        print(f"mpi_rank {mpi_rank} loading domain")

        # ------------------------------------------------ #
        #              Build domain                        #
        # ------------------------------------------------ #

        # Each MPI rank loads only the super areas assigned to its domain,
        # keeping memory usage proportional to 1/mpi_size of the full world.
        domain = Domain.from_hdf5(
            domain_id=mpi_rank,
            super_areas_to_domain_dict=super_area_ids_to_domain_dict,
            hdf5_file_path=args.world_path,
            interaction_config=args.parameters,
        )
        print(f"mpi_rank {mpi_rank} has loaded domain")

        # Geography-specific fixes: disable venue types that cause bugs or are
        # not applicable to certain regions.
        if args.input_world_file == "Gatesnewnorth":
            domain.care_home_visits = None   # Avoids a known bug in this geography
            domain.household_visits = None
        if args.input_world_file == "Darlington":
            domain.city_transports       = None  # No city transport in Darlington world
            domain.inter_city_transports = None
        if args.input_world_file == "Sunderland":
            domain.city_transports       = None
            domain.inter_city_transports = None

        # ------------------------------------------------ #
        #              Build local parameter dataframes    #
        # ------------------------------------------------ #

        # If a betas metadata file was provided, extract per-MSOA local susceptibility
        # and hospitalisation/death rate factors for this simulation.
        # Otherwise, default to uniform susceptibility (beta_local = 1) and rate factor = 1.
        if args.use_local_betas is True:
            local_susceptibility_df = sims_local_susceptibility_df[sims_local_susceptibility_df['sim'] == iii][['MSOA', 'local_susceptibility']]
            local_factor_rates_df   = sims_local_susceptibility_df[sims_local_susceptibility_df['sim'] == iii][['MSOA', 'factor_rate']]
        else:
            msoa_list = []
            for super_area in domain.super_areas:
                msoa_list.append(super_area.name)
            local_susceptibility_df = pd.DataFrame({
                'MSOA': msoa_list,
                'beta_local': 1.0
            })
            local_factor_rates_df = pd.DataFrame({
                'MSOA': msoa_list,
                'factor_rate': 1.0
            })

        # Attach local susceptibility multiplier to each super area object in the domain
        for super_area in domain.super_areas:
            super_area.local_susceptibility = local_susceptibility_df[local_susceptibility_df['MSOA'] == super_area.name]['local_susceptibility'].values[0]

        # ------------------------------------------------ #
        #          Build ingredients for simulator         #
        # ------------------------------------------------ #

        ## 1. Leisure
        leisure = generate_leisure_for_config(domain, CONFIG_PATH)

        ## 2. Epidemiology components

        ### a. Infection selector
        # InfectionSelector determines which exposed individuals become infected and
        # their health trajectory. local_factor_rates_df scales hospitalisation/death
        # probabilities by MSOA-level deprivation (from settings_[]_betas.csv).
        selector  = InfectionSelector.from_file(
            local_factor_rates_df=local_factor_rates_df
        )
        selectors = InfectionSelectors([selector])

        ### b. Infection seeds
        # InfectionSeed places initial cases in specified MSOAs on specified dates,
        # with a per-capita infection rate drawn from the seeds metadata CSV.
        # seed_past_infections=False means seeds are applied as future events only.
        infection_seed  = InfectionSeed.from_manual_setting(
            world=domain,
            infection_selector=selector,
            date_range=seeding_dates_i,
            super_areas_list=seeding_MSOA_i,
            cases_per_capita=seeds_strength_i,
            seed_past_infections=False,
        )
        infection_seeds = InfectionSeeds([infection_seed])

        ### c. Immunity setters (not used in this version)

        ### d. Medical policies (not used in this version)

        ### e. Vaccination campaigns (not used in this version)

        ## 3. Epidemiology: wraps selectors and seeds into a single object for the simulator
        epidemiology = Epidemiology(
            infection_selectors=selectors,
            infection_seeds=infection_seeds
        )

        ## 4. Interaction
        # global_beta_factor scales the overall transmission rate for this simulation
        # (drawn from the LHS in create_input_data.R).
        interaction = Interaction.from_file(
            config_filename=args.parameters,
            global_beta_factor=beta_factor_i
        )

        ## 5. Policies (loaded from default JUNE policy YAML — no interventions)
        policies = Policies.from_file(
            policy_file,
            base_policy_modules=("june.policy", "camps.policy")
        )

        ## 6. Events (default JUNE events file)
        events = Events.from_file()

        ## 7. Travel
        travel = Travel()

        ## 8. Tracker setup (optional — records contact matrices by venue type)
        # Build group_types and domainVenues dicts by checking which venue types
        # are present and non-empty in this domain. Only populated if --tracker is set.
        group_types  = []
        domainVenues = {}
        if domain.households is not None:
            if len(domain.households) > 0:
                group_types.append(domain.households)
                domainVenues["households"] = {
                    "N": len(domain.households),
                    "bins": domain.households[0].subgroup_bins,
                }
            else:
                domainVenues["households"] = {"N": 0, "bins": "NaN"}

        if domain.care_homes is not None:
            if len(domain.care_homes) > 0:
                group_types.append(domain.care_homes)
                domainVenues["care_homes"] = {
                    "N": len(domain.care_homes),
                    "bins": domain.care_homes[0].subgroup_bins,
                }
            else:
                domainVenues["care_homes"] = {"N": 0, "bins": "NaN"}

        if domain.schools is not None:
            if len(domain.schools) > 0:
                group_types.append(domain.schools)
                domainVenues["schools"] = {
                    "N": len(domain.schools),
                    "bins": domain.schools[0].subgroup_bins,
                }
            else:
                domainVenues["schools"] = {"N": 0, "bins": "NaN"}

        if domain.hospitals is not None:
            if len(domain.hospitals) > 0:
                group_types.append(domain.hospitals)
                domainVenues["hospitals"] = {"N": len(domain.hospitals)}
            else:
                domainVenues["hospitals"] = {"N": 0, "bins": "NaN"}

        if domain.companies is not None:
            if len(domain.companies) > 0:
                group_types.append(domain.companies)
                domainVenues["companies"] = {
                    "N": len(domain.companies),
                    "bins": domain.companies[0].subgroup_bins,
                }
            else:
                domainVenues["companies"] = {"N": 0, "bins": "NaN"}

        if domain.universities is not None:
            if len(domain.universities) > 0:
                group_types.append(domain.universities)
                domainVenues["universities"] = {
                    "N": len(domain.universities),
                    "bins": domain.universities[0].subgroup_bins,
                }
            else:
                domainVenues["universities"] = {"N": 0, "bins": "NaN"}

        if domain.pubs is not None:
            if len(domain.pubs) > 0:
                group_types.append(domain.pubs)
                domainVenues["pubs"] = {
                    "N": len(domain.pubs),
                    "bins": domain.pubs[0].subgroup_bins,
                }
            else:
                domainVenues["pubs"] = {"N": 0, "bins": "NaN"}

        if domain.groceries is not None:
            if len(domain.groceries) > 0:
                group_types.append(domain.groceries)
                domainVenues["groceries"] = {
                    "N": len(domain.groceries),
                    "bins": domain.groceries[0].subgroup_bins,
                }
            else:
                domainVenues["groceries"] = {"N": 0, "bins": "NaN"}

        if domain.cinemas is not None:
            if len(domain.cinemas) > 0:
                group_types.append(domain.cinemas)
                domainVenues["cinemas"] = {
                    "N": len(domain.cinemas),
                    "bins": domain.cinemas[0].subgroup_bins,
                }
            else:
                domainVenues["cinemas"] = {"N": 0, "bins": "NaN"}

        if domain.gyms is not None:
            if len(domain.gyms) > 0:
                group_types.append(domain.gyms)
                domainVenues["gyms"] = {
                    "N": len(domain.gyms),
                    "bins": domain.gyms[0].subgroup_bins,
                }
            else:
                domainVenues["gyms"] = {"N": 0, "bins": "NaN"}

        if domain.city_transports is not None:
            if len(domain.city_transports) > 0:
                group_types.append(domain.city_transports)
                domainVenues["city_transports"] = {"N": len(domain.city_transports)}
            else:
                domainVenues["city_transports"] = {"N": 0, "bins": "NaN"}

        if domain.inter_city_transports is not None:
            if len(domain.inter_city_transports) > 0:
                group_types.append(domain.inter_city_transports)
                domainVenues["inter_city_transports"] = {
                    "N": len(domain.inter_city_transports)
                }
            else:
                domainVenues["inter_city_transports"] = {"N": 0, "bins": "NaN"}

        # ------------------------------------------------ #
        #              Build tracker                       #
        # ------------------------------------------------ #
        if args.tracker:
            tracker = Tracker(
                world=domain,
                record_path=args.save_path,
                group_types=group_types,
                load_interactions_path=args.parameters,
                contact_sexes=["unisex", "male", "female"],
                MaxVenueTrackingSize=100000,
            )
        else:
            tracker = None

        # ------------------------------------------------ #
        #              Build simulator                     #
        # ------------------------------------------------ #
        # Assemble all components into the JUNE Simulator.
        # waning_factor sets the per-day immunity decay for this simulation.
        # override_initial_day sets the calendar start date from the seeds metadata.
        simulator = Simulator.from_file(
            world=domain,
            policies=policies,
            events=events,
            interaction=interaction,
            leisure=leisure,
            travel=travel,
            epidemiology=epidemiology,
            config_filename=CONFIG_PATH,
            record=record,
            tracker=tracker,
            waning_factor=waning_factor_i,
            override_initial_day=start_date_i,
        )

        return simulator

    # ------------------------------------------------ #
    #              Run simulator                       #
    # ------------------------------------------------ #

    print(f"mpi_rank {mpi_rank} generate simulator")
    simulator = generate_simulator()
    simulator.run()

    # ------------------------------------------------ #
    #              Combine MPI records                 #
    # ------------------------------------------------ #
    # After the run, each MPI rank has written its own partial .h5 record file.
    # Rank 0 merges them into a single june_record.h5 before reading output tables.

    mpi_comm.Barrier()

    if mpi_rank == 0:
        combine_records(args.save_path)

    mpi_comm.Barrier()

    # ------------------------------------------------ #
    #              Tracker post-processing             #
    # ------------------------------------------------ #
    # If the tracker was enabled, contract contact matrices to age bins and merge
    # results across MPI ranks. Only runs on rank 0 after the barrier.

    if args.tracker:
        if mpi_rank == 0:
            print("Tracker stuff now")

        simulator.tracker.contract_matrices("AC", np.array([0, 18, 100]))
        simulator.tracker.contract_matrices(
            "Paper",
            [0, 5, 10, 13, 15, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100],
        )
        simulator.tracker.post_process_simulation(save=True)

        mpi_comm.Barrier()

        if mpi_rank == 0:
            print("Combine Tracker results")
            Merger = MergerClass(record_path=args.save_path)
            Merger.Merge()

    # ------------------------------------------------ #
    #              Export output tables                #
    # ------------------------------------------------ #
    # Read the combined .h5 record and export each event table to a .feather file.
    # Feather is faster to read in R than CSV and much lighter than the raw .h5.
    # with_people=False and with_geography=False keep only the event-level columns
    # (timestamps, IDs) without joining individual or geographic attributes here —
    # those joins are done in process_output.R via the population/geography tables.

    save_path = str(args.save_path)
    read = RecordReader(save_path)

    infections_df = read.get_table_with_extras(table_name="infections", index="infected_ids", with_people=False,
                                               with_geography=False)
    infections_df.reset_index().to_feather(save_path + "/infectedTable.feather")

    recoveries_df = read.get_table_with_extras(table_name="recoveries", index="recovered_person_ids", with_people=False,
                                               with_geography=False)
    recoveries_df.reset_index().to_feather(save_path + "/recoveredTable.feather")

    deaths_df = read.get_table_with_extras(table_name="deaths", index="dead_person_ids", with_people=False,
                                           with_geography=False)
    deaths_df.reset_index().to_feather(save_path + "/deathsTable.feather")

    hospital_admissions_df = read.get_table_with_extras(table_name="hospital_admissions", index="patient_ids",
                                                        with_people=False, with_geography=False)
    hospital_admissions_df.reset_index().to_feather(save_path + "/hospitalAdmissionsTable.feather")

    icu_admissions_df = read.get_table_with_extras(table_name="icu_admissions", index="patient_ids", with_people=False,
                                                   with_geography=False)
    icu_admissions_df.reset_index().to_feather(save_path + "/ICUAdmissionsTable.feather")

    discharges_df = read.get_table_with_extras(table_name="discharges", index="patient_ids", with_people=False,
                                               with_geography=False)
    discharges_df.reset_index().to_feather(save_path + "/dischargesTable.feather")

    regions_df = read.get_table_with_extras(table_name="regions", index="id", with_people=False, with_geography=False)
    regions_df.reset_index().to_feather(save_path + "/regionsTable.feather")

    locations_df = read.get_table_with_extras(table_name="locations", index="id", with_people=False,
                                              with_geography=False)
    locations_df.reset_index().to_feather(save_path + "/locationsTable.feather")

    areas_df = read.get_table_with_extras(table_name="areas", index="id", with_people=False, with_geography=False)
    areas_df.reset_index().to_feather(save_path + "/areasTable.feather")

    superAreas_df = read.get_table_with_extras(table_name="super_areas", index="id", with_people=False,
                                               with_geography=False)
    superAreas_df.reset_index().to_feather(save_path + "/superAreasTable.feather")

    population_df = read.get_table_with_extras(table_name="population", index="id", with_people=False,
                                               with_geography=False)
    population_df.reset_index().to_feather(save_path + "/populationTable.feather")

    symptoms_df = read.get_table_with_extras(table_name="symptoms", index="infected_ids", with_people=False,
                                             with_geography=False)
    symptoms_df.reset_index().to_feather(save_path + "/symptomsTable.feather")

    # Delete the raw .h5 record files — they are large and no longer needed
    # once the feather tables have been written.
    file1_path = Path(save_path + "/june_record.0.h5")
    file2_path = Path(save_path + "/june_record.h5")
    if file1_path.exists():
        file1_path.unlink()
        print("File june_record.0.h5 removed successfully.")
    else:
        print("File june_record.0.h5 not found.")
    if file2_path.exists():
        file2_path.unlink()
        print("File june_record.h5 removed successfully.")
    else:
        print("File june_record.h5 not found.")


def worker(iii, sims_metadata_df, sims_seeds_df, sims_local_susceptibility_df, args):
    """
    Thin wrapper that runs run_iteration() in a separate subprocess (via mp.Pool).
    Running each simulation in its own process ensures that JUNE's memory is fully
    released between simulations — Python's garbage collector does not reliably free
    C/Fortran extensions held by JUNE objects.
    """
    run_iteration(iii, sims_metadata_df, sims_seeds_df, sims_local_susceptibility_df, args)


def main():
    # ------------------------------------------------ #
    #                  Parse arguments                 #
    # ------------------------------------------------ #

    parser = argparse.ArgumentParser(description="Full run of the England")
    parser.add_argument("-w", "--world_path", help="path to saved world file", required=False, )
    parser.add_argument("-con", "--config", help="Config file", required=False, )
    parser.add_argument("-p", "--parameters", help="Parameter file", required=False, )
    parser.add_argument("-tr", "--tracker", help="Activate Tracker for CM tracing", required=False, default="False", )
    parser.add_argument("-s", "--save_path", help="Path of where to save logger", required=False, )

    # ------------------------------------------------ #
    #               Default values for local runs      #
    # ------------------------------------------------ #
    # These defaults are used when running interactively (not from the HPC Slurm script).
    # On the HPC, all of these are overridden via command-line arguments.

    default_output_folder_prefix = "results_BORRAR"
    default_sim_metadata_file    = "settings_20250813_Gateshead_sims"
    default_seeds_metadata_file  = "settings_20250813_Gateshead_seeds"
    default_betas_metadata_file  = None
    default_world_file           = "Gateshead"

    parser.add_argument('--run_index',            required=False, type=int, default=0)
    parser.add_argument('--tot_index',            required=False, type=int, default=1)
    parser.add_argument('--fol_index',            required=False, type=int, default=None)
    parser.add_argument('--max_iter',             required=False, type=int, default=None)
    parser.add_argument('--server',               required=False, default="PC")
    parser.add_argument('--world_file',           required=False, default=default_world_file)
    parser.add_argument('--output_folder_prefix', required=False, default=default_output_folder_prefix)
    parser.add_argument('--config_file',          required=False, default="config_simple_days")
    parser.add_argument('--sim_metadata_file',    required=False, default=default_sim_metadata_file)
    parser.add_argument('--seeds_metadata_file',  required=False, default=default_seeds_metadata_file)
    parser.add_argument('--betas_metadata_file',  required=False, default=default_betas_metadata_file)
    args = parser.parse_args()

    # ------------------------------------------------ #
    #              Resolve paths by server             #
    # ------------------------------------------------ #
    # local_folder:  root of the JUNE repository (config files, world files, metadata CSVs)
    # local_folder2: root for saving simulation outputs (can be a scratch/work filesystem)

    if args.server == "Brahms":
        args.local_folder  = "/localhome/laurag/JUNE/"
        args.local_folder2 = "/scratch/GENIE/RESULTS_JUNE/"
    elif args.server == "HPC":
        args.local_folder  = "/home/lmg65/rds/JUNE/"
        args.local_folder2 = "/rds-d6/user/lmg65/hpc-work/"
    else:
        args.local_folder  = "/Users/laura/Documents/Github/JUNE4GENIE/"
        args.local_folder2 = "/Users/laura/Downloads/temp/"

    # fol_index distinguishes output folders for different Slurm array tasks;
    # defaults to run_index if not explicitly provided
    if args.fol_index is None:
        args.fol_index = args.run_index

    args.input_world_path = "worlds/"
    args.input_world_file = args.world_file
    args.output_path      = "results_hub/" + args.output_folder_prefix
    args.config_file      = args.config_file + ".yaml"

    # ------------------------------------------------ #
    #              Load metadata CSVs                  #
    # ------------------------------------------------ #
    # Load all three settings files produced by create_input_data.R.
    # betas_metadata_file is optional; if absent, uniform local parameters are used.

    file_sim_metadata    = args.local_folder + "data_simulations/" + args.sim_metadata_file + ".csv"
    sims_metadata_df     = pd.read_csv(file_sim_metadata)

    file_seeds_metadata  = args.local_folder + "data_simulations/" + args.seeds_metadata_file + ".csv"
    sims_seeds_df        = pd.read_csv(file_seeds_metadata)
    sims_seeds_df        = sims_seeds_df.sort_values(by='seed')

    sims_local_susceptibility_df = None
    args.use_local_betas = args.betas_metadata_file is not None
    if args.use_local_betas is True:
        file_betas_metadata          = args.local_folder + "data_simulations/" + args.betas_metadata_file + ".csv"
        sims_local_susceptibility_df = pd.read_csv(file_betas_metadata)

    # Resolve full paths to the world file and JUNE config/parameter YAMLs
    args.world_path  = args.local_folder + args.input_world_path + args.input_world_file + ".hdf5"
    args.config      = paths.configs_path / args.config_file
    args.parameters  = paths.configs_path / "defaults/interaction/interaction.yaml"

    # ------------------------------------------------ #
    #                   Simulation loop                #
    # ------------------------------------------------ #
    # Set a global random seed for reproducibility across the batch, then iterate
    # over the simulations assigned to this array task.
    # With tot_index=N and run_index=k, this task runs simulations k, k+N, k+2N, ...
    # so the workload is evenly striped across all array tasks.

    set_random_seed(0)
    if args.max_iter is None:
        # Run all simulations defined in the metadata file
        num_simulations = len(sims_metadata_df["sim"].unique())
    else:
        # Optionally cap the number of simulations (useful for testing)
        num_simulations = args.max_iter

    for ii in range(args.run_index, num_simulations, args.tot_index):
        # Each simulation is run in a fresh subprocess so that JUNE's memory
        # (C extensions, large numpy arrays) is fully released after each run.
        with mp.Pool(processes=1) as pool:
            pool.apply(worker, args=(ii, sims_metadata_df, sims_seeds_df, sims_local_susceptibility_df, args,))


if __name__ == "__main__":
    mp.freeze_support()  # Required on Windows for multiprocessing to work correctly
    main()
