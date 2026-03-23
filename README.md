# JUNE Model — Modified Fork

This repository is a modified fork of the [JUNE epidemiological modelling framework](https://github.com/IDAS-Durham/JUNE). The changes described below extend JUNE with new features required for the GENIE project, including waning immunity, local susceptibility, super-area seeding, and several bug fixes and calibration tools.

---

## Summary of Changes

### 1. Waning Immunity

Adjusts the degree to which immunity decays over time after recovery.

- **Recovery → Susceptible transition**: When an individual recovers from an infection, they return to the susceptible state.
  - `Epidemiology.recover()` — `Epidemiology.py` L190

- **Per-timestep immunity decay**: Susceptibility increases at every time step for individuals who are neither dead nor currently infected. Decay applies across all previously contracted infections, stored in `immunity.susceptibility_dict`.
  - `Epidemiology.update_health_status()` — `Epidemiology.py` L201, L249–253

- **Waning factor**: Immunity decreases by a `waning_factor`, retrieved via the `Simulator`. New calls were added to several core functions.
  - `Simulator.__init__()` — `Simulator.py` L84, L121
  - `Simulator.from_file()` — `Simulator.py` L137, L184
  - `Simulator.do_timestep()` — `Simulator.py` L351
  - `Epidemiology.do_timestep()` — `Epidemiology.py` L101, L132

---

### 2. Override Initial Simulation Day

The initial day specified in the configuration file can now be overridden at runtime. It is passed as the `override_initial_day` input to the `Simulator` and modifies the `Time` attribute `initial_day`.

- `Simulator.from_file()` — `Simulator.py` L138, L160–163
- `Time.from_file()` — `Time.py` L58, L67–72, L74

---

### 3. Local Susceptibility

In the JUNE model, individuals within a subgroup interact with each other and JUNE calculates effective transmission rates per subgroup. These rates are now multiplied by a **local susceptibility factor** that varies by an individual's residential super area, stored within each `super_area` object.

- **Attribute added to super areas**: `local_susceptibility` is initialised in `run_simulation.py`.
  - `SuperArea.__slots__` — `geography.py` L182

- **Factor applied during interaction**: Local susceptibility is applied as a multiplier to effective transmission rates within each subgroup, read directly from the group's super area attribute.
  - `Interaction.time_step_for_group()` — `interaction.py` L195
  - `Interaction._time_step_for_subgroup()` — `interaction.py` L216–220, L243

---

### 4. Global Interaction Factor (Beta)

The interaction factor `beta` (as defined in the JUNE paper) can now be scaled by a `global_beta_factor` supplied as an input to the `Interaction` constructor.

- `Interaction.from_file()` — `Interaction.py` L47–49, L53–54

---

### 5. Bug Fix: Health Index Probability Calculations

`_set_probability_per_age_bin()` has been corrected to align with the specifications in the JUNE paper. While the effect on outputs is minimal, this resolves an existing calculation bug in the original code.

- `HealthIndexGenerator._set_probability_per_age_bin()` — `health_index.py` L197–203, L219–222

---

### 6. Super-Area-Specific Health Index

JUNE calculates the health index from rates in an external file. This fork modifies that process to multiply severe outcome rates by a factor unique to each super area, loaded from a data frame.

- **Input handling**: Added the `factor_rates_df` data frame as an input to `InfectionSelector` and `HealthIndexGenerator`.
  - `InfectionSelector.from_file()` — `infection_selector.py` L53, L64–65, L73–74
  - `HealthIndexGenerator.__init__()` — `health_index.py` L37
  - `HealthIndexGenerator.from_file()` — `health_index.py` L81, L94

- **Attribute storage**: Added `factor_per_location` (dictionary) to `HealthIndexGenerator` attributes.
  - `HealthIndexGenerator.__init__()` — `health_index.py` L61

- **New functions**: `_get_probabilities_by_sa()` and `_set_probability_per_age_bin__by_sa()` are customised versions of the originals `_get_probabilities()` and `_set_probability_per_age_bin()`. They return a probability dictionary indexed by super area.
  - `HealthIndexGenerator._set_probability_per_age_by_sa()` — `health_index.py` L244–304
  - `HealthIndexGenerator._get_probabilities_by_sa()` — `health_index.py` L305–337

- **Implementation**: `HealthIndexGenerator` now calls these new functions.
  - `HealthIndexGenerator.__init__()` — `health_index.py` L37, L62–63
  - `HealthIndexGenerator.__call__()` — `health_index.py` L144–145

---

### 7. Super-Area Seeding from File

Shifts the infection seeding mechanism from regions to super areas, and adds support for manual seed specification from file.

- **Seeding at super-area level**: `daily_cases_per_capita_per_age_per_region` has been replaced by `daily_cases_per_capita_per_age_per_super_area`.
  - `InfectionSeed.__init__()` — `infection_seed.py` L34, L59, L61, L66, L292, L296–297, L301–302

- **Manual seeding via `from_manual_setting()`**: Constructs an `InfectionSeed` object from three equal-length vectors: super area, date, and seed strength (cases per capita). All manual seeds occur at 09:00.
  - `InfectionSeed.__init__()` — `infection_seed.py` L79–119

- **Infection guarantees**: `infect_super_area()` now ensures at least one person is infected even when seed strength is too low to trigger an infection under standard probability calculations.
  - `InfectionSeed.infect_super_area()` — `infection_seed.py` L145–148, L153–156, L164–174

- **New v2 function**: `infect_super_areas_v2()` incorporates the above changes and omits past-infection seeding (not used in GENIE).
  - `InfectionSeed.infect_super_areas_v2()` — `infection_seed.py` L236–268, L304–305

- **Removed functions**: `_parse_input_dataframe()`, `from_global_age_profile()`, and `from_uniform_cases()` were removed as they are incompatible with the new super-area structure.

- **Unadapted functions**: `_seed_past_infections()`, `_need_to_seed_accounting_secondary_infections()`, and `_adjust_seed_accounting_secondary_infections()` were not modified as they are not used in GENIE.

---

## Original Repository

This fork is based on [JUNE](https://github.com/IDAS-Durham/JUNE). Please refer to the original repository for full documentation, installation instructions, and the JUNE paper.