# Prey switching in the Shiny app: design

Date: 2026-11-17. Base: `main` at ac6bac1. Revised after a review against the code the same day.

## Intent

The user runs their own models in the app, mostly balanced there and sometimes imported from an EwE database. They want chosen predators to shift their diet towards prey that become more abundant: EwE's prey switching, set predator by predator.

The core already does this. `apply_prey_switching` and `prey_switch_weights` (`core/ecosim_deriv.py`) implement EwE's `SetRelaSwitch` formula from each group's `SwitchPower`:

- `_run_setup` (`core/ecosim.py`) builds the search-rate weights at the start of every run, so setting `params.SwitchPower` after a scenario is built takes effect;
- the switching acts on RK4, AB and the continuous methods, and in Ecospace on the map-wide base, as EwE does;
- `tests/test_prey_switching.py` pins it;
- LT2022, one of the models compared against EwE's own output, carries `SwitchPower` 1 on the long-tailed duck.

What is missing is the app:

- **A model balanced in the app** gets a scenario whose `SwitchPower` is 0 for every group (`rsim_params`), and no control changes it.
- **The "EwE" scenario source** runs with its own values: a scenario from an imported EwE database, or the Ecosim settings of an example opened from the Home page (`example_ecosim_scenario`, which reads `switch_power` from the EwE-F files). The app neither shows nor changes those values.
- **Diet rewiring**, PyPath's own feature, never integrated (PR #125, `deviations.md` row DR). Its controls are hidden behind a note saying prey switching is planned as the replacement. Several pages still describe it.

## Goals

1. **A per-predator control on the Ecosim page.** It sets each consumer's switching power, for both scenario sources ("From the balanced model" and "EwE": an imported database scenario or an example).
2. **Balanced route.** Every consumer of the model is listed, starting at 0 (off), as soon as a balanced model is loaded.
3. **EwE route.** The control shows the values of the scenario that source built. Values changed in the app replace them for the run. The source's note, which today says that edits made in this app are not used, says that switching edits are used. That holds for both the database note and the example note.
4. **When values apply.** They take effect when the scenario is created (Create Scenario), the way the vulnerability slider works today. On the EwE route the control fills with the source's values when the scenario is created.
   - While the control's values differ from the current scenario's, a note under it says they apply when the scenario is created again.
   - So before every run the user can see which values the run will use.
5. **Every run shows its switching, for every method.**
   - **Ecosim page.** It shows a line such as "Prey switching: Cod 1.0, Seals 0.5" or "Prey switching: none" with every run's results: RK4, AB and the continuous methods alike. The line sits outside the solver-diagnostics card, which appears only for continuous runs.
   - **Ecospace and IBM pages.** They run the Ecosim page's scenario (`_sim_scenario`), so its switching applies there too. Each shows the same line with its run.
   - **Where the values come from.** The line is built when the result lands, from the scenario that ran (`outcome.source.params.SwitchPower` on the Ecosim page), never from the control. It therefore includes imported values that were not edited.
6. **Accurate wording everywhere diet rewiring or switching is described:**
   - the Ecosim sidebar note and help text ("prey switching is planned");
   - the home page's "Diet Rewiring" card and its "Diet rewiring (prey switching)" feature line;
   - `about.py`'s "Diet rewiring (prey switching)";
   - `forcing_demo.py` (the "prey switching is planned" text and comment);
   - the autofix help label "DD (Prey Switching)" and its tooltip on the Ecosim page. DD is Rpath's handling-time parameter, not EwE's switching, and it would sit next to the new control. The builder confirms DD's meaning in the core before renaming it.
   - The diet-rewiring demo page is rewritten to show EwE's switching, computed with the core's own function, not the `DietRewiring` class.

## Non-goals

- Any change to the switching formula or to the core run.
- Switching per link, or switching that varies over time.
- A scenario export. The app's downloads are the Ecopath model and the results; the core's EwE-compatible CSV writer already writes `SwitchPower`, but the app does not offer it.
- Recording switching in the downloads. The simulation CSV is a bare table (Month plus one biomass column per group), and a comment line or extra column would break every script that reads it. A run-settings download is left for later.
- Reimplementing `DietRewiring` or mapping it onto switching. Its semantics (an update interval, a minimum diet proportion, scaling by the mean prey biomass) do not map onto EwE's.
- A switching control on the Ecospace or IBM page; they use the Ecosim scenario's.

## Anti-goals (failure even if it works)

- **A run whose switching the user cannot tell from what the page shows.** This covers any method and the Ecospace and IBM runs.
- **Edits on the EwE route silently ignored.**
- **Results changing for anyone who does not touch the control.** All values left at 0, or an EwE scenario left as built, must give today's results to the bit. The parity suite and the parity capture (139 arrays) stay identical.
- **A control offered for a group that cannot switch.**

## Constraints

- **Which groups switch.**
  - EwE sets a non-editable flag on `SwitchingPower` for pure producers (`sPP = 1`, `cCore.vb` around line 5508). Detritus stays editable in EwE, but switching acts on a predator's search rates and detritus eats nothing.
  - In the scenario's parameters the test is `PP_type == 0`. `rsim_params` casts the type to an integer, so partial producers are 0, producers 1, and detritus and fleets 2.
- **Range.** Values must be ≥ 0, and 0 means off. EwE's documented range is 0 to 2, which the local EwE sources do not confirm (they hold no variable metadata).
  - The builder confirms it from EwE's variable metadata or user guide. If confirmed, typed values outside it are refused with a message. If not, values ≥ 0 are accepted with a warning above 2. *(my call)*
  - Imported values outside the range are kept and warned about.
  - Two sets of constants must not be reused. `*_PREY_SWITCHING_POWER` in `core/constants.py` belongs to `autofix`'s DD handling. `PARAM_RANGES.switching_power_*` in the app's `config.py` is the old demo's range, 1 to 5, whose minimum would forbid 0.
- **The repository's rules:**
  - app tests go in `packages/pypath-shiny/tests/`;
  - CI's format gate is ruff format 0.15.x;
  - the Shiny smoke check must be green before merge;
  - main auto-deploys, so merging waits for the user's word;
  - never run `scripts/deploy.sh`.
- **Shiny versions.** 1.9.0 in the server environment at the time of writing; the builder confirms what the deployed app runs. Anything used must work there; the app's background runs were checked on 1.8.0.
- **Continuous methods** need no change: the run builds its parameter dict, `SwitchA` included, from `scenario.params.SwitchPower`, and the continuous right-hand side copies that dict.

## Decided by the user

- Switching is set in the app, on the user's own models.
- Both scenario routes are used, the balanced route most.
- Switching is set predator by predator.
- The behaviour played back and accepted: a sidebar control, applied at Create Scenario, EwE values shown and overridable, the run's switching shown, and the old wording replaced. Two of my guesses have since changed:
  - The playback had the values kept "when you export the model". The app has no scenario export, so recording switching in downloads is now a non-goal.
  - Goal 5 now covers every method and the Ecospace and IBM runs.

## My calls

- **Placement.** The new control replaces the sidebar's diet-rewiring note in the same place, with the other scenario controls (fishing, recruitment, metabolism).
- **Overrides.** The control keeps them on top of the source's own values: 0 on the balanced route, the scenario's on the EwE route.
  - An override exists only for a value the user edited. An untouched value never writes, so a rounded display cannot change an imported full-precision value.
  - Editing a value back to the source's value, as displayed, removes its override, and the source's full-precision value is used.
  - With no override, Create Scenario leaves `params.SwitchPower` exactly as the source built it.
- **Overrides per source.** There is one set for the balanced route and one for each EwE scenario or example. Switching routes discards nothing, and a value never carries over to an unrelated scenario.
- **On a model change** (`model_data.set`, including each re-balance):
  - balanced-route overrides are kept for groups that are still present and still consumers, and dropped otherwise;
  - EwE-route overrides are dropped when the source no longer matches the model;
  - the existing model-change notice says which overrides were kept.
- **On the EwE route the scenario's own groups drive the control.** Their names and the producer test come from the scenario's own parameters, not the app's loaded model.
  - A group that cannot switch is not editable.
  - A nonzero imported value on such a group is kept and shown read-only with a note.
- **The run record for Goal 5** is built at landing from the scenario that ran. The scenario-controls metadata (`build_control_metadata`, written to `output.params["applied_controls"]`) may carry the same values, but it is read nowhere today and is not the visible record.
- **Deprecation.** `DietRewiring` warns with a `DeprecationWarning`, pointing to `SwitchPower`, only when constructed with `enabled=True`, and `create_diet_rewiring` always warns. Nothing is removed.
  - `create_advanced_scenario` builds a disabled instance internally, and `rsim_run_advanced`'s message tells users to pass one. Both stay silent.
  - `rsim_run_advanced` keeps raising `NotImplementedError` for enabled diet rewiring.
- **`deviations.md` row DR** changes from "Next: diet rewiring maps onto prey switching" to a statement that the app sets `SwitchPower` and diet rewiring is deprecated.

## Left to the builder

- **The widget:** an editable table (group, switching power) or one numeric input per consumer. It should stay usable with 40 or more consumers.
- **Help text,** including what a power of 0, 1 and 2 does.
- **The demo page's design.** It must use `apply_prey_switching` or `prey_switch_weights` from the core, and state that it illustrates the formula on a small example.
- **The exact wording and position** of the switching line on the three pages.
- **Tests.** At least:
  - the control's values reach `scenario.params.SwitchPower` on the balanced route, the database route and the example route;
  - values left at 0 leave a run identical to today's;
  - an EwE scenario or example keeps its values to the bit unless they are changed, with a value such as 0.30000001 surviving a rounded display;
  - consumers (`PP_type == 0`) are offered, and producers, detritus and fleets are not;
  - the switching line appears for RK4, AB and a continuous run, and on the Ecospace and IBM pages;
  - re-balancing keeps the overrides of groups that remain.
