# Prey switching in the Shiny app: design

Date: 2026-11-17. Base: `main` at ac6bac1.

## Intent

The user runs their own models in the app, mostly balanced there and sometimes imported from an EwE database. They want chosen predators to shift their diet towards prey that become more abundant: EwE's prey switching, set predator by predator.

The core already does this. `apply_prey_switching` and `prey_switch_weights` (`core/ecosim_deriv.py`) implement EwE's `SetRelaSwitch` formula from each predator's `SwitchPower`:

- it runs on every method (RK4, AB and the continuous methods), and in Ecospace on the map-wide base, as EwE does;
- `tests/test_prey_switching.py` pins it;
- LT2022, one of the models compared against EwE's own output, carries `SwitchPower` 1 on the long-tailed duck.

What is missing is the app:

- **A model balanced in the app** gets a scenario whose `SwitchPower` is 0 for every group (`rsim_params`), and no control changes it.
- **An imported EwE scenario** runs with the database's values, but the app neither shows nor changes them.
- **Diet rewiring**, PyPath's own feature, never integrated (PR #125, `deviations.md` row DR). Its controls are hidden behind a note saying prey switching is planned as the replacement. The home page and the diet-rewiring demo page still describe it.

## Goals

1. **A per-predator control on the Ecosim page.** It sets each group's switching power and works for both scenario sources ("From the balanced model" and the imported EwE scenario).
2. **Balanced route.** Every group whose switching power EwE lets you set is listed, starting at 0 (off).
3. **EwE route.** The control shows the database scenario's values. Values changed in the app replace them for the run, and the page says so.
4. **When values apply.** They take effect when the scenario is created (Create Scenario), the way the vulnerability slider works today. The user can see before running which values the next run will use.
5. **A run records its switching.** A run's model options (the solver diagnostics panel's line and the run's stored settings) list the predators that switch and their powers, so a result cannot be mistaken for one without switching.
6. **Accurate wording.**
   - The two "prey switching is planned" notes on the Ecosim page (the sidebar note and the help text) are replaced by the control and a short description.
   - The home page's "Diet Rewiring" card and its "Diet rewiring (prey switching)" feature line describe prey switching as it is now.
   - The diet-rewiring demo page is rewritten to show EwE's switching, computed with the core's own function, not the `DietRewiring` toy class.
7. **Results download.** The simulation download records the switching powers used.

## Non-goals

- Any change to the switching formula or to the core run. The formula is EwE's and already tested.
- Switching per link, or switching that varies over time.
- A scenario export from the app. Today the app downloads the Ecopath model and the results, not the Ecosim scenario. The core's EwE-compatible CSV writer already writes `SwitchPower`, but the app does not offer it.
- Reimplementing `DietRewiring` or mapping it onto switching. Its semantics (an update interval, a minimum diet proportion, scaling by the mean prey biomass) do not map onto EwE's.
- A separate control on the Ecospace page.

## Anti-goals (failure even if it works)

- A run whose switching the user cannot tell from its output.
- Edits on the EwE route silently ignored. The page already says that edits made in the app are not used on that route; for switching it must say the opposite, and the run must honour it.
- Results changing for anyone who does not touch the control. All values left at 0, or an EwE scenario left as imported, must give today's results exactly. The parity suite and the parity capture (139 arrays) stay identical.
- Offering the control for a group that EwE does not let switch.

## Constraints

- **Which groups switch.** EwE sets a non-editable flag on `SwitchingPower` for pure producers (`sPP = 1`, `cCore.vb` around line 5508); the switching acts on a predator's search rates, so it means nothing for detritus. Consumers, including partial producers, get the control.
- **Range.** Values must be ≥ 0, and 0 means off. EwE's documented range is 0 to 2.
  - The builder confirms the range from EwE's variable metadata or the EwE user guide.
  - If it cannot be confirmed: accept values ≥ 0 and warn above 2. *(my call)*
  - The PyPath constants named `*_PREY_SWITCHING_POWER` (`core/constants.py`) belong to `autofix`'s DD handling, not to EwE's `SwitchPower`; do not reuse them.
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
- The behaviour described under Goals 1-6 was played back and accepted. Goal 7 replaces a wrong guess about exporting the model.

## My calls

- Values apply at Create Scenario, not at Run, matching the vulnerability slider. This is a guess the user accepted.
- An edit made after the scenario exists does not apply until the scenario is created again. The control should make that visible: a note under it, or the same treatment the page gives other settings that need re-creation.
- The new control replaces the sidebar's diet-rewiring note in the same place, so it sits with the other scenario controls (fishing, recruitment, metabolism).
- `DietRewiring` and `create_diet_rewiring` (public in `pypath.core.forcing`) emit a `DeprecationWarning` that points to `SwitchPower`; nothing is removed. `rsim_run_advanced` keeps raising `NotImplementedError` for enabled diet rewiring; its message already points to `SwitchPower`.
- `deviations.md` row DR moves from "Next: diet rewiring maps onto prey switching" to a statement that the app sets `SwitchPower` and diet rewiring is deprecated.

The builder check raised the next five; they are settled here:

- **Only changed values are stored.** The control keeps overrides, the groups the user changed and their values, on top of the source's own values: 0 on the balanced route, the database's on the EwE route.
  - With no override, Create Scenario leaves `params.SwitchPower` exactly as the source built it, so an untouched scenario gives today's results to the bit.
  - Setting a value back to the source's value removes its override.
- **Overrides are kept per source.** There is one set for the balanced route and one for each EwE scenario, kept for the session, so a value never carries over to an unrelated scenario and switching routes discards nothing. Loading a model clears them, as it already clears results and the scenario.
- **On the EwE route the scenario's own groups drive the control.** Their names and the producer test come from the scenario's own parameters, not the app's loaded model, because the run uses those parameters.
  - A group EwE does not let switch is not editable.
  - A nonzero imported value on such a group is kept and shown read-only with a note, so the imported scenario still runs as imported.
- **Goals 5 and 7 record what the run used:** the switching powers in the run's scenario, never the control's pending values. When the control holds values that differ from the current scenario's, it says they apply after the scenario is created again.
- **Range.** If EwE's range is confirmed, values typed in the control are limited to it, and a value outside is refused with a message. Imported values outside it are kept and warned about.
- **Where the run's switching is stored.** It goes into the scenario's control metadata: `build_control_metadata` (`services/ecosim_controls.py`), `scenario.applied_controls_metadata`, then `output.params["applied_controls"]`. These feed the page and the one-way JSON results export; no saved format reloads them, so a new field changes no schema.

## Left to the builder

- **The widget:** an editable table (group, switching power) or one numeric input per consumer. Large models have tens of consumers, so the choice should stay usable with 40 or more.
- **How the EwE route shows the database's values before the scenario is created.** One way is to read the scenario's group settings when the scenario is chosen. If that is too costly, filling the control at Create Scenario is acceptable, provided the page says which values the run uses.
- **Help text,** including what a power of 0, 1 and 2 does.
- **The demo page's design.** It must use `apply_prey_switching` or `prey_switch_weights` from the core, and state that it illustrates the formula on a small example.
- **Where the run's switching is stored** for Goal 5, and the exact wording of the diagnostics line.
- **Tests.** At least:
  - the control's values reach `scenario.params.SwitchPower` on both routes;
  - values left at 0 leave a run identical to today's;
  - an EwE scenario keeps its imported values unless they are changed;
  - pure producers and detritus are not offered.
