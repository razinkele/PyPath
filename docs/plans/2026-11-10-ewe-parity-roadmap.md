# PyPath → EwE 6.7.144 Full Parity Roadmap

**Source:** `docs/reviews/2026-11-10-audit-ewe-6.7.144-review.md`, a markdown copy of the living review exported 2026-11-10 at 15:56. Severity and Status are as of that export; check the living doc (https://claude.ai/artifact/RNa2fghx34JudpCeSyaJ5n) for later changes. It covers all sections of `2026-11-10-audit-ewe-6.7.144.md` (§1-8).
**Baseline:** `origin/main` `bb1acbb` (pypath-ewe 1.3.0, pypath-shiny 1.3.0). Parity matrix: `docs/parity/ewe_parity_matrix.csv`, with 84 tables, 31 "implemented" and 53 "partial".

This is a roadmap, not an implementation plan. Each phase gets its own `writing-plans` plan when it starts. Phase 1 already has one.

## What "full parity" means here

PyPath can claim parity with EwE 6.7.144 when all four of these hold:

1. **Every finding is closed.** Each finding in the review (A1–G), each item in its "already known" table, and each A1 slice-2 item is either fixed, or listed in `docs/parity/deviations.md` as a documented deviation that you accepted.
2. **Every in-scope parity-matrix row is implemented.** Each row in `ewe_parity_matrix.csv` inside the agreed scope (decision D4) is `implemented`, with no unread result-changing columns. A test proves each row.
3. **Trajectories match an EwE oracle.** For each project database (LT2022 ×2, Edgaras, Black Sea, Curonian consolidated), Ecopath outputs match to 1e-5 relative. Ecosim monthly biomass and catch match an EwE run for at least 10 years to 1e-3 relative. These tolerances are a proposal. The oracle is decision D2.
4. **The unchecked areas are audited.** The review's "Not checked" list has had a second audit pass, and every finding from it is closed under rule 1.

Until rule 3 holds, phase exits are labelled **formula parity**: matching transcribed EwE code or EwE's stored values. **Trajectory parity**, matching an EwE run, is the stronger label.

## Decisions (recorded 2026-11-10)

All seven recommendations below were approved on 2026-11-10:

- **D1:** one global `compat="ewe"|"rpath"` switch, built in P2, the first phase that needs it.
- **D2:** a headless EwECore spike, falling back to your own exports. **Spike verdict (2026-11-10): feasible.** The unmodified EwE 6.7.144 core (tag `v6.7.144`) runs headless on a GitHub Actions Windows runner, in the private repo `razinkele/ewe-oracle`. Linux can't open `.eweaccdb` files, because EwE's Access layer uses OleDb/ACE. Exit criterion met: the LT2022 detritus export is 5366.089. LT2022 inputs and Ecopath match PyPath to single precision. First finding: EwE loads `UseVarPQ` as False (deviation R1).
- **D3:** oracles stay outside the repo, and only their numbers are committed; production code is reimplemented from the equations.
- **D4:** Tier 1 first, then Tier 2.
- **D5:** Ecotracer is relabelled.
- **D6:** yes / yes / yes.
- **D7:** left to you.

## Decisions needed before Phase 0 ends

| # | Decision | Options | Recommendation | Blocks |
|---|---|---|---|---|
| D1 | **EwE vs Rpath behaviour.** These are Rpath choices: D1 (prey Ftime), A2 (no detritus import), the plus-group weight (E) and the mixotroph (1 − PP) scaling (D4). About 20 test files pin Rpath reference output (`test_rpath_*`, `test_detritus_*`, …). | (a) one global `compat="ewe"\|"rpath"` switch; (b) a per-feature switch like the existing `fast_group_rule`; (c) drop Rpath parity where the two conflict | (a), defaulting to `"ewe"` for EwE imports and `"rpath"` for Rpath CSV models. The existing per-feature switches become aliases. | Phases 2, 3, 5 |
| D2 | **Oracle source** | (a) your own EwE 6 exports (scenario 16 for LT2022 is still pending); (b) a headless EwECore harness (Phase 0 spike); (c) formula parity only | Spike (b). Fall back to (a) for one or two scenarios. | Rule 3, all trajectory exits |
| D3 | **Licence.** EwE is EUPL-1.2, PyPath is MIT. Line-by-line ports (CalculateStanzaParameters, ConDeriv, the BDF2/SOR solver) and the transcribed oracle scripts may not be committable. | (a) keep oracles outside the repo and commit only the numbers they produce; (b) clean-room reimplementation from equations, with EwE line refs as citations only; (c) seek legal advice | (a) for oracles, (b) for production code | Every "move oracle into tests" step |
| D4 | **Target scope** | Tier 1: imported Ecopath + Ecosim + stanzas + fitting (A, C, D, E). Tier 2: Ecospace (B). Tier 3: Ecotracer, options, value chain (F, G, the 21 value_chain rows) | Commit to Tier 1, then Tier 2. Tier 3 by request. | Definition rule 2 |
| D5 | **Ecotracer (F)** | Port EwE's model, or relabel the module as a non-EwE tracer | Relabel now; port only if someone needs it | Phase 8 |
| D6 | **Questions carried over from the A1+A2 plan** | Defer OtherMort and migration (slice 2); accept the formula-derived bird EE pins; report the Edgaras out-of-range EE values without changing them | Yes / yes / yes | Phase 1 |
| D7 | **GPL-2.0 EwE-F Black Sea test data** in the MIT repo | Keep, move or remove | Your call | None |

## Phase overview

```
P0 Foundations ──► P1 Reader+detritus (A1,A2) ──┬─► P2 Stanzas (E) ──────────┐
                                                ├─► P3 Fitting (C1–C3) ──────┤
                                                └─► P4 Ecosim kernel (D1→D2→D3)┤
                                                                             ├─► P5 Ecopath solver & slice 2 (D4, A1-s2, G-Ecopath)
                                                                             ├─► P6 Ecospace (B4→B2, B1+B3, B5)
                                                                             ├─► P7 Ecosim options & fishing (G, known items)
                                                                             ├─► P8 Ecotracer (F)
                                                                             └─► P9 Second audit pass + parity claim
```

P2, P3 and P4 are independent of one another once P1 lands. P6 needs only P1, but its trajectory exit needs the P0 harness.

---

### P0: Foundations (small; no change to model results)

- **Covers:**
  - decisions D1–D5;
  - relabelling the SS 637.0 pin as a regression pin (C3);
  - correcting parity-matrix rows 28, 39, 45, 84 and 85 to `partial`, with notes on the unread columns;
  - a Shiny page banner and a docs note that Ecospace runs of imported models are not EwE-equivalent;
  - creating `docs/parity/deviations.md`.
- **Spike: headless EwE oracle.**
  - Build `Sources/EwECore/EwECore.vbproj` (v6.7.144) and adapt `Examples/2. Automation/2. PerturbatePathRunSim`. The goal is a console app: `ewe-oracle <db> <scenario> → ecopath.csv, ecosim_monthly.csv`.
  - **What's on this machine:** the .NET 10 SDK and VS2019, with the Access driver already used by the reader.
  - **Unknowns:** EwECore's target framework (probably .NET Framework 4.x), and whether it builds without the desktop UI projects.
  - **Timebox:** one day. The outcome is feasible (exit criterion: an LT2022 Ecopath export whose detritus Export is 5366.09) or not feasible, falling back to D2(a).
- **Oracle storage:** outside the repo, per D3(a). Only the CSV outputs get committed, under `packages/pypath/tests/data/ewe_oracle/<db>/`.
- **Exit:**
  - decisions recorded at the top of this file;
  - the matrix rows corrected;
  - the SS pin's docstring says "regression pin, not EwE SS";
  - spike verdict written down.
- **Moves no pins.**

### P1: Reader + detritus import (A1, A2) — plan written

- **Plan:** `docs/plans/2026-11-10-a1-a2-ba-detritus-import.md`.
- **Added to that plan:** reader warnings for every non-default setting PyPath ignores. These are RiskTime, QmQo, CmCo ≠ 1000, FishRateMax, AdditivePredMort, the nutrient settings, BABsplit, LeadingCB ≠ LeadingLifeStage, fractional Type, and the unread EcosimScenario fields (StepSize, SystemRecovery, sorwt, Discount). The warnings are cheap and stop silent errors before the later phases fix them.
- **Oracle:** EwE's stored detritus Export of 5366.09, plus the formula-derived bird EE values. This is formula parity.
- **Rpath:** no reference model has DetInput ≠ 0, so this phase has no Rpath effect.
- **Pins moved:** EwE-F LT2022 EE (Scoters, Long-tailed duck, Detritus) and SS16.

### P2: Multi-stanza groups (E)

- **Covers:**
  - read LeadingCB and anchor stanza consumption on it (`cEcoSimModel.vb:4180-4187`, `4384-4398`);
  - read BABsplit (the reader must supply `BAB`, which `stanzas.py:912` expects);
  - apply recruitment forcing after the RecPower exponent, with the HatchCode logic;
  - EwE's plus-group weight (Ford-Walford) and closed-form tail sum;
  - SpawnProp, FixedFecundity, EggAtSpawn and RecStanza.
- **Oracle:** `audit_ecopath/stz.py`/`stz2.py`, which reproduce EwE's stored stanza values to 7 digits (formula parity). For Edgaras (Karšis), use the P0 harness.
- **Rpath:** the plus-group weight is an Rpath choice, so it goes behind the D1 switch.
- **Pins moved:** LT2022 blue mussel Q/B (13.6086 → 13.6205), Edgaras stanza outputs and SS16.
- **Exit:** the stanza tables match EwE's stored values to 1e-6; Matrix rows 84 and 85 become `implemented`.
- **Status (2026-11-11):** the age table, the Ford-Walford plus group (setup and monthly) and the static WWa are done under the EwE rule, with the D2 stanza feeding-time denominator they need. The tables match EwE's stored values to 1e-6 in all five databases (Edgaras included). Oracle, median monthly deviation: scenario 16 0.44 → 0.06 %, scenario 15 0.64 → 0.19 %, scenario 7 unchanged (5.8 %). LeadingCB is read and anchors the consumption (Edgaras' Karšis leads from stage 6). SplitUpdate takes EwE's rates (last RK4 stage for growth, first for survival). Oracle median in scenarios 16, 15 and 7: 0.01-0.02 %, every group under 0.5 %. BABsplit is read and scales RzeroS. Recruitment forcing and the hatch code follow EwE (EcosimStanzaShape read). SpawnProp, FixedFecundity and RecStanza follow EwE. The P2 stanza items are done; BABsplit, recruitment forcing and SpawnProp are checked against EwE's equations only, since no available database sets them.

### P3: Fitting (C1, C2, C3 + type-5)

- **Covers:**
  - predict catch as B·FishTime from catch series;
  - keep every series per group and sum them;
  - q = mean(ln(obs/pred));
  - sample B at the start of month 6, before the RK step;
  - weights;
  - vulnerability search as v = 1 + exp(p), with blocks;
  - AIC and AICc;
  - the PP anomaly (`fit_pp`);
  - total-mortality series (type 5).
- **Two levels of exit:**
  - **Objective parity:** for a fixed trajectory, PyPath's SS equals EwE's formula. Testable now, without EwE.
  - **Search parity:** the fitted vulnerabilities match EwE's fit on LT2022 scenario 16. This needs the oracle.
- **Rpath:** none, because Rpath has no fitting.
- **Pins moved:** SS16 is redefined as EwE's SS, and a new pin replaces the old one.
- **Status (2026-11-11):** objective parity done (`ewe_sum_of_squares`, `out_FishMort`, WtType read as the weight). Checked on EwE's own trajectories against the oracle's SS to 1e-7 (scenario 16 with datasets 35 and 22); the LT2022 pin is now EwE's SS, 75.22 (PyPath's run: 75.21). Search parity (v = 1 + exp(p), blocks, PP anomaly, AIC) and the reference types other than biomass and catch remain.

### P4: Ecosim kernel (D1 → D2 → D3, plus handling time)

- **Order is forced:**
  - D2's Q must come from Derivt, which needs D1's prey-Ftime remap first.
  - D3's vulnerability-type mediation scales Veff, which only exists after D1.
- **D1:** the arena remap PYY = B/B0, VV′ = 1 + (VV − 1)(1 + f)/2, Q0′ = Q0·VV·f/VV′, also applied to vulnerability forcing. Gate it on `audit_ecosim_core/arena_check.py` (transcribed Derivt) before merging.
- **D2:** feeding-time update from the month-start Derivt Q, averaged over RK steps; stanza denominator ΣN·WWa; integrated groups only; read RiskTime and add the Qmain/Qrisk split.
- **Handling time:** Hden pooling across prey with CmCo; the Hden reset to CmCo/(CmCo + 1); read QmQo.
- **D3:**
  - weights applied to the mediator, not the output;
  - normalised at the base point;
  - one combined X across mediators;
  - application by ApplicationType (search rate, vulnerability, arena area or Mo);
  - fleet effort allowed as a mediator (done: oracle case C, `docs/parity/deviations.md` D3);
  - truncated table lookup, as EwE does.
- **Oracle:** `arena_check.py` gives formula parity; the P0 harness on LT2022 scenario 16 and Curonian gives trajectory parity.
- **Rpath:** D1 is an Rpath choice, so it goes behind the D1 switch. The Rpath-reference tests must stay green under `compat="rpath"`.
- **Pins moved:** SS16, and every LT2022 or Curonian Ecosim trajectory pin.
- **Exit:** Matrix row 45 is `implemented`; trajectory parity holds for scenarios without mediation, plus one with mediation if any database has one.

### P5: Ecopath solver, slice 2 and Ecopath reporting (D4, A1-s2, G-Ecopath)

- **D4:**
  - recompute EE for every living group after the solve (EstimEEAgain), or warn per D1;
  - estimate Q/B where EwE can (EstimateGE, EstimateQBorB, GIM, SolvenoBnoQB) and raise `MissingParameter` where EwE stops;
  - keep fractional Type, with the mixotroph scaling behind the D1 switch.
- **A1 slice 2:** OtherMort → EE = 1 − OtherMort; Immigration, Emigration and EmigRate in the Ecopath balance and in Ecosim (Emig·B loss, constant Immig, Immig into detritus).
- **Ecopath reporting (G):**
  - no clipping of negative M0;
  - a negative-respiration check;
  - stanza BA reported;
  - detritus B required, not filled;
  - detritus export reported.
- **Oracle:** `audit_ecopath/lt.py`/`qb.py`, plus the harness for EE recomputation.
- **Rpath:** the mixotroph scaling and EE recomputation go behind the switch.
- **Exit:** Matrix row 28 is `implemented`.

### P6: Ecospace (B4 → B2, B1 + B3, B5)

- **Order:** B4 first, because B2 uses B4's normalised capacity. B1 and B3 go together, because correct units put typical models at the RK4 stability limit.
- **B4:**
  - capacity = input map × preference × habitat share × environmental responses;
  - normalised once by the starting maximum;
  - recomputed when drivers change;
  - always on (not gated on `environmental_drivers`). **Done (2026-11-13)**, with EwE's HabCap read from the database on EwE's grid; environmental responses and recomputing the capacity when drivers change are still open.
- **B2:** asymmetric per-edge flow k·ρᵢ·min(1, cⱼ/cᵢ), plus the low-capacity gradient (AdjustLowHabCaps, factor 0.4 per cell).
- **B1:** Mrate = Mvel/(π·CellLength), or Mvel/√CellLength for migratory groups; fix both docstrings. **Reader side done (2026-11-13):** `read_ecospace` sets Mrate as the dispersal rate and places its grid as EwE's basemap (it had centroids in km on a degree scale, so diffusion saw neighbours 111 times too far apart); the `EcospaceParams` docstring matches `diffusion_flux`. Its stability warning moved to `rsim_run_spatial` with B3 (only for `implicit_movement=False`). The Shiny page and wizard label the dispersal input as a rate per year (they had km²/month).
- **B3:** an implicit movement step (backward Euler on the sparse adjacency matrix, or EwE's BDF2 + SOR as a clean-room build per D3). **Done (2026-11-13), backward Euler:** after each RK4 step every dispersing group is diffused by solving (I + h·rate·Lap·D_s) B = B_old (`dispersal.implicit_diffusion_step`, sparse LU cached per rate), the implicit form of `diffusion_flux`; advection and external fluxes stay explicit, and fast-equilibrium groups stay unmoved as before. `rsim_run_spatial(implicit_movement=False)` reproduces the explicit term bit for bit. It also removed an explicit instability on the Black Sea (juvenile anchovy, P/B 23 per year, with diffusion).
- **B5:**
  - EatEff and RelVulBad;
  - RelMoveBad and fitness-based emigration;
  - migration maps;
  - the IsAdvected mapping;
  - per-cell feeding time;
  - catch in spatial outputs.
- **Oracle:** `audit_ecospace/exp1-3.py` gives formula parity: equilibrium at density ∝ HabCap, stability at 300 km/yr. Trajectory parity needs the P0 harness, if it can run Ecospace. **Spike verdict (2026-11-12): it can.** The oracle's optional Ecospace stage runs the Black Sea test case, and PyPath is 15–22 % off it (`docs/parity/ecospace_spike.md`). Its first next step is done (2026-11-13): `read_ecospace` builds its grid from EwE's DepthMap and ExclusionMap, pinned to the oracle's 304 water cells.
- **Rpath:** none, because Rpath has no Ecospace.
- **Exit:** the banner from P0 is removed; matrix ecospace rows are reviewed.

### P7: Ecosim options and fishing (G, already-known items)

- **Fishing:**
  - discards built from effort under forced F;
  - forced catch overrides forced F;
  - DiscardMortality defaults to 0;
  - surviving discards routed to detritus;
  - QBmax/QBo;
  - F6 pooling weights.
- **Ecosim options:**
  - non-additive predation mortality (PaddP);
  - BoutFeeding: closed with no code change, because EwE 6.7.144 never sets it (`docs/parity/deviations.md`, BF);
  - environmental and mortality response functions (Mo = (−ln(1 − p)·12 + mo)/mo);
  - a MortOther forcing value of 0 is ignored;
  - relative prey-energy GE scaling;
  - nutrient limitation 2·NutFree/(NutFree + NutFreeBase) (the Black Sea case).
- **Reader:**
  - fleets sorted by Sequence;
  - EcosimScenario StepSize, SystemRecovery, sorwt and Discount;
  - fleet-dynamics columns.
- **Detritus self-fate cascade:** the strict xfail in the Black Sea tests turns into a pass.
- **Exit:** matrix row 39 is `implemented`; the Black Sea trajectory matches EwE-F or the harness.

### P8: Ecotracer (F)

- **Under D5 = relabel:** rename it in the docs and on the Shiny page as a "simplified tracer (not EwE Ecotracer)", and add it to `deviations.md`.
- **Under D5 = port:** a clean-room ConDeriv/Cupdate, with an environment pool, CassimProp, all loss terms, stanza transfer and adaptive AB.
- **Exit:** the harness matches EwE Ecotracer on one model.

### P9: Second audit pass and parity claim

- **Audit:** the review's "Not checked" list:
  - EwE's automatic Ecopath diet changes, diet-sum validation and M2 outputs;
  - forcing application types and mediation shapes;
  - the time-series reader;
  - stanzas when leading biomass is forced;
  - fleet dynamics and PredictSimEffort;
  - Ecospace effort gravity, MPAs, stanza IFD and mortality maps;
  - the remaining reader tables.
- New findings go back into the P2–P7 phases.
- **Claim:** when the four rules in "What full parity means here" hold, update CLAUDE.md's parity figures and the docs.

## Pin ledger

Every phase that moves a pin must name it in its PR. The table lists each pin's history.

| Pin | File | Moved by |
|---|---|---|
| LT2022 EwE-F EE (22 groups) | `test_lt2022_ewef_parity.py` | P1 (3 groups), P5 (EE recomputation, if adopted) |
| EwE SS16 = 75.2231 and dataset 22's 42447.14, by group (oracle); was PyPath's linear SS 617.0 until P3 | `test_ecosim_lt2022_parity.py` | Kernel changes (P4) |
| Rpath reference outputs | `test_rpath_*`, `test_detritus_*` (~20 files) | Must stay green under `compat="rpath"` in every phase |
| Black Sea detritus self-fate xfail | Black Sea tests | P7 |
| Stanza B and Q/B | `test_stanza_plus_group.py`, `test_ewemdb_stanza_z.py` (LT2022 juvenile mussels 15.005259 / 13.620459, to 1e-6) | P2 (age table and plus group: done) |

## Risks

- **No trajectory oracle (D2 fails).** Then every phase ends at formula parity, and rule 3 can't be met. Mitigation: you run EwE 6 by hand for LT2022 scenario 16 and one Curonian scenario.
- **The compat switch spreads through the code (D1).** Mitigation: one `compat` value is carried on `RsimParams` and `Rpath`; it is not threaded through as kwargs.
- **SS16 moves many times.** Mitigation: P0 relabels it, and each PR records the reason in the existing history comment.
- **Licence (D3).** Mitigation: no EwE source is copied into the repo; oracles produce numbers, and only the numbers are committed.
