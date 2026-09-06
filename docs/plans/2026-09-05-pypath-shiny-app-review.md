# pypath-shiny app review (workflow-driven)

Date: 2026-09-05. Scope: `packages/pypath-shiny/src/pypath_shiny/` (all 22 source files, read in full).
Method: 7 reviewer agents (one per file group) produced 79 candidate findings; 7 independent skeptic agents re-read the cited code and tried to refute each one. 76 survived, 3 were refuted (listed at the end).

Severity: high = crash or wrong results in normal use; medium = wrong behavior on plausible edge cases; low = latent defect or repo-rule violation.

## High (25)

### H1. `pages/analysis.py:452` — correctness

plot_trophic_spectrum is called with keyword `metric=`, but the core signature is plot_trophic_spectrum(rpath, by=..., n_bins=..., ...); the call always raises TypeError and the except branch renders a placeholder.

**Failure scenario:** Balance any model, open Trophic Analysis tab -> Trophic Spectrum card always shows 'Could not plot spectrum: ... unexpected keyword argument metric' instead of the plot, regardless of Biomass/Production selection.

**Verifier note:** Confirmed by execution. plot_trophic_spectrum signature is (rpath, by=..., n_bins=..., title=..., figsize=..., ax=...); calling plot_trophic_spectrum(model, metric='biomass') raised TypeError: unexpected keyword argument 'metric', so trophic_spectrum_plot always falls into the placeholder branch. Passing by='biomass' works.

### H2. `pages/analysis.py:492` — correctness

plot_mti_heatmap(mti, model) passes the Rpath object as the second positional argument, which is `group_names: Optional[List[str]]`; matplotlib set_xticklabels/set_yticklabels then fails on a non-iterable dataclass.

**Failure scenario:** Balance any model for which get_mti_matrix succeeds, open Trophic Impacts tab -> heatmap area always shows 'Could not plot MTI: ...' placeholder. Fix: pass list(model.Group[:model.NUM_LIVING + model.NUM_DEAD]).

**Verifier note:** Real but currently latent. plot_mti_heatmap(mti, model) passes the Rpath as group_names; with a synthetic (n x n) matrix it raised TypeError 'unsupported format string passed to Rpath.__format__' from set_xticklabels, while passing list(model.Group[:NUM_LIVING+NUM_DEAD]) works. Caveat: core mixed_trophic_impacts currently raises for every model (IndexError on no-fleet example model; broadcast ValueError on a fleet model because DC[1:n+1,1:n+1] of a (NUM_GROUPS+1, NUM_LIVING) matrix is never n x n when NUM_DEAD>=1), so get_mti_matrix is always None and the user today sees a blank heatmap plus the mti_status alert, not 'Could not plot MTI'. The bug surfaces as soon as the core is fixed.

### H3. `pages/analysis.py:524` — correctness

mti_positive_table/mti_negative_table iterate enumerate(model.Group) (NUM_GROUPS entries incl. fleets) and index mti[i, j], but mixed_trophic_impacts returns an (NUM_LIVING+NUM_DEAD)-square matrix; fleet indices raise IndexError (same at line 551).

**Failure scenario:** Balance a model with at least one fishing fleet (typical), open Trophic Impacts -> both tables show 'Could not extract impacts'. Loop should be bounded by NUM_LIVING + NUM_DEAD.

**Verifier note:** Real but currently latent. mixed_trophic_impacts is documented/coded to return an (NUM_LIVING+NUM_DEAD)-square matrix while model.Group has NUM_GROUPS entries (7 vs 6 on the fleet fixture), so enumerate(model.Group) indexing mti[i,j] must IndexError for any model with fleets. Caveat as in claim 1: today mixed_trophic_impacts itself always raises, so get_mti_matrix is None and the tables show 'No MTI available' rather than 'Could not extract impacts'.

### H4. `pages/analysis.py:590` — correctness

keystoneness_index returns an array with index 0 unused (documented 'for each group (1 to NUM_GROUPS)'), but keystoneness_table zips groups[:len(ks)] with ks directly, shifting every value to the wrong group (same misuse in keystoneness_plot lines 614-626 via ks[:len(biomass)]).

**Failure scenario:** Model with fleets: the first group is shown with keystoneness 0 (the unused slot) and every other group is labelled with the previous group's value, so the 'Top Keystone Species' ranking names wrong species. Model without fleets: len(ks) = NUM_GROUPS+1 > len(groups) -> pandas length-mismatch ValueError -> 'Could not extract keystoneness'.

**Verifier note:** Confirmed with a synthetic MTI matrix on the 7-group fleet fixture: keystoneness_index returns length n+1 with ks[0]=0.0 unused; the page's DataFrame({'Group': groups[:len(ks)], 'Keystoneness': ks}) labelled Phytoplankton=0.0, Zooplankton with Phytoplankton's value, ... Fishery=0.0 — every value shifted one group. For a no-fleet model len(ks)=NUM_GROUPS+1 > len(groups) so the DataFrame constructor raises a length mismatch. Same off-by-one in keystoneness_plot via ks[:len(biomass)]. Currently masked because the core MTI call always fails (see claim 1), but the misuse is real.

### H5. `pages/analysis.py:679` — correctness

balance_summary reads check.get('balanced') and check.get('issues'), but check_ecopath_balance returns keys 'is_balanced', 'messages', 'ee_issues', 'diet_issues', 'balance_issues'.

**Failure scenario:** Balance any model, open Balance Check tab -> badge always reads 'Issues Found' (is_balanced defaults to False) and the issue list is always empty, even for a perfectly balanced model; real diagnostic messages are never displayed.

**Verifier note:** Confirmed. check_ecopath_balance returns keys ['is_balanced','ee_issues','diet_issues','balance_issues','messages'] (verified by running it). balance_summary reads check.get('balanced', False) and check.get('issues', []), so the badge is always 'Issues Found' and the list is always empty; the real messages (e.g. 'Group 1: Diet sum = 0.1000 != 1') are never displayed.

### H6. `pages/data_import.py:1121` — correctness

User-entered biomass estimates are keyed by common_name but biodata_to_rpath looks them up by scientific_name (its default group_names), so every manual biomass value is silently discarded.

**Failure scenario:** Fetch 'Atlantic cod' on the Biodiversity tab, enter Biomass 5.0 in the generated input, click 'Create Ecopath Model'. biomass_estimates == {'Atlantic cod': 5.0} but the core iterates group_name == 'Gadus morhua' (species_data['scientific_name']), finds no match, and falls through to the OBIS occurrence-count proxy (or NaN if no occurrences). The resulting model's Biomass column never reflects what the user typed. Fix: key the dict by row['scientific_name'] or pass group_names=df['common_name'].tolist() to biodata_to_rpath.

**Verifier note:** Confirmed. data_import.py:1112-1116 keys biomass_estimates by row['common_name'] (the string the user typed, e.g. 'Atlantic cod' — batch_get_species_info sets common_name=input name at biodata.py:986/1079). The app passes no group_names (data_import.py:1130-1132), so biodata.py:1188-1189 defaults group_names to species_data['scientific_name'] and :1199-1202 does `group_name in biomass_estimates` with the scientific name. No match ever occurs unless the user typed the scientific name as the species name, so the manual value is dropped and the occurrence-proxy/NaN branch (:1205-1215) runs instead. Real high-severity defect.

### H7. `pages/diet_rewiring_demo.py:674` — correctness

Download handler calls the @render.code Renderer diet_code_example() directly, which raises TypeError in shiny 1.7 (Renderer.__call__ requires _fn); a plain string return would also be interpreted as a file path rather than content.

**Failure scenario:** User clicks 'Download Code' on the Diet Rewiring page -> TypeError from Renderer.__call__(), download fails with a server error; no diet_rewiring_example.py is produced.

**Verifier note:** Confirmed. diet_rewiring_demo.py:671-675 is the same pattern verbatim: diet_code_example is a @render.code Renderer (line 622) called with no args at :674 -> TypeError in shiny 1.7.0 (verified), and a str return would be interpreted as a filesystem path by the download machinery.

### H8. `pages/ecopath.py:159` — correctness

_recreate_params_from_model only copies DC rows with i < nliving; rows for detritus prey (nliving <= i < nliving+ndead) and the Import row (DC[NUM_GROUPS]) are never written into params.diet, so detritivore diets and import fractions are lost.

**Failure scenario:** Balanced model where Zooplankton eats 30% Detritus: recreated params.diet has NaN/0 for the Detritus row, so the predator diet column sums to 0.7; re-balancing (or exporting) yields wrong consumption on detritus and detritus EE. Triggered automatically via line 965/398 on every balance.

**Verifier note:** Confirmed. Lines 157-160 loop i over NUM_GROUPS but only write when i < nliving, so DC rows for detritus prey (nliving..NUM_GROUPS-1) and the Import row DC[NUM_GROUPS] (diet_out has shape (ngroups+1, nliving), core ecopath.py:817-820) are never copied. params.diet from create_rpath_params has those rows ('prey_groups + [Import]') initialised to NaN; core rpath() converts them via np.nan_to_num(nan=0.0) (ecopath.py:361), so detritivore diet fractions and imports become 0. This runs automatically on every balance via lines 965/398.

### H9. `pages/ecopath.py:778` — ui-server-id-mismatch

Effect reads input.model_params_table_cell_edit(), an input that Shiny 1.7.0 never emits (no '_cell_edit' anywhere in the installed shiny package; editable DataGrid edits are surfaced via .data_view()/.cell_patches()/set_patch_fn). The effect therefore never fires and user edits in the Basic Parameters grid never reach params.

**Failure scenario:** User loads a model, edits Biomass of 'Small Fish' in the editable Model Parameters grid (cell shows the new value), clicks Balance Model -> rpath() runs on the original, unedited p.model; results and diagnostics ignore the edit with no error or warning.

**Verifier note:** Confirmed. Installed shiny is 1.7.0; grep of the whole shiny package and www/py-shiny/data-frame/data-frame.js finds no 'cell_edit'. The JS bundle only emits <id>_cell_selection, _column_filter, _column_sort, _data_view_rows and sends edits as 'patches' via RPC handled by the renderer's set_patch_fn/set_patches_fn (render/_data_frame.py:604-678); pypath_shiny defines no patch fn anywhere. input.model_params_table_cell_edit() is therefore a MISSING input and Value.get() raises SilentException (reactive/_reactives.py:454-455), so the effect exits silently. rpath(p) at line 964 uses p.model which was never updated; the grid keeps showing the edit because the default patch fn echoes it back.

### H10. `pages/ecopath.py:839` — ui-server-id-mismatch

Same defect for the diet matrix: input.diet_matrix_table_cell_edit() does not exist in the installed Shiny version, so diet edits are silently discarded.

**Failure scenario:** User changes a diet fraction in the Diet Matrix grid and balances -> p.diet unchanged; balanced EE/TL computed from the old diet while the grid displays the edited value.

**Verifier note:** Confirmed for the same reason as [0]: input.diet_matrix_table_cell_edit() (line 839) is never populated by shiny 1.7.0, the effect silently exits via SilentException, and p.diet is never written. No set_patch_fn/cell_patches usage anywhere in the package.

### H11. `pages/ecopath.py:965` — reactive-misuse

_balance_model sets model_data to the balanced Rpath, which triggers _sync_model_data (line 398) to replace params with _recreate_params_from_model(). That reconstruction drops landings/discards, detritus-fate columns, DetInput, stanzas and remarks (see lines 155-163 and the docstring), so the user's editable parameter set is destroyed by the act of balancing.

**Failure scenario:** Import an EwE model with fleets and catches, click Balance Model once (correct), then click Balance Model again -> second rpath() call runs with all landings/discards zeroed, detritus fate reset to 1/n_det, stanza info gone; EE/M0/TL differ from the first run and the Fisheries tab now shows all-zero catches. Also shows a spurious 'Loaded balanced model' notification after every balance.

**Verifier note:** Confirmed. _balance_model calls model_data.set(model) with a fresh Rpath (Value._set only short-circuits on identity), which re-runs _sync_model_data (line 384). The Rpath dataclass has no .model attribute so is_rpath_params() is False and is_balanced_model() (hasattr NUM_LIVING) is True, so params.set(_recreate_params_from_model(model)) runs and shows the 'Loaded balanced model' toast (lines 398-405). create_rpath_params builds landings/discards as 0.0, detritus-fate columns as NaN (later forced to 1/n_det at lines 927-943), DetInput NaN, empty stanzas, and _recreate_params_from_model copies none of Landings/Discards/DetFate/stanzas. A second Balance click runs rpath() on a params object with zeroed catches and default detritus fate.

### H12. `pages/ecospace.py:745` — error-handling

gpd.read_file errors from the pyogrio engine (pyogrio.errors.DataSourceError, a RuntimeError subclass) are not covered by `except (ValueError, IOError, OSError, zipfile.BadZipFile)` in either load_boundary_on_upload (line 764) or create_spatial_grid (line 943); the uncaught exception in an effect terminates the session.

**Failure scenario:** In Custom grid mode the user uploads a malformed .geojson/.json (or a .zip whose .shp is truncated) -> gpd.read_file raises DataSourceError (verified: MRO is DataSourceError -> RuntimeError) -> not caught -> Session._unhandled_error closes the session for that user instead of showing 'Error loading boundary'.

**Verifier note:** Confirmed. Installed geopandas 1.1.2 selects the pyogrio engine by default (_check_engine -> 'pyogrio'); pyogrio.errors.DataSourceError MRO is (DataSourceError, RuntimeError, Exception). Reproduced: gpd.read_file on a malformed .geojson raises DataSourceError. Neither except clause -- ecospace.py:764 (load_boundary_on_upload) nor :943 (create_spatial_grid) -- lists RuntimeError, so the error propagates out of the @reactive.effect and AppSession._unhandled_error closes the session.

### H13. `pages/ecospace.py:783` — correctness

Clearing the nx/ny (or 1D n_patches) numeric field yields None; create_regular_grid(bounds=(0,0,None,None), nx=None, ny=None) raises TypeError, which is not in the `except (ValueError, OSError)` clause at line 954, so the uncaught exception in the @reactive.effect closes the whole Shiny session (Session._unhandled_error -> close()).

**Failure scenario:** User clears the 'Number of Columns (nx)' box (input.grid_nx() -> None) and clicks 'Create Grid' -> TypeError in create_regular_grid -> 'Unhandled error' printed and the session is closed (page goes grey), instead of a warning notification. Same at line 796 for grid_n_patches.

**Verifier note:** Confirmed. Shiny 1.7.0 number binding (shiny.js:2186-2191) returns null for an empty box, so input.grid_nx()/grid_ny()/grid_n_patches() is None. Empirically: create_regular_grid(bounds=(0,0,None,None), nx=None, ny=None) and create_1d_grid(n_patches=None) both raise TypeError ('unsupported operand type(s) for -: NoneType and int'). create_spatial_grid (ecospace.py:772-957) only catches (ValueError, OSError) at line 954; no None guard exists. Effect_._run catches the leftover Exception and calls session._unhandled_error, and AppSession._unhandled_error prints 'Unhandled error' and awaits self.close() -- session terminated.

### H14. `pages/ecospace_wizard.py:306` — correctness

shared_data.params is a reactive.Value (app.py:329), not an RpathParams, so `shared_data.params is not None` is always True and `shared_data.params.model.Group` raises AttributeError ('Value' object has no attribute 'model'); the same pattern is repeated at lines 357 and 479.

**Failure scenario:** User reaches Step 5 (Assign Preferences) -> wizard_preference_editor render raises AttributeError and the output shows an error; Step 6 wizard_dispersal_table fails the same way; at Step 7 'Create Ecospace Model' the exception is caught (line 532) and only logged, so no EcospaceParams is ever built. The fallback 5 dummy groups never triggers. Should be `p = shared_data.params(); if p is not None: list(p.model['Group'])`.

**Verifier note:** Confirmed. app.py:329 `self.params = reactive.Value(None)` and app.py:367-369 passes that SharedData to ecospace_wizard_server. `shared_data.params is not None` is always True; `shared_data.params.model` -> AttributeError on reactive.Value at lines 306, 357, 479. The dummy-group fallback is unreachable; the render.ui outputs error once hab/grid exist, and _create_ecospace_model swallows it at :532. multistanza.py:207 shows the correct `shared_data.params()` usage.

### H15. `pages/forcing_demo.py:361` — correctness

Forcing type 'fishing' is passed as a StateVariable name, but the core enum only defines 'fishing_mortality' (pypath/core/forcing.py StateVariable), so StateForcing.add_forcing raises ValueError.

**Failure scenario:** User selects 'Fishing Mortality' in the Forcing Type dropdown and clicks 'Generate Forcing' -> StateVariable('fishing') raises ValueError('fishing' is not a valid StateVariable) inside the effect; forcing_obj is never set and Shiny reports an error. The generated code example (line 576, variable='fishing') has the same failure when run.

**Verifier note:** Confirmed. pypath/core/forcing.py StateVariable enum has only 'fishing_mortality'; add_forcing() does StateVariable(variable.lower()). Empirically: StateForcing().add_forcing(variable='fishing', ...) -> ValueError: 'fishing' is not a valid StateVariable ('primary_production' works). forcing_demo.py:361 passes forcing_type verbatim, and the generated code at :576 emits variable='fishing'. Nuance: time_series_data.set(df) at :337 runs before the raise, so the plots still appear; the effect itself errors and forcing_obj is never set.

### H16. `pages/forcing_demo.py:611` — correctness

Download handler calls the @render.code Renderer object forcing_code_example() with no args; in shiny 1.7 Renderer.__call__ requires a value function, so it raises TypeError and the download never produces a file.

**Failure scenario:** User opens 'Code Example' tab and clicks 'Download Code' -> download.handler() raises TypeError: Renderer.__call__() missing 1 required positional argument '_fn' (verified empirically); browser gets a server error instead of forcing_example.py. Even if it returned the string, @render.download treats a plain str return as a file path (session/_session.py: isinstance(contents,str) -> FileResponse(Path(contents))), not as content; the handler must yield the text.

**Verifier note:** Confirmed empirically in shiny 1.7.0: a @render.code object is a Renderer; calling it with no args raises TypeError: Renderer.__call__() missing 1 required positional argument: '_fn'. forcing_demo.py:611 does exactly that. Also verified in shiny/session/_session.py: a str returned from a download handler is treated as a file path (FileResponse(Path(contents))), so returning the code text would fail anyway; the handler must yield it.

### H17. `pages/ibm.py:307` — correctness

The IBM page requires both an Ecosim scenario and model.model, but the Ecosim page only creates a scenario when model_data is a balanced Rpath (ecopath.py sets model_data to the Rpath object), and Rpath has no `.model` attribute; only RpathParams does.

**Failure scenario:** Normal flow: import -> balance -> create Ecosim scenario -> IBM page. _update_group_choices returns early (hasattr(model,'model') False) so the Functional Group dropdown stays empty and 'Initialize IBM' always answers 'Please select a functional group'. Even if a choice existed, line 463 `model.model` raises AttributeError inside the try after the IBM has already been set, producing an 'IBM initialization failed' notification. Should use model.Group/model.type for Rpath.

**Verifier note:** Confirmed. Rpath is a dataclass without a 'model' field (hasattr(rpath_obj,'model') is False at runtime; rpath() never sets one). ecopath.py line 965 sets model_data to the Rpath, and ecosim's _create_scenario refuses anything that is not a balanced Rpath, so in the normal flow model_data is an Rpath whenever a scenario exists. _update_group_choices returns at 'if not hasattr(model, "model")' leaving the select empty, and _initialize_ibm later dereferences model.model at line 463 after ibm_group_instance.set(ibm). Only an unusual re-import after scenario creation avoids this.

### H18. `pages/ibm.py:545` — correctness

group_index passed to SmeltIBM and used as the ibm_groups key is the 0-based DataFrame row index, but SmeltIBM documents group_index as 1-based (0 = Outside), ecosim_deriv only applies IBMs when 1 <= i <= NUM_LIVING, and apply_ibm_to_derivative writes deriv[group_idx] (1-based). Lines 451/539/745/881 meanwhile use group_index+1 for biomass/plots of the intended group.

**Failure scenario:** Select the first consumer row (idx 0): the IBM is silently never applied and the 'IBM-enhanced' curve equals standard Ecosim. Select row k>0: the IBM overrides the derivative of group k (1-based), i.e. the group BEFORE the selected one, while its initial biomass comes from B_BaseRef[k+1] and the comparison plot reads column k+1, so the displayed 'IBM' trajectory shows no IBM effect and a different group's dynamics are corrupted.

**Verifier note:** Confirmed by source: IBMGroup/SmeltIBM docstrings define group_index as 'One-based index ... (0 is reserved for the Outside placeholder)'; apply_ibm_to_derivative writes deriv[group_idx] and ecosim_deriv applies IBMs only 'if 1 <= i <= NUM_LIVING' keyed by ibm_groups. The page uses the 0-based DataFrame row index for SmeltIBM(group_index=...) and ibm_groups[ibm.group_index], while using group_index+1 for B_BaseRef and the out_Biomass plot columns. Row 0 is never applied; row k overrides group k (1-based), i.e. the preceding group.

### H19. `pages/multistanza.py:436` — correctness

download_stanzas returns df.to_csv(...) as a str; @render.download interprets a str return value as a filesystem path (FileResponse(Path(contents))), so the CSV text is treated as a filename and the download fails. When stanza_data is None the handler returns None, which is also not a valid download payload.

**Failure scenario:** User calculates stanzas, then clicks 'Download CSV' -> Shiny builds FileResponse(Path('Stanza,Age_Start,...')) which does not exist -> download errors instead of serving stanza_configuration.csv. Clicking before calculating returns None -> exception in download handler.

**Verifier note:** Confirmed. multistanza.py:436 returns df.to_csv(index=False) (a str). shiny/session/_session.py download branch: `if isinstance(contents, str): FileResponse(Path(contents), ...)` -> the CSV text is used as a path. When stanza_data is None the handler returns None, which falls to the `for chunk in contents` branch -> TypeError. Handler should yield the CSV text.

### H20. `pages/optimization_demo.py:771` — correctness

opt_download_code calls the @render.code renderer object opt_code_example() with no args; Renderer.__call__ requires a value function, so this raises TypeError.

**Failure scenario:** Click "Download Code" -> TypeError: Renderer.__call__() missing 1 required positional argument: '_fn' (verified against shiny 1.7.0). Also the handler returns rather than yields the text, which would then be treated as a file path. Build the code string in a shared helper and `yield` it.

**Verifier note:** Confirmed. opt_code_example (optimization_demo.py:707-766) is a `render.code` Renderer instance (Outputs.__call__'s set_renderer returns the renderer object). Renderer.__call__ signature in shiny 1.7.0 is `(self, _fn: ValueFn[IT]) -> Self`, so `opt_code_example()` at line 771 raises TypeError (missing `_fn`). Even if it returned the text, line 772 `return`s a str, which shiny treats as a file path.

### H21. `pages/results.py:44` — correctness

Plot Style choices "seaborn" and "dark" are not valid matplotlib style names; tl_bar_plot (line 249) calls plt.style.use(style) with no try/except and crashes.

**Failure scenario:** Balance a model, choose Plot Style = Seaborn or Dark Background -> tl_bar_plot raises OSError "'seaborn' is not a valid package style..." (matplotlib >= 3.8 renamed it to 'seaborn-v0_8'; dark is 'dark_background'). results_biomass_plot catches the error at 551 so those two options silently do nothing there. Neither option ever works.

**Verifier note:** Confirmed empirically with matplotlib 3.10.8: 'seaborn' and 'dark' are not in plt.style.available; `plt.style.use('seaborn')` and `plt.style.use('dark')` both raise OSError "... is not a valid package style ...". tl_bar_plot (line 249) has no try/except so it crashes; results_biomass_plot catches OSError at 551 and silently falls back, so neither option ever takes effect.

### H22. `pages/results.py:58` — correctness

Color palette choice "husl" is not a matplotlib colormap; plt.colormaps["husl"] raises KeyError in tl_bar_plot (line 260) and results_biomass_plot (line 563).

**Failure scenario:** Balance a model, select Color Palette = HUSL -> both plots error with KeyError "'husl' is not a known colormap name" (verified with matplotlib 3.10). The try/except at line 549 only wraps style.use, not the colormap lookup.

**Verifier note:** Confirmed empirically with matplotlib 3.10.8 in the shiny env: `'husl' in plt.colormaps` is False and `plt.colormaps['husl']` raises KeyError "'husl' is not a known colormap name". Nothing in pypath or pypath_shiny registers a 'husl' colormap (grep). Lines 260 and 563 do the lookup outside any try/except.

### H23. `pages/results.py:707` — correctness

download_model_csv returns the CSV text (or "") from a @render.download handler; Shiny treats a returned str as a file PATH, so the download always fails.

**Failure scenario:** Load any model and click "Download Model (CSV)": shiny/session/_session.py does FileResponse(Path(contents)) with contents = the CSV text -> file-not-found error, no file delivered. With no model, Path("") is a directory -> same failure. Handler must `yield` the string (as analysis.py:840 does).

**Verifier note:** Confirmed. results.py:706-718 `return`s the CSV string / "" from a @render.download handler. In the installed shiny 1.7.0, session/_session.py does `contents = download.handler()` then `if isinstance(contents, str): ... FileResponse(Path(contents), ...)` — a returned str is treated as a file path. The CSV text is not a path (and Path("") is the cwd directory), so the download errors. analysis.py:840 correctly uses `yield`.

### H24. `pages/results.py:721` — correctness

download_sim_csv returns a CSV string instead of yielding it; Shiny interprets it as a file path.

**Failure scenario:** Run Ecosim, click "Download Simulation (CSV)" -> FileResponse(Path("Month,Seals,...")) -> error response, no download.

**Verifier note:** Confirmed. results.py:720-733 returns `df.to_csv(index=False)` (or "") instead of yielding; same shiny 1.7.0 code path `isinstance(contents, str)` -> `FileResponse(Path(contents))`, so no file is delivered.

### H25. `pages/results.py:736` — correctness

download_annual_csv returns a CSV string instead of yielding it; Shiny interprets it as a file path.

**Failure scenario:** Run Ecosim, click "Download Annual Summary" -> FileResponse on a non-existent path -> error, no download.

**Verifier note:** Confirmed. results.py:735-750 returns the CSV string (or "") instead of yielding; same str-as-path handling in shiny 1.7.0 session/_session.py -> FileResponse on a nonexistent path.

## Medium (26)

### M1. `pages/analysis.py:309` — dead-code

system_indices/flow_indices look up attribute names that do not exist on NetworkIndices (total_production, total_consumption, total_respiration, system_omnivory_index, ascendency, development_capacity, overhead, num_links); the hasattr guard silently drops them.

**Failure scenario:** Balance any model, open Network Analysis -> 'System Indices' card lists only Total System Throughput and Total Biomass; 'Flow Indices' lists only Finn Cycling Index and Connectance. Real fields (system_omnivory, n_links, linkage_density, mean_trophic_level, transfer_efficiency) are never shown.

**Verifier note:** Dead code confirmed: NetworkIndices dataclass fields are n_groups, n_living, n_links, connectance, linkage_density, omnivory_index, system_omnivory, mean_trophic_level, max_trophic_level, total_biomass, total_throughput, transfer_efficiency, finn_cycling_index; total_production, total_consumption, total_respiration, system_omnivory_index, ascendency, development_capacity, overhead and num_links do not exist and are silently skipped by hasattr. Scenario caveat: calculate_network_indices raised IndexError on both test models, so today the cards show 'No indices available.' rather than the two-row tables; once the core works, only the two/two rows described would appear.

### M2. `pages/data_import.py:362` — reactive-misuse

_update_selected_model has no @reactive.event and no else branch, so a stale _selected_rows index is re-applied to a freshly loaded model list, overriding the selected_model_id.set(None) reset and never clearing on deselect.

**Failure scenario:** Click 'Search' for 'Baltic', click row 3 in the grid, then search 'coral'. _search_models sets ecobase_models then selected_model_id=None; in the same flush _update_selected_model is invalidated by ecobase_models, reads the old input.ecobase_models_table_selected_rows() == [3], and sets selected_model_id to coral-results.iloc[3]. The info panel shows 'Selected: <coral model>' the user never clicked, and 'Download Selected Model' downloads it. Because there is no else branch, a later empty selection from the client cannot clear selected_model_id either.

**Verifier note:** Confirmed. _update_selected_model (data_import.py:362-372) is a plain @reactive.effect reading ecobase_models.get() and input.ecobase_models_table_selected_rows(). _search_models (:293-334) and _list_all_models (:279-291) call ecobase_models.set(...) then selected_model_id.set(None); the set() invalidates the effect, which re-runs in the same flush before the browser has re-rendered the grid, so it reads the stale [3] and (if the new result set has >3 rows — the only guard at :370 is row_idx < len(models)) sets selected_model_id to the wrong model, overriding the None reset. Shiny 1.7.0's data-frame.js still emits `${id}_selected_rows` and clears selection on data change, but the resulting [] fails the `if selected_rows and len(...)>0` test at :368 and there is no else branch, so selected_model_id is never cleared. ecobase_selected_info (:376-410) and _download_ecobase (:417) then act on the stale id.

### M3. `pages/ecopath.py:236` — dead-ui

ui.input_file('upload_params') is defined but no server code ever reads input.upload_params(); uploading a CSV does nothing.

**Failure scenario:** User clicks 'Upload Parameters (CSV)' and selects a valid model CSV -> file is uploaded, no parameters are loaded, no error or message shown.

**Verifier note:** Confirmed. 'upload_params' appears exactly once in the whole pypath_shiny package (ecopath.py:237, the ui.input_file definition). No server code reads input.upload_params(), so an uploaded CSV is silently ignored.

### M4. `pages/ecopath.py:657` — dead-ui

fisheries_table is rendered with editable=True but there is no edit handler at all for it (not even a broken one), so landings/discards edits are never persisted to params.

**Failure scenario:** User edits a landing value in the Fisheries grid and clicks Balance -> rpath() uses the original catches; grid shows the edited value while results ignore it.

**Verifier note:** Confirmed. fisheries_table (line 577) returns render.DataGrid(..., editable=True) (line 657) and there is no effect, patch fn, or any code referencing fisheries_table edits anywhere in the package; edits stay in the renderer's cell_patches and never reach params.model landings/discards columns.

### M5. `pages/ecopath.py:921` — correctness

Before balancing, every living group (Type<2, producers included) whose Unassim equals 0 is overwritten with DEFAULTS.unassim_consumers (0.2). An explicit, valid user value of 0 is silently replaced and producers get a non-zero unassimilated fraction.

**Failure scenario:** User enters Unassim=0 for a producer with QB>0 (or deliberately 0 for a consumer), clicks Balance -> p.model is mutated to 0.2, rpath() computes extra flow B*QB*0.2 to detritus, altering detritus EE/M0; the parameter grid then shows 0.2 instead of the entered 0.

**Verifier note:** Confirmed. Lines 920-923: living_mask = Type < 2 (includes producers, Type 1) and every row with Unassim == 0 is overwritten with DEFAULTS.unassim_consumers (0.2). This is not hypothetical in-repo: home.py:662-665 deliberately sets producers' Unassim to DEFAULTS.unassim_producers (0.0) for the example model, which this code then overwrites to 0.2 on Balance. Core rpath() computes loss = B*QB*unassim for every living_idx including producers (ecopath.py:650-655), and p.model is mutated in place so the grid shows 0.2.

### M6. `pages/ecosim.py:260` — dead-ui

The 'Biomass Forcing' card inputs `forcing_group` and `forcing_multiplier` are never read by the server; rsim_run_advanced is called with state_forcing=None and scen.forcing.ForcedPrey is never modified, so the control is a silent no-op.

**Failure scenario:** User selects a group, sets Forcing Multiplier to 0.5, creates and runs the scenario: results are byte-identical to multiplier 1.0, while the help text (line 600) tells users to 'review the effort preview and biomass forcing options'.

**Verifier note:** Confirmed. `forcing_group` and `forcing_multiplier` are defined in the UI (:249, :261); the only server reference is ui.update_select('forcing_group', ...) at :1213 — there is no `input.forcing_group()` or `input.forcing_multiplier()` anywhere in the package. rsim_run_advanced is called with state_forcing=None (:1365) and rsim_run (:1375) with no forcing argument; scen.forcing is never modified, so the slider has no effect on results.

### M7. `pages/ecosim.py:1199` — reactive-state

_create_scenario replaces `scenario` but never clears `sim_output`/`sim_results`, so every renderer pairs an old simulation output with the new scenario (spname, NUM_LIVING).

**Failure scenario:** Balance model A, Create Scenario, Run Simulation; balance model B (different group count) on the Ecopath page; return and click Create Scenario. `annual_catch_table` builds pd.DataFrame(old annual_Catch[:, 1:new_NUM_LIVING+1], columns=new group names) -> ValueError on column/shape mismatch; `biomass_timeseries` indexes old out_Biomass with indices from the new spname (wrong group lines or IndexError); simulation_status labels old crashed indices with new names. Fix: sim_output.set(None) (and sim_results.set(None)) next to scenario.set(new_scenario).

**Verifier note:** Confirmed. `sim_output.set()` occurs only at ecosim.py:1377 (inside _run_simulation) and `sim_results.set()` only at :1378; no other file calls sim_results.set (grep across pypath_shiny). _create_scenario (:1199-1201) sets `scenario`/`sim_scenario` but never clears sim_output. model_data.set on the Ecopath page (:965) also does not reset it. annual_catch_table (:1643-1647) builds pd.DataFrame(output.annual_Catch[:, 1:scen.NUM_LIVING+1], columns=scen.spname[1:NUM_LIVING+1]) from the stale output and the new scenario, which raises on a column-count mismatch when the new model has more living groups; biomass_timeseries and simulation_status likewise mix old output indices with new spname.

### M8. `pages/ecosim.py:1398` — correctness

Recovery check compares absolute end biomass against THRESHOLDS.recovery_threshold (0.01) while the core crash detection uses an absolute 1e-4 threshold; groups whose baseline biomass is below 0.01 can never be reported as recovered.

**Failure scenario:** A group with baseline biomass 0.005 t/km2 dips to 5e-5 in year 3 (flagged crashed) and returns fully to 0.005 by the end: the run notification and simulation_status (line 1456) report 'Population crash ... Groups did not recover.' even though the plot shows full recovery.

**Verifier note:** Confirmed. config.py:181-182 defines crash_threshold=0.0001 and recovery_threshold=0.01, both absolute. Core rsim_run flags a crash when absolute state biomass < 1e-4 (ecosim.py core :1889, :2250-2251). The app's recovery test at :1398 and :1456 is `output.end_state.Biomass[i] > 0.01` (absolute). A group with baseline biomass 0.005 that fully recovers to 0.005 fails this test, so the notification and status say 'Groups did not recover.'

### M9. `pages/ecospace.py:232` — correctness

For hexagon_size_km >= 3.0 (the slider max is exactly 3.0, so reachable) and 21..40 generated patches, the code silently drops the smallest patches until only 19 remain (comment admits it is 'to satisfy ... tests'), deleting valid coverage of the study area; the notification only reports the final count.

**Failure scenario:** User uploads a boundary that tessellates into e.g. 35 hexagons at 3 km and creates the grid -> 16 hexagons (the coastal/edge ones) are discarded, leaving holes in the domain; grid_info reports 19 patches and the total area is understated.

**Verifier note:** Confirmed. config.py:87 sets max_hexagon_size_km = 3.0 and the slider (ecospace.py:445-451) uses that max, so hexagon_size_km == 3.0 is reachable; the gate at :232 is 'hexagon_size_km >= 3.0 and 20 < n_patches <= 40', and the loop at :237-238 drops the smallest patches until 19 remain. The code comment acknowledges this is to 'satisfy reasonable patch-count expectations in tests'. Deliberate does not make it non-defective: kept hexagons all passed the coverage test (center inside or >50% clipped area) and are silently removed, and the notification at :933 reports only the final count.

### M10. `pages/ecospace.py:741` — error-handling

When geopandas is not installed (_HAS_GIS False; geopandas is only in the optional 'spatial' extra of pypath-ewe and not a pypath-shiny dependency) the code raises ImportError inside reactive effects (lines 741, 852, and 76 via create_hexagonal_grid_in_boundary at 919), but the surrounding except clauses only catch ValueError/IOError/OSError/BadZipFile, so the ImportError is unhandled and closes the session.

**Failure scenario:** Install pypath-shiny without the [spatial] extra, choose 'Custom Polygons', upload a shapefile zip -> load_boundary_on_upload raises ImportError -> session terminated rather than a notification telling the user to install geopandas.

**Verifier note:** Confirmed, with a corrected premise. The claim's statement that geopandas is not a pypath-shiny dependency is wrong for pip: packages/pypath-shiny/pyproject.toml:26 requires 'pypath-ewe[interactive,spatial]>=0.4.2', which pulls geopandas. However the conda install path advertised in CLAUDE.md does hit it: conda-recipes/pypath-shiny/meta.yaml run deps are pypath-ewe, shiny, shinyswatch, httpx, uvicorn, and conda-recipes/pypath-ewe/meta.yaml run deps are only numpy/pandas/scipy/matplotlib-base -- no geopandas. All geopandas imports in pypath.spatial (gis_utils.py:21, connectivity.py:21, ecospace_params.py:22) are try/except-guarded, so the page imports fine and the upload handler is reachable. ImportError raised at ecospace.py:741 / :852 (and via create_hexagonal_grid_in_boundary at :76) is not in the except tuples at :764 / :943 / :954, so the effect's exception reaches Session._unhandled_error and closes the session.

### M11. `pages/ecospace.py:999` — correctness

The 'custom' habitat pattern is never implemented: input habitat_upload (UI line 534) is never read in the server, and _compute_habitat_vector falls through to the else branch returning a constant 0.5 for every patch.

**Failure scenario:** User selects Habitat Pattern = 'Custom (upload CSV)' and uploads a habitat matrix -> Habitat Map shows 0.50 everywhere and the spatial simulation runs with uniform habitat_preference 0.5; the uploaded file is silently ignored with no warning.

**Verifier note:** Confirmed. 'habitat_upload' appears only in the UI (ecospace.py:534); grep of src/pypath_shiny shows no server read of input.habitat_upload(). _compute_habitat_vector (:958-1000) handles uniform/gradient/patchy/core_periphery and falls to the else branch returning np.ones(n_patches)*0.5 for 'custom'. Both habitat_plot (:1178) and _run_spatial_simulation (:1027) use this helper, so the uploaded CSV is ignored silently.

### M12. `pages/ecospace.py:1237` — correctness

All four polygon renderers (lines 1201, 1237, 1506, 1684) only draw patches whose geom_type == 'Polygon'; MultiPolygon patches from a user shapefile loaded via load_spatial_grid (which passes geometry through unchanged) are skipped entirely, so those patches are invisible on the grid map, habitat map and biomass map while their values still exist in the arrays.

**Failure scenario:** In 'Use uploaded polygons as-is' mode the user uploads a coastal grid where several cells are MultiPolygons (islands) -> those cells never appear on any map or get labels, giving a misleading picture of the domain and biomass distribution; grid_info still reports the full patch count.

**Verifier note:** Confirmed. All four loops check geom.geom_type == 'Polygon' with no MultiPolygon branch: :1201 (large-grid GeoJSON features), :1237 (small-grid individual polygons + labels), :1506 (habitat_plot), :1684 (biomass_animation_plot). load_spatial_grid (gis_utils.py:33-123) stores gdf as grid.geometry unchanged and build_adjacency_from_gdf has no geometry-type filter, so MultiPolygon patches exist in the arrays but are never drawn. Only the hex generator converts MultiPolygon to its largest part (:194-199); the use_polygons path does not.

### M13. `pages/ecospace.py:1669` — correctness

_spatial_results is never invalidated when a new grid is created, so biomass_animation_plot indexes the old results' patch dimension with the new grid's geometry/centroids (patch_biomass[idx] at 1686, scatter c= length mismatch at 1700), and spatial_metrics_table does np.dot(weights, g.patch_centroids[:,0]) with mismatched lengths at line 1789.

**Failure scenario:** Run a spatial simulation on a 5x5 grid, then set nx=8, ny=8 and click Create Grid -> Biomass Animation tab raises IndexError/ValueError (render error shown) and Spatial Metrics tab raises 'shapes (25,) and (64,) not aligned' until the simulation is re-run.

**Verifier note:** Confirmed. _spatial_results.set(...) occurs only at :1067 inside the simulation handler; create_spatial_grid (:772-957) calls grid.set() without clearing _spatial_results. biomass_animation_plot reads grid() fresh (:1663) and indexes patch_biomass sized to the old grid (:1686 polygon path, :1700 scatter c= with mismatched length -> ValueError). spatial_metrics_table (:1789-1790) does np.dot(weights, g.patch_centroids[:,0]) with old-vs-new lengths -> 'shapes not aligned'. Both are render functions so the error is shown in the output rather than closing the session, as the claim states.

### M14. `pages/ecospace_wizard.py:123` — correctness

The 'hexagonal' grid type selected in Step 2 is ignored: EcospaceGrid.from_regular_grid is always used, while the log message and Step 7 summary report the selected type.

**Failure scenario:** User selects Grid Type = 'Hexagonal' and advances -> a regular rectangular grid is created; the summary says 'Type: hexagonal', so the user gets a silently wrong model configuration.

**Verifier note:** Confirmed. ecospace_wizard.py:116 reads wizard_grid_type but it is used only in the logger.info at :129-135 and in the summary at :395; :123 always calls EcospaceGrid.from_regular_grid. Additionally EcospaceGrid exposes only from_regular_grid and from_shapefile (no hexagonal constructor), so 'hexagonal' can never be honored.

### M15. `pages/ecospace_wizard.py:524` — dead-code

The wizard stores its result on `shared_data.ecospace_params`, but no other module ever reads that attribute (ecospace.py receives model_data/sim_results/sim_scenario, not shared_data, and keeps its own private _ecospace_params), and the wizard shows no success/failure feedback to the user.

**Failure scenario:** Even with the AttributeError above fixed, the user completes all 7 steps and clicks 'Create Ecospace Model' -> nothing visible happens and the Ecospace page never sees the wizard-built parameters; on failure the error is only logged (line 533) and the UI stays silent.

**Verifier note:** Confirmed. grep over the package: `shared_data.ecospace_params` is written only at ecospace_wizard.py:524 and never read anywhere. ecospace.py:678-691 ecospace_server(input, _output, _session, _model_data, _sim_results, _sim_scenario) does not receive shared_data and keeps its own `_ecospace_params = reactive.Value(None)`. The wizard's create effect has no status output; success/failure only go to logger (:526, :533).

### M16. `pages/forcing_demo.py:105` — ui-server-mismatch

The 'Run Demo Simulation' button (id forcing_run_demo) has no server handler, and the forcing_obj reactive.Value set at line 371 is never read anywhere; the 'Simulation Comparison' tab is computed purely from the time-series DataFrame regardless of the button.

**Failure scenario:** User clicks 'Run Demo Simulation' -> nothing happens; the placeholder text 'Generate forcing first, then run demo simulation' is misleading because the comparison plot appears as soon as forcing is generated and the actual StateForcing object is never used.

**Verifier note:** Confirmed. grep: 'forcing_run_demo' appears only in the UI (forcing_demo.py:106); 'forcing_obj' appears only at :296 (creation) and :371 (.set) — never read. forcing_comparison_plot (:442-526) depends solely on time_series_data and input.forcing_mode, so it renders as soon as forcing is generated, regardless of the button.

### M17. `pages/forcing_demo.py:548` — correctness

The generated Python code only defines `values` for the 'seasonal' and 'trend' patterns; for 'pulse', 'step' and 'custom' the emitted script references an undefined `values`, and the boilerplate calls rsim_run without importing it.

**Failure scenario:** User picks Pattern = 'Recruitment Pulses', views/downloads the code and runs it -> NameError: name 'values' is not defined at create_biomass_forcing(...); and later NameError for rsim_run even for seasonal/trend.

**Verifier note:** Confirmed by reading forcing_demo.py:548-557: `values = ...` is emitted only for pattern == 'seasonal' or 'trend'; pulse/step/custom produce a script referencing undefined `values`. The import block (:540-542) imports create_biomass_forcing, StateForcing, rsim_run_advanced only, but :593 calls rsim_run -> NameError for every pattern.

### M18. `pages/ibm.py:616` — correctness

ibm_init_status computes min(ages)/max(ages)/100*n_mature/n_ind without guarding against an empty individuals list; the numeric input min is only a browser hint, not enforced server-side.

**Failure scenario:** User types 0 (or clears then types 0) into 'Number of super-individuals' and clicks Initialize: initialize_from_ecosim builds 0 individuals, the success notification fires, then the Configuration tab renderer raises ValueError('min() arg is an empty sequence') (or ZeroDivisionError at line 625) and the status output errors.

**Verifier note:** Confirmed. ui.input_numeric min is client-side only; int(input.ibm_n_super()) == 0 gives np.linspace(0.5, max_age, 0) -> empty, total_raw == 0.0 -> initialize_from_ecosim logs a warning and returns with individuals == []. get_aggregate_biomass is sum() over an empty generator (0), so the success notification fires with '0 super-individuals'; ibm_init_status then calls min(ages) on an empty list (ValueError) and would divide by n_ind == 0.

### M19. `pages/multistanza.py:82` — ui-server-mismatch

The 'Save Configuration' button (save_stanzas) and the 'Select Group' dropdown (stanza_group) are defined in the UI but never read by the server, so the stanza configuration is never applied to the model.

**Failure scenario:** User selects a group, calculates stanzas and clicks 'Save Configuration' -> nothing happens and no feedback is shown; the Help text ('Save configuration to apply to your model') promises behaviour that does not exist.

**Verifier note:** Confirmed. grep over the package: 'save_stanzas' occurs only in the UI (multistanza.py:83); 'stanza_group' occurs only in the UI (:25) and in ui.update_select writes (:212, :216) — there is no input.save_stanzas() or input.stanza_group() read anywhere, so the button and dropdown have no effect and the Help text at :182 is unfulfilled.

### M20. `pages/optimization_demo.py:402` — correctness

generate_synthetic_data replaces synthetic_data without clearing optimization_results, so comparison plot and results_table pair the old fit with the new observed series.

**Failure scenario:** Run Demo Optimization, then click Generate Synthetic Data: opt_comparison_plot (640-642) and results_table (687-701) plot best_x fitted to the previous dataset against the newly generated Observed_Biomass and compute Error/Error_% across mismatched data, presented as optimized results. Reset optimization_results.set(None) when regenerating.

**Verifier note:** Confirmed: generate_synthetic_data (lines 380-402) only calls `synthetic_data.set(df)` and never resets `optimization_results`. opt_comparison_plot (640-642) and results_table (687-701) read both reactives and would pair the stale best_x with the newly generated observed series.

### M21. `pages/optimization_demo.py:427` — shared-state

np.random.seed(42) reseeds the process-global NumPy RNG on every demo run, affecting all concurrent sessions and any other code using np.random (IBM, dispersal, generate_synthetic_data).

**Failure scenario:** Session A clicks Run Demo Optimization while session B is running a stochastic IBM/Ecospace simulation -> B's random stream is reset mid-run; also every subsequent "Generate Synthetic Data" click in any session produces the same noise. Use a local np.random.default_rng(42).

**Verifier note:** Confirmed: optimization_demo.py:427 `np.random.seed(42)` reseeds the global RNG on every Run Demo click, shared across all sessions in the single process. generate_synthetic_data (line 397/416) draws from the same global stream, so the first Generate after a demo run with identical settings yields identical noise in any session; core pypath/ibm/behavior.py also uses global np.random. Minor imprecision in the scenario: Shiny for Python is single-threaded async, so a sync computation is not reseeded 'mid-run' — its subsequent draws are affected. The shared-state hazard stands.

### M22. `pages/optimization_demo.py:472` — correctness

Sliders allow n_initial (max 20) > n_iterations (min 10); range(n_iterations - n_initial) is then empty so no optimization steps run, yet results are reported as an optimization and summary prints a negative step count.

**Failure scenario:** Set Number of Iterations=10, Initial Random Points=20, Run -> 20 random evaluations, zero BO iterations, convergence plot shows 20 points, summary line reads "Optimization Steps: -10".

**Verifier note:** Confirmed: config.py gives n_iterations min=10 (optimization_iterations_min) and n_initial max=20 (optimization_init_points_max); no guard between them. With 10/20, `range(n_iterations - n_initial)` = range(-10) is empty (zero BO steps) and the summary prints 'Optimization Steps: -10' while results are still stored and plotted.

### M23. `pages/optimization_demo.py:559` — correctness

optimization_summary reports live slider/select values (n_initial, n_iterations, acquisition, objective) instead of the values used for the stored run.

**Failure scenario:** Run with n_iterations=30/n_initial=10, then move n_iterations slider to 100: summary shows "Total Evaluations: 30" next to "Optimization Steps: 90" and a different objective/acquisition label than what produced the plotted results. Store these in the results dict at line 491.

**Verifier note:** Confirmed: optimization_summary (lines 559-563) reads live `input.n_initial()`, `input.n_iterations()`, `input.acquisition()`, `input.objective()` while the results dict built at line 491 stores none of them, so changing a slider after a run alters the displayed summary without re-running.

### M24. `pages/prebalance.py:588` — correctness

rpath_diag_status calls load_rpath_diagnostics(_DIAG_DIR) unconditionally; when the package is installed outside the monorepo _DIAG_DIR is None and utils.load_rpath_diagnostics does Path(None) -> TypeError. rpath_diagnostics_summary handles None but is only called afterwards.

**Failure scenario:** pip/conda install of pypath-shiny (no packages/pypath tree, PYPATH_REPO_ROOT unset) -> opening the Pre-Balance page raises TypeError in the rpath_diag_status output on every session; the sidebar badge shows a Shiny error box.

**Verifier note:** Confirmed. _DIAG_DIR is None when _resolve_repo_root() finds no PYPATH_REPO_ROOT and no packages/pypath/pyproject.toml above the installed file. rpath_diag_status calls load_rpath_diagnostics(_DIAG_DIR) unconditionally and that helper does Path(diag_dir) first; Path(None) raises TypeError ('argument should be a str or an os.PathLike object ... not NoneType'). rpath_diagnostics_summary has the None guard but is only called after the failing line.

### M25. `pages/prebalance.py:761` — correctness

diagnostic_plot passes model_data() straight to plot_biomass_vs_trophic_level / plot_vital_rate_vs_trophic_level, which require RpathParams (they read model.model), while diagnostic_report persists after model_data has been replaced by a balanced Rpath.

**Failure scenario:** Run diagnostics on the unbalanced model, balance it on the Ecopath page (model_data becomes Rpath), return to Pre-Balance -> Visualization tab: AttributeError 'Rpath' object has no attribute 'model' rendered as red 'Error generating plot' text. Should re-check is_rpath_params(data) as _run_diagnostics does.

**Verifier note:** Confirmed by reading. diagnostic_report is a persistent reactive.Value set in _run_diagnostics (which checks is_rpath_params), but diagnostic_plot only checks 'report is None or data is None' and passes model_data() straight to plot_biomass_vs_trophic_level / plot_vital_rate_vs_trophic_level, which index model.model (prebalance.py lines 311/376). After balancing, model_data is an Rpath with no .model, so the plot renders the red 'Error generating plot: ...' text.

### M26. `pages/results.py:249` — shared-state

plt.style.use() mutates process-global rcParams and is never reset, so a style chosen in one session leaks into every other session's and page's matplotlib figures, and selecting "Default" afterwards does not restore defaults.

**Failure scenario:** User A picks GGPlot on the Results page; every subsequent matplotlib plot in user B's session (Ecosim, Analysis, etc.) renders in ggplot style. A then picks Default -> nothing changes because no style.use('default') is issued. Same at line 550. Use `with plt.style.context(style):`.

**Verifier note:** Confirmed by code: results.py:249 and :550 call `plt.style.use(style)` which mutates process-global rcParams; the only style.use calls in the whole app are these two (grep), so no other page resets the style, and the `if style != 'default'` guard means selecting Default never issues `style.use('default')`. Cross-session/cross-page leakage of e.g. 'ggplot' is therefore real; `plt.style.context` would fix it.

## Low (25)

### L1. `logger.py:27` — correctness

The file handler is only attached when the logs directory does NOT already exist (`if not log_dir.exists():`), so file logging works only on the very first run; additionally the module is never imported anywhere in the package, so this configuration is dead.

**Failure scenario:** Second and later launches find the 'logs' directory present, skip the whole block and never add the FileHandler -> pypath_app.log stops being written; in practice nothing imports pypath_shiny.logger, so the intended console/file handlers are never installed at all.

**Verifier note:** Confirmed. logger.py:27 `if not log_dir.exists():` wraps the FileHandler creation, so it is only attached on the very first run; src/logs/pypath_app.log already exists on disk, so subsequent imports skip it. grep: nothing in pypath_shiny imports pypath_shiny.logger or get_logger (only tests/test_app_import.py does); app.py:39 uses logging.getLogger('pypath_app') without importing logger.py, so at runtime neither the console nor file handler is installed.

### L2. `pages/analysis.py:474` — error-handling

When mixed_trophic_impacts/keystoneness_index raise, get_mti_matrix/get_keystoneness return None and mti_status/keystone_status tell the user to 'Balance an Ecopath model' even though the model is balanced; the real error is only in the server log.

**Failure scenario:** Balance a model with no fishing fleets: core mixed_trophic_impacts indexes rpath.Biomass[pred+1] past the end of the 0-based Biomass array (IndexError). UI then shows 'Balance an Ecopath model to calculate Mixed Trophic Impacts' on an already-balanced model, hiding the failure. (The 1-vs-0-based indexing itself is a core pypath bug outside this file.)

**Verifier note:** Confirmed. Running mixed_trophic_impacts on the bundled example model (no fleets, 12 groups) raised IndexError 'index 12 is out of bounds for axis 0 with size 12' from Biomass[pred+1]; get_mti_matrix logs and returns None, and mti_status/keystone_status then render 'Balance an Ecopath model to calculate ...' on an already balanced model. (The fleet fixture fails too, with a broadcast ValueError, so the misleading message is shown for every model.)

### L3. `pages/analysis.py:855` — error-handling

download_model_data swallows the export exception into the CSV body without any logger call, violating the repo's error-handling rule and hiding the failure from the server log.

**Failure scenario:** export_ecopath_to_dataframe raises (e.g. shape mismatch) -> user downloads a file containing 'Error exporting: ...' and nothing is logged for the developer.

**Verifier note:** Confirmed by reading lines 855-856: 'except Exception as e: yield f"Error exporting: ..."' with no logger call, contrary to the repo rule. Reachable: export_ecopath_to_dataframe raised ValueError ('All arrays must be of the same length' / 'Shape of passed values is (6, 4), indices imply (6, 5)') on both test models, so the downloaded CSV contains the error text and nothing is logged.

### L4. `pages/data_import.py:516` — dead-code

_n_stages is computed and never used; the sibling code at lines 612-617 shows it was intended to appear in the 'with N life stages' import notification.

**Failure scenario:** Import an .eweaccdb with multi-stanza groups: the notification reads 'Imported model with N groups, 1 stanza group(s)' and omits the life-stage count that was computed, while the preview banner (line 621) reports it. Harmless but indicates an unfinished message.

**Verifier note:** Confirmed as dead code: grep shows `_n_stages` at data_import.py:516 is its only occurrence; the notification at :522 uses only n_stanza, while the parallel code at :612-616 (preview banner) and :692-698 do include the life-stage count. The underscore prefix suggests the unused-variable lint warning was silenced rather than the message finished; the 'intended for the notification' reading is a plausible inference, not proven. Trivial, but the assertion is accurate.

### L5. `pages/data_import.py:1131` — correctness

input.biodata_area() is passed straight through as area_km2 without a None guard; a cleared numeric field yields None and the core does area_km2 / 1000.0.

**Failure scenario:** Clear the 'Model Area (km²)' field (Shiny returns None), fetch species with OBIS data, click 'Create Ecopath Model'. Because of the biomass key mismatch the occurrence-proxy branch is always taken, so biodata_to_rpath raises TypeError on None / 1000.0 and the user sees only 'Error creating model: unsupported operand type(s)'.

**Verifier note:** Confirmed. input_numeric('biodata_area', value=1000, min=1) at data_import.py:186-192 is passed unguarded at :1131. In shiny 1.7.0, NumberInputBinding.getValue returns null for an empty field (shiny.js:2186-2190) and the Python 'shiny.number' handler passes it through (input_handler.py:173-175), so input.biodata_area() is None. Because claim 0 forces the proxy branch, biodata.py:1208 evaluates `row['occurrence_count'] / (None / 1000.0)` -> TypeError, caught at data_import.py:1143 and shown only as 'Error creating model: unsupported operand type(s)...'. Requires OBIS occurrence_count to be present (default include_occurrences=True). Low severity, real.

### L6. `pages/ecopath.py:972` — error-handling

Balancing errors are dumped with traceback.print_exc() to stdout instead of the module logger; the module defines no logger = logging.getLogger(__name__) at all, violating the repo logging rule.

**Failure scenario:** rpath() raises inside _balance_model under a server deployment that captures only logging output -> the traceback is lost to stdout and only the truncated str(e) notification survives for diagnosis.

**Verifier note:** Confirmed. ecopath.py contains no 'import logging' and no logger; the except block at lines 969-973 does 'import traceback; traceback.print_exc()' to stdout and shows only str(e). This violates the repo rule requiring logger = logging.getLogger(__name__). Low severity as claimed.

### L7. `pages/ecosim.py:696` — documentation-mismatch

Help text claims the crash threshold is '1/10,000 of reference biomass' (relative), but pypath.core.ecosim.rsim_run uses an absolute crash_threshold = 1e-4 on raw state biomass.

**Failure scenario:** A user with a large-biomass model (e.g. phytoplankton 100 t/km2) expects a crash flag when it drops to 0.01 (1e-4 relative); no crash is reported, contradicting the help text.

**Verifier note:** Confirmed. Help text at :696 says the threshold is '1/10,000 of reference biomass'. Core rsim_run sets crash_threshold = 1e-4 (:1889) and compares raw absolute state biomass (`state = scenario.start_state.Biomass.copy()` at :1062; `state[1:NUM_LIVING+1] < crash_threshold` at :2250-2251) — no division by reference biomass. A 100 t/km2 group dropping to 0.01 would not be flagged.

### L8. `pages/ecosim.py:1109` — dead-ui

'Custom' fishing scenario promises CSV upload ('Upload custom effort CSV or define in Results tab') but no upload input exists anywhere and _apply_fishing_scenario has no custom branch, so it silently behaves as baseline.

**Failure scenario:** User picks 'Custom', looks for an upload control that does not exist, clicks Create Scenario and gets 'Scenario created successfully!' with constant effort = 1.0.

**Verifier note:** Confirmed. The only ui.input_file calls in the package are in data_import.py, ecopath.py, ecospace.py, ecospace_wizard.py — none in ecosim.py. fishing_params_ui for 'custom' returns only a text paragraph (:1109-1112). _apply_fishing_scenario (:1220-1265) has branches for baseline/increase/decrease/closure only; 'custom' falls through leaving ForcedEffort at 1.0, and the effect still shows 'Scenario created successfully!'.

### L9. `pages/ecosim.py:1133` — correctness

sim_years numeric input allows PARAM_RANGES.years_min = 1 (help text at line 585 says 1-500), but rsim_scenario raises ValueError for fewer than 2 years.

**Failure scenario:** User enters 1 for Simulation Years and clicks Create Scenario -> notification 'Error creating scenario: Years must be a range of at least 2 years'.

**Verifier note:** Confirmed. config.py:213 sets years_min=1 and the numeric input uses it as min (:53). With sim_years=1, years=range(1, 2) has len 1, and core rsim_scenario (:894-895) raises ValueError('Years must be a range of at least 2 years'), caught at :1217 and surfaced as 'Error creating scenario: ...'. Help text at :585 says 'Simulation years (1-500)'.

### L10. `pages/ecosim.py:1169` — dead-code

Diet reconstruction loop into orig_params.diet (lines 1169-1175) is never consumed: rsim_scenario builds params from rpath.DC via rsim_params and only copies the 'model' attribute from rpath_params. Similarly `_final_biomass` (1392), `_show_autofix_help` (442) and the `elif is_rpath_params` branch (1149, unreachable after _require_balanced_model_or_notify) are dead.

**Failure scenario:** Detritus/import diet rows are deliberately skipped by the loop (i < nliving), which would give a wrong diet if this code were ever relied upon; today it is only wasted work and misleading maintenance surface.

**Verifier note:** Confirmed. rsim_scenario builds params via rsim_params(rpath, ...) from rpath.DC (:898) and only copies INSTRUMENT_GROUPS, VERBOSE_DEBUG, instrument_callback, spname, INSTRUMENT_ASSUME_1BASED, and 'model' from rpath_params (:923-932); rpath_params.diet is never read (stanza path is inactive because create_rpath_params(groups, types) defines no stanzas). `_final_biomass` (:1392) is assigned and never referenced; `_show_autofix_help` (:442) is never referenced (the btn_autofix_help handler at :470 does not use it); the `elif is_rpath_params(model)` branch (:1149) is unreachable since _require_balanced_model_or_notify (:1129) already returned False for anything failing is_balanced_model. Minor correction to the claim: rsim_scenario copies several attributes, not only 'model', but none is the diet.

### L11. `pages/ecosim.py:1209` — correctness

groups[:num_living_dead] assumes canonical living->detritus->fleet ordering; the core explicitly supports non-canonical order (rsim_params uses np.where(rpath.type == 3) for fleets).

**Failure scenario:** A model where a fleet row precedes a detritus row: plot_groups/forcing_group choices include the fleet and omit the last detritus group. Should filter by `types` instead of slicing.

**Verifier note:** Confirmed. :1204-1209 computes num_living_dead = NUM_LIVING + NUM_DEAD and slices `groups[:num_living_dead]` from model.Group in original input order. Core rpath() explicitly preserves original group order (ecopath.py :242, :298) and rsim_params uses np.where(rpath.type == 3) with a comment about non-canonical order (:417-418); regression tests test_arbitrary_ordering_pipeline.py and test_ecosim_deriv_gear.py ('Fleet first') exist. For such a model the slice includes the fleet and drops the last living/detritus group from plot_groups/forcing_group.

### L12. `pages/ecosim.py:1217` — error-handling

Module defines no logger; both `except Exception as e` handlers (here and line 1419) only show a notification and never log, so server-side tracebacks for scenario/simulation failures are lost (repo rule: logger.debug('...: %s', e)).

**Failure scenario:** rsim_run raises deep inside the integrator; the user sees 'Simulation error: <message>' and nothing is written to the server log for diagnosis.

**Verifier note:** Confirmed. grep for 'logger' / 'logging' in pages/ecosim.py returns nothing. The two handlers at :1217 and :1419 only call ui.notification_show(f'... {str(e)}') with no logging, so exceptions from rsim_scenario/rsim_run leave no server-side trace. This departs from the repo convention (logger.debug('...: %s', e)) though it is not a functional crash.

### L13. `pages/ecosim.py:1281` — correctness

scenario_status reports the live input.sim_years() rather than the years the scenario was actually built with.

**Failure scenario:** Create a scenario with 50 years, then change the Simulation Years field to 100 without re-creating: status reads 'Scenario ready: N groups, 100 years' while the scenario (and the run) still covers 50 years.

**Verifier note:** Confirmed. scenario_status (:1281) formats `input.sim_years()` (the live input), not the years stored in the scenario. Because @render.ui is reactive on the input, changing the field after Create Scenario immediately updates the label while `scenario` remains the old object built with the previous years.

### L14. `pages/ecospace.py:613` — dead-code

input habitat_view_group is populated with group choices at line 1080 but never read by any server code, so the 'View Habitat for Group' selector on the Habitat Map tab does nothing.

**Failure scenario:** After a simulation the user changes 'View Habitat for Group' -> the habitat map does not change (it is group-independent), leaving the user to believe the per-group view is broken.

**Verifier note:** Confirmed. grep of src/pypath_shiny shows 'habitat_view_group' only at :614 (UI) and :1080 (ui.update_select populating choices); input.habitat_view_group() is never read. habitat_plot (:1178) depends only on grid() and the habitat pattern inputs, so changing the group selector has no effect.

### L15. `pages/ecospace.py:699` — reactive-misuse

load_boundary_on_upload returns early when grid_type != 'custom' without clearing boundary_polygon, so a previously uploaded boundary persists after switching grid type; grid_plot then centres/fits the map on the stale boundary (lines 1131, 1413-1423) and grid_info reports it.

**Failure scenario:** User uploads a Baltic boundary in Custom mode, switches to 'Regular 2D Grid' and clicks Create Grid -> the map zooms to the Baltic boundary while the new regular grid sits at lon/lat 0-5 off-screen, and the info box still lists the boundary.

**Verifier note:** Confirmed. load_boundary_on_upload (:695-768) reads input.grid_type() and returns at :699-700 without touching boundary_polygon; the only set(None) calls are at :706 (no file) and :768 (error). create_spatial_grid's regular_2d/1d branches (:778-806) do not clear it either. grid_plot then takes the has_boundary path for centre (:1131-1135) and fit_bounds (:1413-1416), so the map fits the stale boundary while create_regular_grid places the new grid at lon/lat 0..nx/0..ny; grid_info (:1444-1459) still reports the boundary.

### L16. `pages/ecospace.py:990` — shared-state

np.random.seed(42) reseeds the process-global NumPy RNG every time the habitat vector is computed (each habitat_plot render and each simulation run), which resets the random sequence for every other session/page in the same process (e.g. IBM stochastic spawning/mortality); should use a local np.random.default_rng(42).

**Failure scenario:** One user has the Habitat Map tab open with 'Patchy' selected while another session runs a stochastic IBM simulation -> the IBM's np.random stream is reset to seed 42 mid-run whenever the first user's plot re-renders, producing correlated/non-reproducible results.

**Verifier note:** Defect confirmed but the cited victim is wrong. ecospace.py:990 calls np.random.seed(42) on the process-global RNG from _compute_habitat_vector, which runs on every habitat_plot render and every simulation. However pypath.ibm does not use the global stream: smelt.py:266 uses np.random.default_rng(42), behavior.py:214 uses default_rng(), reproduction.py:228 uses stdlib random -- so the IBM scenario cannot occur. A real victim exists in the same process: pages/optimization_demo.py:397 and :416 draw np.random.normal from the unseeded global stream (before that page's own seed(42) at :427), so a concurrent ecospace render does perturb another session's results. Fix is still np.random.default_rng(42) locally.

### L17. `pages/ecospace.py:1583` — correctness

The 'Gravity (follow biomass)' fishing allocation uses a constant biomass array np.ones((2,n_patches))*10, so allocate_gravity always returns a uniform allocation regardless of alpha; the 'habitat' option (UI line 553) is not handled and also falls to uniform; and none of the Spatial Fishing settings are passed to rsim_run_spatial (EcospaceParams has no fishing field), so the whole panel has no effect on the simulation.

**Failure scenario:** User selects 'Gravity' and moves the alpha slider from 0 to 2, or selects 'Habitat-based' -> the Fishing Effort plot is identical uniform effort in all cases and the simulation ignores the choice, with no indication that the setting is inert.

**Verifier note:** Confirmed empirically. With biomass = np.ones((2,n))*10 (ecospace.py:1583), allocate_gravity returns identical uniform effort for alpha=0.0 and alpha=2.0 (attractiveness = 10**alpha per patch, constant). 'habitat' is not matched at :1577-1596 and falls to the else -> allocate_uniform. EcospaceParams (ecospace_params.py:350-390) has fields grid, habitat_preference, habitat_capacity, dispersal_rate, advection_enabled, gravity_strength, external_flux, environmental_drivers -- no fishing/effort field -- and _run_spatial_simulation (:1010-1097) never reads fishing_allocation/gravity_alpha/port_*. The 'for demonstration' comment at :1582 does not change the user-visible fact that the panel is inert.

### L18. `pages/ecospace_wizard.py:76` — reactive-misuse

@reactive.event(input.wizard_drawn_polygon) uses the default ignore_none=True, so the JS DELETED handler's Shiny.setInputValue('wizard_drawn_polygon', null) never triggers the effect and the else branch (drawn_polygon.set(None)) is unreachable.

**Failure scenario:** User draws a polygon, then deletes it with the draw toolbar -> drawn_polygon still holds the old geometry; subsequent grid creation/download use the deleted polygon.

**Verifier note:** Confirmed. shiny 1.7.0 reactive.event signature: ignore_none: bool = True; the trigger does `if ignore_none and all(map(_is_none_event, vals)): req(False)`. The JS DELETED handler (:646-651) sets the input to null, so _capture_polygon at :75-83 is skipped and the else branch (drawn_polygon.set(None)) never runs; the stale polygon remains.

### L19. `pages/ecospace_wizard.py:311` — reactive-misuse

wizard_preference_editor is a @render.ui that writes preference_matrix.set(prefs) as a side effect; the matrix used by 'Create Ecospace Model' therefore depends on whether/when this output last rendered rather than on explicit state.

**Failure scenario:** If the Step 5 output has not rendered (e.g. an exception in the render, as with the shared_data.params bug), preference_matrix stays None and _create_ecospace_model silently falls back to an all-ones habitat preference matrix without telling the user.

**Verifier note:** Confirmed. ecospace_wizard.py:321 calls preference_matrix.set(prefs) inside the @render.ui wizard_preference_editor; the matrix is thus only populated if/when that output renders. _create_ecospace_model (:487-493) silently falls back to np.ones((n_groups, n_patches)) when prefs is None with no user-facing message. Given claim 4 (AttributeError inside that render whenever shared_data is passed), this fallback is what actually happens in the wired app.

### L20. `pages/ibm.py:572` — correctness

ibm_param_summary formats numeric inputs with f'{...:.1f}'/'.4f' without None guards; clearing any ui.input_numeric (ibm_t_ref, ibm_ra, ibm_n_super...) yields None.

**Failure scenario:** User clears the 'Reference temperature' field while editing -> input.ibm_t_ref() is None -> TypeError in the render.table, Parameter Summary table shows an error until a value is retyped.

**Verifier note:** Confirmed by reading ibm_param_summary: f'{input.ibm_t_ref():.1f}', f'{input.ibm_ra():.4f}' etc. with no None guards; ibm_t_ref and ibm_ra are ui.input_numeric, which yields None when the field is cleared, so the render.table raises TypeError until a value is retyped. Low severity as stated.

### L21. `pages/ibm.py:780` — error-handling

`except (IndexError, AttributeError): pass` swallows the error silently, violating the repo rule to log via logger.debug('...: %s', e).

**Failure scenario:** model_data is an Rpath (no .model) -> AttributeError swallowed with no log entry; plot title silently falls back to 'Group N'.

**Verifier note:** The bare 'except (IndexError, AttributeError): pass' at ibm.py ~line 780 is a genuine violation of the repo's logging rule. However the stated scenario is inaccurate: the block is guarded by 'if model is not None and hasattr(model, "model")', so an Rpath never enters the try and no AttributeError is swallowed for it; the except is only reachable via an out-of-range group_index (IndexError) or a .model lacking .iloc. Style-rule defect real, scenario wrong.

### L22. `pages/multistanza.py:365` — correctness

fig.to_html(div_id='growth_plot') injects a <div id='growth_plot'> inside the Shiny output container that already has id='growth_plot' (same for 'biomass_plot' at line 429), producing duplicate DOM ids; Plotly.newPlot('growth_plot') then resolves to whichever element getElementById returns first (the outer output container).

**Failure scenario:** On each re-render the Plotly script targets the outer shiny-html-output div instead of its own inner div; Shiny's subsequent innerHTML replacement and Plotly's purge fight over the same node, which can leave an empty or stale plot after changing parameters and re-calculating.

**Verifier note:** Root defect confirmed empirically (app launched, Multi-Stanza page, Calculate clicked 3x with changed vb_k): document.querySelectorAll('[id=growth_plot]').length == 2 (outer 'shiny-html-output' and inner 'plotly-graph-div'); Plotly.newPlot resolves to the OUTER container (it gains class js-plotly-plot, .plot-container is inserted as its first child) while the inner 400px plotly-graph-div stays empty (0 children) and sits below the plot, leaving a ~400px blank block in the card. However, the hedged consequence 'empty or stale plot' was NOT reproduced: traces/ticks updated correctly on each re-render. Same pattern at :429 for biomass_plot. (Side observation: first render logged 'ReferenceError: Plotly is not defined' due to the include_plotlyjs='cdn' race — separate issue.)

### L23. `pages/prebalance.py:637` — dead-code

Neither Session.show_modal nor ui.modal_dialog exists in shiny 1.7.0 (the API is ui.modal_show(ui.modal(...))); the 'preferred' modal branch always raises AttributeError and only the inline fallback ever runs (same at line 644).

**Failure scenario:** Click the info button next to the Rpath diagnostics badge: no modal ever appears; the verification output is only ever injected inline into the sidebar. The primary path is unreachable dead code.

**Verifier note:** Confirmed at runtime: shiny 1.7.0 has hasattr(ui,'modal_dialog') False and hasattr(Session,'show_modal') False (ui.modal / ui.modal_show exist, and app.py itself uses ui.modal_show). session.show_modal is evaluated first and raises AttributeError, so both 'preferred' branches are unreachable and only the inline fallback ever runs.

### L24. `pages/prebalance.py:641` — error-handling

The outer `except Exception as e` in _show_rpath_modal displays str(e) but never logs it (no logger.error/debug), hiding subprocess/loader failures from the server log.

**Failure scenario:** run_verify_rpath or load_rpath_diagnostics raises unexpectedly -> only the terse message is shown inline; nothing recorded in logs for diagnosis.

**Verifier note:** Confirmed by reading lines 641-647: the outer 'except Exception as e' builds err_body from str(e) and shows/sets it, with no logger.debug/error call, so subprocess/loader failures from run_verify_rpath / load_rpath_diagnostics are not recorded in the server log.

### L25. `pages/results.py:630` — dead-code

Branch `model.params` for balanced models is unreachable (a balanced Rpath always has summary()) and would raise AttributeError if reached, since Rpath has no `params` attribute.

**Failure scenario:** Dead code in params_data_table (630) and download_model_csv (715); the apparent intent (show editable params for a balanced model) never executes, and if the guard at 627 were ever loosened it would crash with AttributeError: 'Rpath' object has no attribute 'params'.

**Verifier note:** Confirmed dead code: Rpath has no `params` attribute (inspected class; never assigned) but does have `summary`. utils.get_model_info sets `params: model.params if hasattr(model,'params') else None` -> None for balanced models, so the `elif info['params'] is not None` branch is never entered for a balanced model, and the first branch (`is_balanced_model and hasattr(model,'summary')`) always wins. `model.params` at 630/715 is unreachable and would raise AttributeError if reached. Low severity, as claimed.

## Refuted by verifiers (not defects)

- `pages/ecopath.py:790` — Edit handlers use p.model.loc[row, ...] / p.diet.loc[row, ...] with the positional row index from the grid; if params.model/diet has a non-RangeIndex (e.g. a DataFrame filtered/reordered upstream without reset_index), loc resolves by label and hits the wrong row or KeyError. Latent because the handlers currently never run (see line 778).
  Reason: Speculative; no code path produces the precondition. Every RpathParams constructor yields a RangeIndex: create_rpath_params builds pd.DataFrame from dicts, read_rpath_params uses read_csv, and ewemdb.py:641, ecobase.py:747, biodata.py:1195/1241 all go through create_rpath_params. No pypath_shiny page reassigns, filters, sorts or drops rows of params.model/params.diet in place (grep for .model =, .drop, .sort_values, .diet = finds none). Additionally the handlers never execute (see [0]) and the edit dict shape they expect (edit['row']) is not what Shiny provides, so the failure scenario cannot be concretely reproduced.

- `pages/home.py:738` — Broad `except Exception as e` only shows a notification; nothing is logged and the module defines no logger, violating the repo rule for exception handlers.
  Reason: Observation is accurate (home.py has no logger/logging import; the handler at ~738 only calls ui.notification_show with str(e)), but no behavioral defect: the exception is not swallowed silently — its message is surfaced to the user. The CLAUDE.md rule targets bare `except Exception: pass`, which this is not; the notification is not truncated by the code. Adding a logger.exception call is an optional improvement, not a defect.

- `pages/results.py:553` — Inline `import logging` / logging.getLogger(__name__) inside the render function instead of a module-level logger; the module defines no logger, and the sibling tl_bar_plot has no handling at all.
  Reason: No functional defect: the inline `import logging; logging.getLogger(__name__).warning(...)` at 553-557 works correctly and is functionally equivalent to a module-level logger; it is a style/convention nit only. The substantive part (tl_bar_plot crashing on invalid styles) is already claim [5] and is confirmed there.

---

## Fix log (2026-09-05, same day)

Verified with: `pytest packages/pypath-shiny/tests --ignore=tests/ui` (295 passed), `pytest packages/pypath/tests/test_analysis.py test_indicators.py` (53 passed), core `-k "analysis or indicator or instrument or plotting"` (82 passed), and a live Playwright session (example model → edit Phytoplankton biomass in the grid → Balance → exported CSV shows 12.5; all seven downloads respond 200, the three with no simulation/stanza data yield an empty file; Analysis tab renders network indices and MTI tables with no server errors). Not driven live: the diet-matrix patch handler and the invalid-edit revert branch (same code shape as the verified params-grid handler).

### Fixed in this pass

| Report item | Fix |
|---|---|
| H `pages/ecopath.py:778`, `:839` (dead `_cell_edit` inputs) | Replaced both effects with `@<grid>.set_patch_fn` handlers. Rows are resolved by Group name, not position; invalid edits revert the cell. |
| H `pages/results.py:707/721/736`, `pages/multistanza.py:436` (download returns str) | Handlers now `yield` CSV text on every branch. |
| H `pages/forcing_demo.py:611`, `pages/diet_rewiring_demo.py:674`, `pages/optimization_demo.py:771` (download calls a renderer) | Code-example text moved to module-level `build_*_code()` functions shared by renderer and download; download `yield`s. Forcing script now defines `values` for every pattern and imports `rsim_run`. |
| H `pages/results.py:44`, `:58` (invalid style/palette names) | Choices are now `seaborn-v0_8`, `dark_background`, `viridis`; unknown styles are logged and ignored. |
| M `pages/results.py:249` (global `plt.style.use` leak) | Style applied inside `plt.rc_context()` per render. |
| H `pages/analysis.py:452/492/524/551/590/679` (wrong core keywords, args, keys, indexing) | `by=`, `group_names=`, `is_balanced`/`messages`, MTI bound = `mti.shape[0]`, keystoneness aligned to `len(ks)`. |
| M `pages/analysis.py:309` (non-existent NetworkIndices fields) | Both attribute lists now use real dataclass fields. |
| H `pages/ecospace.py:783`, `:745`; M `:741` (session-closing exceptions) | None guards on nx/ny/n_patches; except tuples include `TypeError`, `RuntimeError` (pyogrio), `ImportError` (no geopandas); errors logged. |
| M `pages/ecospace.py:1669`, `pages/ecosim.py:1199`, `pages/optimization_demo.py:402` (stale results) | Results cleared when a new grid / scenario / dataset is created. |
| H `pages/data_import.py:1121` (biomass keyed by common name) | Keyed by `scientific_name`, matching `biodata_to_rpath`. |
| L `pages/results.py:630` (unreachable `model.params`) | Branch removed (also from the download). |

### Core defects found while fixing (outside the app review scope)

- **`pypath/core/analysis.py` assumed 1-indexed Rpath arrays** (`Biomass[i+1]`, `DC[1:n+1, 1:n+1]`), but real `Rpath` arrays are 0-indexed with length `NUM_GROUPS`, and `DC` is `(n_groups+1, NUM_LIVING)`. `mixed_trophic_impacts` raised IndexError on every real model, so the Analysis page's MTI/keystoneness were always blank. **Fixed**: MTI, keystoneness (now returns `n_groups` values, 0-indexed), network indices, balance check (messages use group names), and export (Group/From/To are names). `tests/test_analysis.py` rewritten to 0-based mocks plus a real-model regression class.
- **`pypath/core/indicators.py` has the same 1-indexed assumption** (`_build_flow_matrix`, `transfer_efficiency`, `ecosystem_indicators`, `system_maturity`, ...) and its tests mock 1-indexed arrays. **Not fixed.** `calculate_network_indices` now catches the IndexError and logs a warning, so Finn Cycling Index and Transfer Efficiency show 0 on the Analysis page until it is.
- **MTI magnitude**: on `example_model_data/model.csv` the MTI matrix reaches |1e5|; the Leontief step uses consumption/production ratios that can exceed 1. Values look plausible (|0.07|) on the app's built-in example. Validate against an EwE reference (`compare-ewe` skill) before trusting keystoneness rankings.
- **`example_model_data/diet.csv`** leaves Large pelagics and Seabirds with zero diet although both have QB > 0; the balance check now reports this.

### Environment note

The `shiny` micromamba env has a non-editable `pypath` 0.4.2 copy in site-packages plus editable `.pth` files pointing at `.worktrees/batch1-bugfixes`. The `pypath-shiny` CLI therefore runs stale code; tests above were run with `PYTHONPATH` set to `packages/pypath/src` and `packages/pypath-shiny/src`. Reinstall both packages editable from `packages/` to pick up these fixes.

### Fix log, second pass (2026-09-05 evening)

Verified with: full Shiny suite 303 passed (`--ignore tests/ui`), plus a live Playwright session on the built-in example: Balance keeps the Fisheries tab populated (landings survive balancing); the IBM page lists the 7 consumer groups from the balanced Rpath, and Initialize IBM succeeds after creating an Ecosim scenario; no tracebacks in the server log.

| Report item | Fix |
|---|---|
| H `pages/ecopath.py:965` (balancing wipes editable params) | `_sync_model_data` now skips re-creation when the incoming Rpath is the one just produced by `_balance_model` (`balanced_model` is checked under `reactive.isolate()`). The user's RpathParams, including landings, stanzas and remarks, survive. |
| H `pages/ecopath.py:159` (detritus/import diet rows dropped) | `_recreate_params_from_model` copies the full DC by prey/predator *name* (living, detritus and Import rows), plus landings/discards per fleet and detritus-fate columns. Still not reconstructed: stanzas, remarks. |
| M `pages/ecopath.py:921` (producers' Unassim overwritten) | Default applies only to consumers (`Type < 1`); an explicit 0 for a consumer is still replaced by the default, by design. |
| H `pages/ibm.py:307` (needs `.model`) | New `_model_group_table()` reads either an RpathParams or a balanced Rpath; used for the dropdown and the notification name. |
| H `pages/ibm.py:545` (0- vs 1-based group index) | Row index converted once to the 1-based Ecosim index at initialization; `B_BaseRef`, `ibm_groups` key and output columns use it directly (no `+ 1` left). |
| H `pages/ecospace_wizard.py:306/:357/:479` (`shared_data.params` is a reactive.Value) | New `shared_group_names()` reads the reactive value and handles RpathParams, Rpath and None. |
| M `pages/forcing_demo.py:361` (`'fishing'` not a StateVariable) | UI choice key changed to `fishing_mortality`; test asserts every key is a `StateVariable` value. |

Not driven live: the IBM comparison run itself (Run button), the wizard steps that call `shared_group_names` (covered by unit tests), The landings/discards copy in `_recreate_params_from_model` is unit-tested with an inline 4-group model that includes a fleet.

### Fix log, third pass (2026-09-05 evening)

All 42 items that remained after the second pass were worked through: 41 fixed, 1 deferred with the reason below.

Verified with: full Shiny suite **319 passed** (`--ignore tests/ui`), plus a live Playwright session on the built-in example.

**Batch A, logging, guards and dead code (20 items).** Module loggers added to `ecosim`, `home` and `multistanza`; `traceback.print_exc()` and two silent `except` handlers replaced with logger calls; None guards on every numeric input read by the IBM parameter summary, the IBM init status (also guards an empty individuals list), and the biodata study area; the unused `_n_stages` value now appears in the import notification as intended; dead `_final_biomass`, `_show_autofix_help` and the diet-reconstruction loop removed from `ecosim`; the crash-threshold help text now matches the absolute threshold the core uses; `years_min` raised from 1 to 2 because `rsim_scenario` rejects a single year; duplicate Plotly DOM ids fixed in `multistanza`; the wizard's polygon-deleted handler now fires (`ignore_none=False`); the Rpath diagnostics modal uses the real `ui.modal_show` API instead of a branch that always raised; `rpath_diag_status` handles an absent diagnostics directory; both `np.random.seed(42)` calls replaced with local generators so they no longer reseed the process-global RNG shared across sessions.

**Batch B, correctness (10 items).** Hexagonal grids are no longer silently pruned to 19 patches (see below); MultiPolygon patches now render on all four maps via one `_iter_polygons` helper; a stale uploaded boundary is cleared when the grid type changes; the EcoBase table selection is keyed on the selection input and clears properly; the wizard's preference matrix is a reactive calc rather than a render side effect; the optimization demo refuses a run where initial points exceed total evaluations, and its summary reports the settings the stored run used; recovery after a crash is judged relative to each group's baseline biomass rather than an absolute floor that small groups could never clear; `scenario_status` reports the years the scenario was built with; group choices are selected by type instead of assuming canonical ordering; prebalance diagnostic plots require unbalanced parameters and say so.

**Batch C, dead UI (11 items).** Wired: fisheries landings/discards editing (a third `set_patch_fn`), the Biomass Forcing card (now writes `ForcedPrey` through the new `apply_prey_forcing` helper), CSV parameter upload (now accepts a model + diet pair), the custom habitat CSV pattern (new `parse_habitat_csv`), the wizard's hexagonal grid choice, the wizard's output (handed to the Ecospace page through a shared reactive value, with user feedback), the forcing demo's Run button, and multi-stanza Save (writes stanza age ranges into `stindiv`). Removed: the Custom fishing scenario option and the habitat-per-group selector, neither of which had any server implementation. `logger.py` no longer configures logging at import time; `configure_logging()` is called from `main()` and attaches the file handler on every run, not only the first.

### Found while fixing, not in the original review

- **An eighth broken download.** `download_params` on the Ecopath page returned a CSV string, so Shiny treated it as a file path and the request failed with a 500 (`OSError: The filename, directory name, or volume label syntax is incorrect`). The review's reviewers had missed this handler. Fixed, and `test_review_fixes.py` now asserts that no `@render.download` handler in any page returns a value instead of yielding.
- **A test that was enforcing a bug.** `test_maximum_size_3km` asserted fewer than 20 hexagons, which is exactly what the pruning code was written to satisfy; the real tessellation of that boundary is 28 patches. The test now checks that large hexagons produce far fewer patches than small ones over the same boundary, and the pruning is gone.
- Two tests pinned shapes that legitimately changed: `pypath_shiny.logger` still exports a module-level `logger`, and the `ecospace_server` signature test now allows optional trailing arguments.

### Deferred, with reason

- **Spatial fishing settings (`pages/ecospace.py`)**: the allocation method, gravity alpha and port settings are never passed to `rsim_run_spatial`, and the gravity preview uses a constant biomass array. Wiring these through the spatial simulation is a feature, not a fix. The Fishing Effort tab now carries a visible "preview only, not applied to the simulation" caption.

  **Resolved on 2026-09-05, see "Spatial fishing implementation" below.**

Still outstanding from the second pass (not part of these 42 app findings): **`pypath/core/indicators.py`** assumes 1-indexed Rpath arrays, so Finn Cycling Index and Transfer Efficiency read 0 on the Analysis page.

Live checks this pass: a fisheries cell edit persisted into the parameters (`OtherGroundfish` landings 0.38 to 0.42, read back from the parameters CSV export); the parameters download returns a real 616-byte CSV where it previously returned a 500. The `apply_prey_forcing` helper was verified directly against a real scenario rather than through the browser, because the slider value did not reach the server from a synthetic DOM event.

---

## Spatial fishing implementation (2026-09-05)

The one item deferred from the third pass is now implemented: the Ecospace page's Spatial Fishing settings drive the spatial simulation.

### The gap

`pypath.spatial.fishing` already had a `SpatialFishing` dataclass and four allocation functions, but nothing connected them to the solver. `rsim_run_spatial` had no parameter for them and `EcospaceParams` no field, so the page's controls only ever drew a preview plot.

### Design

The solver applies a fleet's `ForcedEffort` in **every** patch, so effort is a per-patch density rather than a quantity split across the grid. The allocation functions, by contrast, return an effort that **sums** to the fleet total. Feeding one into the other directly would have silently divided total fishing by the patch count.

The new `effort_multipliers()` bridges them by normalising each gear's allocation to **mean 1.0** across patches. Total fleet effort is therefore unchanged, effort is only redistributed, and a `uniform` allocation reproduces a run with no spatial fishing bit for bit. That backwards-compatibility property is asserted by a test.

Effort allocation reaches the same per-patch hook that MPA closures already used, so the two compose by multiplication: an MPA closure still zeroes a patch that the gravity model would otherwise favour.

### Changes

| File | Change |
|---|---|
| `pypath/spatial/fishing.py` | New `effort_multipliers()`: per-patch, per-gear multipliers for uniform, gravity, port, habitat, prescribed and custom allocations, each normalised to mean 1.0. Any allocation that cannot be computed logs a warning and falls back to uniform. |
| `pypath/spatial/integration.py` | `rsim_run_spatial(..., spatial_fishing=None)`; the multiplier is computed once per month from the biomass at the start of the step (held constant across the RK4 stages, like the MPA mask) and passed to `deriv_vector_spatial` as `effort_multiplier`, where it composes with the MPA mask. |
| `pypath/spatial/__init__.py` | Exports `effort_multipliers`. |
| `pypath-shiny/pages/ecospace.py` | `_build_spatial_fishing()` turns the controls into a `SpatialFishing` and the run passes it. Port indices are validated with a warning instead of a silent fallback. The effort preview now uses real biomass (the latest spatial result, else Ecopath biomass spread over patches) rather than a constant array, and the previously unhandled "habitat" option is drawn. The "preview only" caption is replaced by a note that effort is redistributed relative to the grid mean. |

### Verification

`packages/pypath/tests/test_spatial_fishing_allocation.py`, 11 tests: uniform is all ones; gravity is proportional to biomass and averages 1.0; a higher alpha sharpens the allocation; homogeneous biomass yields uniform effort; habitat allocation targets preferred patches; missing inputs fall back to uniform; uniform matches a run with no spatial fishing; the allocator is called every month with live per-patch biomass; a habitat allocation changes spatial biomass and makes per-patch biomass diverge; a port allocation with no ports does not crash the run.

Live: with gravity selected, a 25-patch grid ran the spatial simulation to completion with no server errors.

One behaviour worth knowing: gravity has **no effect when biomass is homogeneous across patches**, which is the default for a fresh grid with uniform habitat. That is correct, not a bug. Habitat allocation, which depends on the habitat map rather than biomass, changes results immediately.

### Still not wired

`SpatialFishing.target_groups` is honoured by the core but the page has no control for it, so gravity currently follows total biomass across all groups rather than a chosen target species.

### Post-review corrections (same day)

Two follow-ups after the first verification pass:

1. **Preview biomass indexing.** `_preview_biomass()` in `ecospace.py` returned the
   Ecopath biomass vector without an "Outside" row, while `allocate_gravity()` iterates
   `range(1, n_groups)` in the Ecosim layout. The first real group was therefore dropped
   from the preview. A zero row is now prepended so both branches share one layout. This
   only ever affected the sidebar preview; the simulation always passed `current_biomass`,
   which already carries the Outside slot.
2. **MPA composition is now covered by a test.** The doc previously claimed a closure
   still zeroes a patch the allocation favours without exercising the combined path.
   `test_mpa_closure_composes_with_the_allocation` runs the same scenario with habitat
   allocation alone and with a one-patch `MPAConfig`, and asserts the closed patch retains
   more fish biomass.

Final counts: 12 core spatial-fishing tests, 42 page tests, 322 in the full Shiny suite.

## Carried-forward fix: `core/indicators.py` indexing (2026-09-05)

The review noted that `indicators.py` still assumed 1-indexed Rpath arrays, leaving
Finn Cycling Index and Transfer Efficiency dead on the Analysis page. That is now fixed.
It is the same defect class as the `analysis.py` conversion in pass 1.

### What was wrong

Real `Rpath` arrays are 0-based of length `NUM_GROUPS`; `DC` is `(NUM_GROUPS + 1, NUM_LIVING)`
with a trailing Import row; `Landings`/`Discards` are `(NUM_GROUPS, NUM_GEARS)`. The module
was written against a 1-based layout, which produced three distinct failures:

| Site | Symptom |
|---|---|
| `range(1, n_living + 1)` in six functions | First group skipped; loop reads one index past the living groups |
| `rpath.DC[prey, pred]` with 1-based `prey`/`pred` | Diet matrix misaligned by one row and one column |
| `np.sum(rpath.Landings[i, 1:])` | Gear 0 dropped, so a one-fleet model reported **zero catch** |

Measured on the shipped example model, the pre-fix code raised
`IndexError: index 12 is out of bounds for axis 0 with size 12` inside `flow_analysis()`.
That is exactly the exception `calculate_network_indices()` was catching and logging after
pass 1, which is why the page rendered zeros rather than crashing.

On a 5-group test model with detritivory and one fleet:

| Indicator | Before | After |
|---|---|---|
| Finn Cycling Index | 0.00000 | 0.04878 |
| Mean TL of catch | `nan` | 2.70 (the landed group's TL) |
| Catch/Biomass | 0.00000 | 0.01290 |
| Gross efficiency | `nan` | 0.000333 |
| Total living biomass | 21.0 (dropped the producer, counted detritus) | 31.0 |
| P/R ratio | 0.462 | 1.505 |

### Two conventions, deliberately kept apart

`ecosystem_indicators_timeseries()` mixes both layouts: `annual_Biomass`/`annual_Catch` come
from Ecosim and *are* 1-based with index 0 = Outside, while the `rpath.TL`/`rpath.type`
companions in the same expressions are 0-based. Only the Rpath side was converted; Ecopath
group `g` is read at Ecosim index `g + 1`. A blanket search-and-replace would have broken this.

### Why the tests passed before

`test_indicators.py` built `MagicMock` Rpath objects with hand-written 1-based arrays, so the
module and its tests shared the same wrong contract. The fixtures now match the real layout,
and a new `TestRealModel` class runs every indicator against an actual balanced model plus a
fished model that exercises the gear columns and closes a detrital cycle.

### Not changed

`FlowAnalysis.transfer_efficiency` is per-TL-bin and `flow_analysis`/`_finn_cycling_index_from_matrix`
operate on the internally 0-based flow matrix, so no return-length contract changed here —
unlike `keystoneness_index` in pass 1. The `DetFate` TODO in `_build_flow_matrix` is untouched.

A Finn Cycling Index of 0 on the shipped example model is **correct**: its diet matrix has no
detritivory, so the food web contains no cycle.

### Sweep for the same defect elsewhere in core

`range(1, n_living + 1)` / `[1 : n_living + 1]` against an `Rpath`, after the fix:

| Site | Verdict |
|---|---|
| `indicators.py:639`, `indicators.py:659` | Correct — these slice **Ecosim** output, which is 1-based with Outside at index 0 |
| `ibm/integration.py:60` | Correct — indexes `QQ`, an Ecosim array; its docstring says "excluding the 0-index padding" |
| `core/plotting.py:111, 468, 471, 474, 477, 694` | **Same defect, not fixed in this pass** |

`plotting.py` takes an `Rpath` and uses the 1-based layout, so the food web network plot and
the trophic-level spectrum plot drop the first living group and pull the first detritus group
into the living range. `rpath.DC[prey, pred]` is misaligned there for the same reason. This is
a separate, self-contained fix and was left out deliberately rather than expanding this pass.

Confirmed by measurement that the Ecosim convention is genuinely 1-based:
`annual_Biomass.shape == (n_years, NUM_GROUPS + 1)` and `scenario.params.spname[0] == "Outside"`.
`test_timeseries_keeps_the_two_conventions_apart` asserts the mean trophic level of the catch
equals the landed group's TL (2.7); a 0-based reading of the Ecosim array would return 2.0.

## Carried-forward fix: `core/plotting.py` indexing (2026-09-05)

The third and last module carrying the 1-based Rpath assumption. Same defect class as
`analysis.py` (pass 1) and `indicators.py` (earlier today).

### Measured symptom

On the shipped 12-group example model, the pre-fix `plot_foodweb()` raised
`IndexError: index 12 is out of bounds for axis 0 with size 12`. The Analysis page wraps
that call in a `try/except` that draws the error onto the canvas, so users saw the panel
render the text **"Could not plot food web: index 12 is out of bounds..."** rather than a plot.

`plot_trophic_spectrum()` was worse, because it failed silently. `rpath.TL[1 : n_living + 1]`
shifted the window by one group:

| | Groups plotted | Total biomass |
|---|---|---|
| Before | Macroalgae … Seabirds, **plus Detritus** | 34.85 |
| After | **Phytoplankton**, Macroalgae … Seabirds | 44.85 |

The old plot dropped Phytoplankton — the primary producer base — and substituted a detritus
group, a 22% error in a chart whose entire purpose is the trophic pyramid.

### Converted

| Function | Sites |
|---|---|
| `plot_foodweb` | node loop, edge loops, `max(Biomass[...])`, node labels |
| `plot_trophic_spectrum` | all four `[1 : n_living + 1]` slices |
| `plot_foodweb_interactive` | node loop, edge loops, hover text, `max(Biomass[...])`, labels |

Two details that a mechanical shift would have got wrong:

- `is_detritus=i > n_living` became `i >= n_living`. Under 0-based indexing the first
  detritus group sits *at* `n_living`, so the strict comparison mislabels it as living.
- `range(n_total)` for prey now stops before `DC`'s trailing Import row, and `range(n_living)`
  for predators matches `DC`'s column count exactly.

Left alone: `plot_biomass`, `plot_catch`, `plot_biomass_grid`, `plot_biomass_interactive` all
slice `RsimOutput`, which genuinely is 1-based with Outside at index 0. `plot_mti_heatmap`
labels a matrix it never indexes.

### One judgment call beyond indexing

Node labels were `f"G{i}"`. Because node numbering shifts with the fix, `G1` would silently
come to mean a different group than it did yesterday, so labels now use `rpath.Group[i]` via a
new `_group_labels()` helper, falling back to `G{i}` when no usable `Group` array exists.
The Analysis page already passes real names to `plot_mti_heatmap`, so this makes the three
plots consistent. **This is the one change that is not purely an indexing fix and is the
easiest thing to revert if unwanted.**

### Tests

`test_plotting.py` used 1-based `MagicMock` fixtures, so the module and its tests shared the
same wrong contract — the same reason `test_indicators.py` passed over a broken module.
Fixtures now match the real layout, and a `TestRealModel` class asserts against a balanced
model that the graph has exactly `NUM_LIVING + NUM_DEAD` nodes, that node 0 exists, that the
detritus count is right, and that the spectrum bars sum to the living biomass. That last
assertion also checks the shifted sum differs, so it fails under the old slice rather than
passing by coincidence.

With this, no module in `pypath/src` still assumes 1-based Rpath arrays.

## Follow-up: a bug I introduced, and the last two spatial fishing gaps (2026-09-06)

### Regression found and fixed: the Run button was dead

While wiring `target_groups` I found that my own earlier edit had broken the Ecospace page.
Inserting `_build_spatial_fishing()` immediately before `def _run_spatial_simulation():`
placed it **between** the decorators and the function they were meant to decorate:

```python
# Run spatial simulation handler
@reactive.effect
@reactive.event(input.run_spatial_sim)
def _build_spatial_fishing():   # <-- captured the decorators
    ...
    return SpatialFishing(...)  # an effect discards this

def _run_spatial_simulation():  # <-- orphaned, never called
```

So the Run Spatial Simulation button invoked a helper whose return value was thrown away, and
the real handler was dead code. Confirmed against `HEAD`, where the decorators sit on
`_run_spatial_simulation`. The decorators are now reattached.

This is worth recording because **nothing caught it**: the module imports cleanly, every unit
test passed, and ruff was happy. Three structural tests now cover it
(`TestReactiveDecoratorsBindTheRightFunction`), and each was verified to *fail* when the broken
arrangement is deliberately reintroduced:

- no `@reactive.event` handler may be a value-returning `_build`/`_parse`/`_make` helper
- the only function bound to `run_spatial_sim` must be `_run_spatial_simulation`
- `_build_spatial_fishing()` must actually be called by the handler

An AST sweep of every page module for undecorated, never-referenced nested functions found one
other orphan, `ecopath.py:add_header_tooltips` — pre-existing at `HEAD`, not introduced here,
and left alone.

### Gap 1 closed: `target_groups` now has a UI control

A multi-select "Target Groups" appears under the gravity method. Empty means "follow total
biomass", which is `target_groups=None` and the previous behaviour, so the default is unchanged.
Choices are keyed by **Ecosim index**, because the biomass array the allocator sees is 1-based
with index 0 = Outside: Ecopath group `g` is offered as `g + 1`. The selection feeds both the
simulation and the sidebar preview, so the preview no longer contradicts the run.

### Gap 2 closed: the prescribed and custom branches are tested

Nine new tests cover per-gear 2-D allocations, 3-D month selection (including clamping past the
end), a custom callable receiving the live biomass and month, and three fallback paths: a
callable that raises, one that returns the wrong length, and an out-of-range target group.
One test premise was wrong and the code was right — `SpatialFishing("prescribed")` without an
array raises in `__post_init__`, so `effort_multipliers` never sees that case; the test now
asserts the constructor rejects it.

### Live verification (this time actually conclusive)

The earlier "live run completed" claim for the spatial fishing work is **void** — it cannot
have been true if it post-dated the decorator break. Redone properly on port 8771:

1. Uploaded `example_model_data/model.csv` + `diet.csv`, clicked **Balance Model**
2. **Create Scenario** on the Ecosim page ("No scenario created" notice cleared)
3. Ecospace → **Create Grid** → "Patches: 25, Connections: 40, Avg neighbors: 3.2"
4. Spatial Fishing panel → method `gravity`, **Target Groups = Demersal fish**
   (the picker listed all 12 real group names, confirming `_populate_target_groups`)
5. **Run Spatial Simulation**

Proof the handler fired, rather than merely an absence of errors: the "View Biomass for Group"
dropdown came back populated with all 12 group names. That `ui.update_select` call lives
*inside* `_run_spatial_simulation`, after `rsim_run_spatial()` returns, so it can only run if
the button reached the handler and the simulation completed. While the decorators were
misplaced that dropdown stayed empty.

Server error count for the whole session: **0**.
