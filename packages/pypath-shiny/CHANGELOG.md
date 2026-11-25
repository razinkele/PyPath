# Changelog — pypath-shiny

All notable changes to the **pypath-shiny** web frontend.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

## v1.23.0 (2026-11-25)

### Features

- **shiny**: Home and About pages brought up to date; a tighter Home
  ([`56c4acd`](https://github.com/razinkele/PyPath/commit/56c4acd315839d7ca3636bc49e3350a7e21f952f))


## v1.22.0 (2026-11-25)

### Features

- **ecospace-page**: The box output panel takes a monthly box file
  ([`d691a90`](https://github.com/razinkele/PyPath/commit/d691a90f18c3f51b3d2980ea61476565abf1daa2))


## v1.21.1 (2026-11-24)


## v1.21.0 (2026-11-24)

### Features

- **shiny**: Load the Curonian Lagoon example's own Ecospace set-up
  ([`f3e88c0`](https://github.com/razinkele/PyPath/commit/f3e88c034cf5d116f8070fa36d41050557ba04a7))


## v1.20.0 (2026-11-24)

### Bug Fixes

- **shiny**: The run-time estimate counts the sub-steps a month
  ([`29b109e`](https://github.com/razinkele/PyPath/commit/29b109e5c6ed7cfedf40b9bbbf5fdd71bc5d2b4c))

- **shiny**: The spatial run-time estimate follows measured runs
  ([`74f288e`](https://github.com/razinkele/PyPath/commit/74f288e9a2561ea0c1fedf2f95d5d2d4c78cbf6c))

### Features

- **shiny**: Draw a stored Ecospace scenario's cells where they lie on the ground
  ([`fd2357d`](https://github.com/razinkele/PyPath/commit/fd2357d8fb870bb96fa4795aa655ad0b9c225488))


## v1.19.0 (2026-11-24)

### Bug Fixes

- A stored Ecospace scenario stays with the groups it was read for
  ([`a53aa93`](https://github.com/razinkele/PyPath/commit/a53aa93a7873aa2d16d179af69da204a14ddf26f))

- **shiny**: Check a stored Ecospace scenario's grid size before reading it
  ([`9b2974f`](https://github.com/razinkele/PyPath/commit/9b2974ffdfc28aac4ccb01fbb2b953798a9da7f6))

### Features

- **shiny**: Open an example on the Ecosim page; load a stored Ecospace scenario or an example's
  grid
  ([`3e1e3e7`](https://github.com/razinkele/PyPath/commit/3e1e3e7bb54bd605097b1abff544c7d6d1fc945c))


## v1.18.0 (2026-11-22)

### Bug Fixes

- **shiny**: Box output on a grid with named patches, and a plain moderate message
  ([`a9464be`](https://github.com/razinkele/PyPath/commit/a9464bebe284111b09d310c1570170704e3c5235))

- **shiny**: The sub-step check compares integrated groups, and server-side bounds
  ([`295fead`](https://github.com/razinkele/PyPath/commit/295feaddf40256486400f6881d315675463ddadb))

### Features

- **shiny**: A SHYFEM box model's fields and water transport on the Ecospace page
  ([`8a05c8f`](https://github.com/razinkele/PyPath/commit/8a05c8f216cb9d70e72db786704fe94949d9936e))


## v1.17.0 (2026-11-21)

### Bug Fixes

- **spatial**: Bound the mesh reader and the upload; docstrings the docs build accepts
  ([`f878a01`](https://github.com/razinkele/PyPath/commit/f878a01ac4c9034e813043838ad49e22678ba58d))

### Features

- **shiny**: SHYFEM mesh as an Ecospace grid type
  ([`4c9391f`](https://github.com/razinkele/PyPath/commit/4c9391f0adde597bf325a51c46a668cbf49a0361))


## v1.16.0 (2026-11-20)

### Bug Fixes

- **shiny**: Black lines and markers stay visible on a dark chart
  ([`7a4643c`](https://github.com/razinkele/PyPath/commit/7a4643cc1db762e5a12a3220aa0353fea2752a0f))

### Features

- **shiny**: Charts drawn for the dark theme; dark fixes on the Home page
  ([`d13ec68`](https://github.com/razinkele/PyPath/commit/d13ec688f9059977afa6bab705e5084a7c9eb8c1))


## v1.15.0 (2026-11-20)

### Features

- **shiny**: A dark theme that follows the header's toggle
  ([`4879a27`](https://github.com/razinkele/PyPath/commit/4879a27f5a783adbdd1e88698dfc0507baeaa611))


## v1.14.0 (2026-11-20)

### Features

- **shiny**: Download buttons carry a download icon
  ([`5e9746a`](https://github.com/razinkele/PyPath/commit/5e9746afc641290786438f355e7f1499f477b3e3))

- **shiny**: Ecospace runs can use worker processes, opt-in
  ([`b6ca6e7`](https://github.com/razinkele/PyPath/commit/b6ca6e7b9c58cc7630249a2c81d2226ae3dcb233))


## v1.13.0 (2026-11-20)

### Bug Fixes

- **shiny**: Cell tooltips follow edits, see unrounded numbers, match labels exactly
  ([`1c6f238`](https://github.com/razinkele/PyPath/commit/1c6f238d57a1b544b4f91b3571354f07bd987cd8))

### Features

- **shiny**: A tooltip on every table cell, tables that scroll under fixed headings
  ([`739439b`](https://github.com/razinkele/PyPath/commit/739439b38690a34e2468295c7630e20011eae845))

- **shiny**: Tables that scroll show every group; a wide table keeps its row names
  ([`f394a66`](https://github.com/razinkele/PyPath/commit/f394a663d7e53b901b6e3c79937ae936da32d19a))

- **shiny**: Unrounded values on great_tables tables; no stale or false-precision tooltips
  ([`6a5812a`](https://github.com/razinkele/PyPath/commit/6a5812a9f812d2be10ca74d47afeac23b24f19cd))


## v1.12.0 (2026-11-20)

### Bug Fixes

- **shiny**: Clear a study area of any origin; area across the antimeridian
  ([`1155cc3`](https://github.com/razinkele/PyPath/commit/1155cc3f6e3408d1ba6001b2ca9bc66ba60473a1))

- **shiny**: The Ecospace wizard's study-area map works, on shiny-deckgl
  ([`4dfd7ff`](https://github.com/razinkele/PyPath/commit/4dfd7ffa23a1703b4463f2b273560cdccb8671ed))

### Features

- **shiny**: Cancel and month progress for Ecospace runs
  ([`a96d6fe`](https://github.com/razinkele/PyPath/commit/a96d6feaa7468ac35920ce0312921ef3639af0f7))


## v1.11.0 (2026-11-19)

### Features

- **shiny**: The Data Import page names SeaLifeBase and shows the trait source
  ([`6bf107a`](https://github.com/razinkele/PyPath/commit/6bf107a1894d88adb68dfbba84fb292293c1a2e0))


## v1.10.0 (2026-11-19)

### Bug Fixes

- **shiny**: A full-screen map is never smaller than its normal height
  ([`9c09c38`](https://github.com/razinkele/PyPath/commit/9c09c3871f90d222ff370a48aa2b7376f9a5f03b))

### Features

- **shiny**: An interactive food-web network on the Results page
  ([`2361b06`](https://github.com/razinkele/PyPath/commit/2361b06b6fef2b82faf31b5445b499e26c35b1e6))

- **shiny**: Busy run buttons, full-screen plot cards, dark Ecospace maps
  ([`2bff3ef`](https://github.com/razinkele/PyPath/commit/2bff3efeeec3768e11b72035abd00019734082ca))


## v1.9.0 (2026-11-19)

### Bug Fixes

- **shiny**: Ecospace and IBM runs no longer freeze every session
  ([`0080602`](https://github.com/razinkele/PyPath/commit/0080602cea4ffff79082c0e33fafcfcaa2cb22e6))

### Features

- **shiny**: Note when the scenario was edited after a run started
  ([`0c3384f`](https://github.com/razinkele/PyPath/commit/0c3384fae089b22b2660653ffea4116ad7f660b6))

### Refactoring

- **shiny**: Replace deprecated Shiny calls
  ([`1722c36`](https://github.com/razinkele/PyPath/commit/1722c366adf3bc4e55cfdc29cde5674f6db31b6f))


## v1.8.1 (2026-11-18)

### Bug Fixes

- **tutorial**: Adjust_fishing and set_vulnerability snippets use real keywords
  ([`2a40086`](https://github.com/razinkele/PyPath/commit/2a40086b871f2dac175feacf4e794928949c3645))


## v1.8.0 (2026-11-18)

### Bug Fixes

- **ecosim**: Refuse groups out of block order; ensembles read biomass by type
  ([`c5befca`](https://github.com/razinkele/PyPath/commit/c5befcaff9368c6aca9424ff984539e6f037228a))

### Features

- **import**: Accept SQLite copies of EwE databases in the EwE File upload
  ([`ea5c336`](https://github.com/razinkele/PyPath/commit/ea5c336130b82f0b8a26cfcd355f5b490f57a6d2))


## v1.7.0 (2026-11-18)


## v1.6.0 (2026-11-17)

### Bug Fixes

- **d1**: Harden the origin rule after review
  ([`c033978`](https://github.com/razinkele/PyPath/commit/c0339786e787a8ef4050ff42faad53d113681c65))

### Features

- **shiny**: The Ecosim rule select follows the model's origin (D1)
  ([`153d698`](https://github.com/razinkele/PyPath/commit/153d698c45ecc50f6c778d620b519a2ba2cede37))


## v1.5.0 (2026-11-17)

### Bug Fixes

- **shiny**: A Balance of new parameters is not a re-balance
  ([`22dd1c3`](https://github.com/razinkele/PyPath/commit/22dd1c3e521d786df9d09216a74de4517370e9b3))

- **shiny**: A closed Ecosim tab cancels its run
  ([`a9c27b2`](https://github.com/razinkele/PyPath/commit/a9c27b291bb05094b4db99fa66edb0ae36f07e83))

- **shiny**: A load clears the new-params flag
  ([`ac72773`](https://github.com/razinkele/PyPath/commit/ac72773056afde2933f34f40e0a8c5c9b2705e8b))

- **shiny**: A model load drops the old scenario and results; runs say Starting until a worker
  begins
  ([`44b826b`](https://github.com/razinkele/PyPath/commit/44b826bc753ff38f853e6973ffe8fe12da55cca5))

- **shiny**: Bound Ecosim worker threads; a session waits for its cancelled run to finish
  ([`f990400`](https://github.com/razinkele/PyPath/commit/f99040030dba5d1cd1794367e99010c2d34add79))

- **shiny**: Cancel texts say the run stops at the end of its month
  ([`1093ef6`](https://github.com/razinkele/PyPath/commit/1093ef6be5ae9020aea2a6e0c788edd61a67a77d))

- **shiny**: Ecosim final-review fixes: empty qs rate warns, honest cancel texts, relevant hints
  ([`b0f91bd`](https://github.com/razinkele/PyPath/commit/b0f91bd53c9876f18c462232d870f66d6d0f1d38))

- **shiny**: Finish a stopped run as cancelled; firmer abort drives
  ([`1c78b6c`](https://github.com/razinkele/PyPath/commit/1c78b6c8bc452e6dd4b978329cbc2d2e55ffb513))

- **shiny**: Help states the 3-decimal rule as it is; ui drives run off Linux
  ([`cf7443b`](https://github.com/razinkele/PyPath/commit/cf7443bf4bc2011589071c5f89d6b87a2abed783))

- **shiny**: No pending note beside a control that shows a message
  ([`1544b2d`](https://github.com/razinkele/PyPath/commit/1544b2d7675287ff8a5ddae45c5f7248a48a3a6f))

- **shiny**: Only "Create parameters" makes the next Balance a new model
  ([`1a5e58d`](https://github.com/razinkele/PyPath/commit/1a5e58d116ec4eec115922e57446555bc219d47c))

- **shiny**: Pin "untouched means unchanged" in one Create-time step
  ([`1a3260a`](https://github.com/razinkele/PyPath/commit/1a3260a507e1c1590b745857d76f2a430c84abea))

- **shiny**: Prey-switching note only while values differ; ask for a model
  ([`7055010`](https://github.com/razinkele/PyPath/commit/7055010f8a31c4b2f4703434426e2ace16fe61db))

- **shiny**: The cleared-settings note no longer claims a load
  ([`16bd2e2`](https://github.com/razinkele/PyPath/commit/16bd2e202f65d1fd125b8b8fb75e9351c33db275))

- **shiny**: The prey switching example code runs as written
  ([`c74bb40`](https://github.com/razinkele/PyPath/commit/c74bb4000935711dbab56bdf7dc37aea79ceaa6a))

- **shiny**: The scenario-source radio keeps the user's choice on re-render
  ([`4de5651`](https://github.com/razinkele/PyPath/commit/4de5651e343bfd57bdb686d41247e9c951de4080))

- **shiny**: The switching note shows when the scenario came from another source
  ([`ad7936e`](https://github.com/razinkele/PyPath/commit/ad7936e3b48b81bf60fdfee4e5c53af8e591e744))

### Features

- **shiny**: Continuous methods, tolerance preset and model options on the Ecosim page
  ([`a0b597b`](https://github.com/razinkele/PyPath/commit/a0b597bcf139f579542678924898392ab0c955d4))

- **shiny**: Ecosim page stops a run at its next month and shows months done
  ([`c98805d`](https://github.com/razinkele/PyPath/commit/c98805dcf52f658d9335ad33d3130e29a00d4bfa))

- **shiny**: Every run shows the prey switching it used
  ([`722705a`](https://github.com/razinkele/PyPath/commit/722705a38e221aa4ff60cad4f68d1e0ce4690792))

- **shiny**: Per-predator prey switching on the Ecosim page
  ([`4b89f78`](https://github.com/razinkele/PyPath/commit/4b89f78e978d654062340a925416d6fb531aa099))

- **shiny**: Prey-switching service
  ([`8465121`](https://github.com/razinkele/PyPath/commit/8465121698535b10d7c4382e03a26dbc990d7c4a))

- **shiny**: Re-balancing keeps prey-switching settings, a new model clears them
  ([`a094fbf`](https://github.com/razinkele/PyPath/commit/a094fbfd0914553d8998f00b7f21aaca18409470))

- **shiny**: Run Ecosim off the event loop (extended task); cancel discards the result
  ([`026bf0d`](https://github.com/razinkele/PyPath/commit/026bf0d4ff494610d52bfe2a4b277432aa359f63))

- **shiny**: Run the continuous methods from the Ecosim page; invalid combinations notify; diet
  rewiring hidden
  ([`2aef2a5`](https://github.com/razinkele/PyPath/commit/2aef2a507b4d813ec11414bdb33d9b8965eec76a))

- **shiny**: RunControl stops a run at its next month and reports progress
  ([`6ee4bf3`](https://github.com/razinkele/PyPath/commit/6ee4bf3b63c30f72663520a02c73b25bc15e27e0))

- **shiny**: Solver diagnostics panel for continuous Ecosim runs
  ([`2fe6340`](https://github.com/razinkele/PyPath/commit/2fe6340f845798735d75d7b84e647ed084574771))

- **shiny**: Solver request service for the continuous Ecosim methods
  ([`4889356`](https://github.com/razinkele/PyPath/commit/4889356a1733fcfe7cef98a2ae857cda05ff4bc8))

- **shiny**: The demo page shows EwE's prey switching
  ([`c1881e7`](https://github.com/razinkele/PyPath/commit/c1881e7d3bdd165f3f1ec0e32add1801f1ded9b8))

- **shiny**: The switching line also sits on the Ecosim Summary tab
  ([`4e11cc2`](https://github.com/razinkele/PyPath/commit/4e11cc278a4b883d075cbf8f939723b731cacf0e))


## v1.4.0 (2026-11-14)

### Bug Fixes

- **deploy**: Keep the package's examples folder
  ([`9db7d1c`](https://github.com/razinkele/PyPath/commit/9db7d1ce842002ada6ba1a1d5b530e3600e2e6d3))

- **shiny**: Consumer P/B limit 200 a year
  ([`8288750`](https://github.com/razinkele/PyPath/commit/828875073b113ff31e16e9ffc855edb1d58f3b80))

- **shiny**: Dispersal note: wizard precedence, a rate on square cells (review)
  ([`4ba6767`](https://github.com/razinkele/PyPath/commit/4ba6767e5c8fbebf27575438666d7e572d5eaaac))

- **shiny**: Ecospace page gives group 1 its habitat and movement (no Outside row)
  ([`2d561f9`](https://github.com/razinkele/PyPath/commit/2d561f926e7adfdc69763f2409a4bcb0c5f0401f))

- **shiny**: Ecospace page runs the Ecospace Wizard's configuration
  ([`bea26b5`](https://github.com/razinkele/PyPath/commit/bea26b5dfdfa4dd1aa4109aa61c2a098b0b44066))

- **shiny**: Ecospace Wizard builds its configuration when depth or salinity is loaded
  ([`1dec86d`](https://github.com/razinkele/PyPath/commit/1dec86d2576c75a106081f538c74358f5db79f17))

- **shiny**: Label the Ecospace dispersal input as a rate per year
  ([`4441d84`](https://github.com/razinkele/PyPath/commit/4441d84477752a80d7a615e855975e9b95151828))

- **shiny**: Loading a model on the Home page clears old results
  ([`3e4f780`](https://github.com/razinkele/PyPath/commit/3e4f7802907dc7b847685e52a8a5206d0d0219d9))

### Features

- **shiny**: Open the example models from the Home page
  ([`9909684`](https://github.com/razinkele/PyPath/commit/990968442f129800b6218687efbd9d14689c6404))


## v1.3.1 (2026-11-12)

### Bug Fixes

- **stanzas**: EwE's age table and Ford-Walford plus group (EwE rule)
  ([`e32412a`](https://github.com/razinkele/PyPath/commit/e32412a97a20363a74744efbe4dc2a06572d2be6))


## v1.3.0 (2026-11-09)

### Bug Fixes

- Warn about detritus that flows to itself; Shiny defaults it to 0
  ([`a824df8`](https://github.com/razinkele/PyPath/commit/a824df897ae754d839e92dd90d2b512ff9bd699d))

- **shiny**: Home example detritus no longer flows to itself
  ([`05ecdf4`](https://github.com/razinkele/PyPath/commit/05ecdf498d5633fba4b0a34c775af2b6eab8e221))

### Features

- **shiny**: Run an imported EwE database's Ecosim scenario; choose the fast-group rule
  ([`4e5fc9e`](https://github.com/razinkele/PyPath/commit/4e5fc9e06b032d919c3d16d46509b2d482490a5f))


## v1.2.0 (2026-11-04)

### Features

- **shiny**: Phase 3 - Plotly charts, deck.gl Ecospace maps, browser-check fixes
  ([#46](https://github.com/razinkele/PyPath/pull/46),
  [`032a56a`](https://github.com/razinkele/PyPath/commit/032a56abc99c52d7f5161d8f83f87114c5d24dce))


## v1.1.0 (2026-11-03)

### Bug Fixes

- **shiny**: Data Import panel was invisible (sidebar outside a layout)
  ([`3c0b71f`](https://github.com/razinkele/PyPath/commit/3c0b71fcdca3a4622854af5b0eee79b6ac366f1d))

- **shiny**: Ecosim "Create Scenario" failed; charts sat below the window
  ([`49b66ee`](https://github.com/razinkele/PyPath/commit/49b66ee0ab19f08a4200529ded41d4abb1235d78))

- **shiny**: Give every Plotly widget a height (Ecosim Summary charts were ~1600 px)
  ([`1877eac`](https://github.com/razinkele/PyPath/commit/1877eac2be8d4d6c3588a0f31b0686805a4344a6))

- **shiny**: Handle proxied health endpoint paths
  ([`0daf001`](https://github.com/razinkele/PyPath/commit/0daf001519a6e40c28c332bc921033b76a7f46a0))

- **shiny**: Keep fleet discard fates when balancing
  ([`4b9c728`](https://github.com/razinkele/PyPath/commit/4b9c7280a99be59e1a1a295337be5c0f74ae8a6e))

- **shiny**: Readable header text in format_params_table
  ([`bdd118d`](https://github.com/razinkele/PyPath/commit/bdd118ddc8fa91f02028c3cf2a8f94faf979d2d5))

- **shiny**: Repair the Phase 2 great_tables integration
  ([`1c13b33`](https://github.com/razinkele/PyPath/commit/1c13b338105d84f2641fcffb019f8a82be88d409))

- **shiny**: Restore pypath-ewe dependency floor >=1.0.0
  ([`f08f58d`](https://github.com/razinkele/PyPath/commit/f08f58da421e0c32d1535871874a36244188cf7d))

- **shiny**: Show a message instead of an empty table in @render.ui outputs
  ([`f77c777`](https://github.com/razinkele/PyPath/commit/f77c777d2bcd5b5659606807891cb48396d440e6))

- **shiny**: Source health endpoint from pypath-ewe
  ([`5c960f0`](https://github.com/razinkele/PyPath/commit/5c960f0dd7bf1037254eb51f98c0c8d83bc819a0))

### Features

- Convert render.data_frame to great_tables in 3 pages
  ([`2da25e7`](https://github.com/razinkele/PyPath/commit/2da25e7d510225e4a477275ad1bec524ea4b56d2))

- **shiny**: Add /api/health endpoint
  ([`c45b751`](https://github.com/razinkele/PyPath/commit/c45b75161c0230dcda4e176f81e31130883d1735))

- **shiny**: Add GT formatters service for professional tables
  ([`d24a729`](https://github.com/razinkele/PyPath/commit/d24a729494e4b2e082e589ac896b16b3eb3b4b0d))

- **shiny**: Add modern UI stack dependencies
  ([`96621c6`](https://github.com/razinkele/PyPath/commit/96621c67a2f79dcec2a53bba0ab1368bd09aeb20))

- **shiny**: Ecopath trophic-level and EE charts on Plotly
  ([`857ca4e`](https://github.com/razinkele/PyPath/commit/857ca4e85099eb5c7480aca6752dffd6718c2d3d))

- **shiny**: Integrate great_tables into data import page
  ([`6600b5d`](https://github.com/razinkele/PyPath/commit/6600b5dbb87356f5f7ad08d99d19f3ea813e68c2))

- **shiny**: Integrate great_tables into ecopath page
  ([`e1c94bb`](https://github.com/razinkele/PyPath/commit/e1c94bb20284575a8820fa8a955c9195dabc60f8))

- **shiny**: Integrate great_tables into prebalance diagnostics tables
  ([`a80d0e8`](https://github.com/razinkele/PyPath/commit/a80d0e8355690eaa66ea67f1098e9aeb16ca6211))

- **shiny**: Integrate great_tables into results page and add time_series formatter
  ([`1ec5918`](https://github.com/razinkele/PyPath/commit/1ec591816d98025385093bab1e1a87c11984ef54))

- **shiny**: Interactive effort preview with one line per fleet
  ([`32e88aa`](https://github.com/razinkele/PyPath/commit/32e88aae688a6d8d97a589083906494fef28b93b))

- **shiny**: Interactive Plotly biomass and catch plots on the Ecosim page
  ([`92e2264`](https://github.com/razinkele/PyPath/commit/92e2264f1d94e3b48aa52082e59a3fdcb696e54b))

- **shiny**: Interactive Plotly charts on the Analysis page
  ([`56b827e`](https://github.com/razinkele/PyPath/commit/56b827ecad69003f8ee140c6872a4f48d9a82f49))

- **shiny**: Interactive Plotly charts on the Ecosim Summary tab
  ([`b1776bc`](https://github.com/razinkele/PyPath/commit/b1776bcfff2a83e7739f2456ca26d6076da75ba4))

- **shiny**: Phase 3 (WIP) - Plotly figures, UI components, home redesign
  ([`c472b6e`](https://github.com/razinkele/PyPath/commit/c472b6eaef3ddd9d39f5d36c4bddbc561c53ecc6))

- **shiny**: Results trophic charts and food web on Plotly; palette/style fixes
  ([`6ebae01`](https://github.com/razinkele/PyPath/commit/6ebae018f773c1d520f33833dbf56dd7232155a8))


## v0.4.2 (2026-09-06)

### Fixed

- **Ecospace**: The Spatial Fishing controls (allocation method, gravity alpha, port distance decay, habitat targeting) now drive the simulation instead of only redrawing a preview. Added a Target Groups multi-select for `SpatialFishing.target_groups`; leaving it empty follows total biomass, as before.
- **Ecospace**: Fixed a decorator that had been captured by a helper inserted above the run handler, which left the Run Spatial Simulation button doing nothing.
- **Ecospace**: Removed a pruning block that discarded 9 of 28 hexagonal patches; habitat CSV parsing now sniffs for a header instead of eating the first data row.
- **Ecopath**: Table edits are applied through `set_patch_fn`; the previous `input.<id>_cell_edit` effects do not exist in Shiny 1.7, so edits to the model, diet and fisheries tables were silently discarded. Rows resolve by group name rather than position.
- **All pages**: Eight `@render.download` handlers returned a `str`, which Shiny treats as a file path; `download_params` raised `WinError 123`. All now yield.
- **Ecosim**: The Biomass Forcing card now writes into `forcing.ForcedPrey`; previously it had no effect on the scenario.
- **Analysis**: Food web, trophic spectrum and network indices work again, following the indexing fixes in pypath-ewe 0.5.0.
- **logger.py**: `configure_logging()` clears existing handlers and creates the log directory on every run.

### Changed

- Requires `pypath-ewe >= 0.5.0`.


## v0.4.1 (2026-04-06)

### Fixed
- **pages/analysis.py**: Return `ui.div()` instead of `None` from `render.ui` (prevents blank panel crash)
- **pages/ecosim.py**: Guard `scen is not None` before accessing `scen.params`

### Tests
- Comprehensive test coverage: 143 new tests (280 total, up from 137)
- New test files covering: config dataclasses + IBM/SmeltParams alignment, all 5 validation
  functions, `format_dataframe_for_display` (sentinel masking, rounding, type labels, stanza/remarks),
  `create_cell_styles` (no_data priority, QB non-applicable, remarks, stanza styles),
  `get_model_info` (balanced/unbalanced models), `load_rpath_diagnostics` (corrupted JSON, missing
  CSVs), ecopath helpers (`_get_groups_from_model`, `_recreate_params_from_model` with diet
  reconstruction), `_get_version`, tutorial helpers (`_code_block`, `_step_card` with badge),
  `_resolve_repo_root` (env var, walk-up), UI render assertions for all page modules,
  demo page server signature checks



### Changed
- Updated to use pypath-ewe v0.3.3 (time series calibration, mediation,
  Monte Carlo, ecotracer, fleet dynamics, ecological indicators, value
  chain I/O, taxonomy integration — 86% EwE table coverage)

## v0.3.2 (2026-03-11)

### Changed
- Updated to use pypath-ewe v0.3.2 (EwE 6.6+ schema compatibility fix)

## v0.3.1 (2026-03-10)

### Added
- Ecospace Data Wizard improvements with KDTree-based environmental sampling

### Changed
- Updated to use pypath-ewe v0.3.1 core improvements

## v0.3.0 (2026-03-09)

### Added
- Ecospace Data Wizard page with EMODnet integration
- Sidebar navigation with Bootstrap Icons

### Changed
- Replaced inline hasattr() checks with is_balanced_model()/is_rpath_params() utilities
- Replaced brittle parents[5] with _resolve_repo_root() in prebalance.py
- Ruff lint config moved to [tool.ruff.lint]

## v0.2.2 (2025-12-19)

- Initial release (see commit history)
