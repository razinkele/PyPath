# EwE 6.7.144 oracle numbers: Black Sea EwE-F test case, Ecospace scenario 1

These numbers come from a headless run of the **unmodified** EwE 6.7.144 core (tag `v6.7.144`, commit 2165a98), run on a Windows runner with the Access Database Engine x64. The run was ewe-oracle run 43288513485 (2026-11-12), with the optional Ecospace stage (`--ecospace 1`). It used `Data/BlackSea_EwEF_testcase.eweaccdb`, sha256 e4949ea9…6c44c6d.

Following roadmap decision D3, only numbers are committed here; the harness and EwE stay outside the repo. Only the rows the tests read are kept. The full run output (area means at every step, cell maps, habitat capacity) is in the private oracle repository; see `docs/parity/ecospace_spike.md`.

| File | Rows | Content |
|---|---|---|
| `ecospace_water_cells.csv` | the 304 water cells of the 31 x 42 grid | `cEcospaceDataStructures.Depth(row, col)` > 0, after `UpdateDepthMap`; rows and columns 1-based |
| `ecospace_habcap.csv` | the 304 water cells | `HabCap(group)(row, col)` for the 22 groups after `SetHabCap` (normalised, with the low-capacity gradient); rows and columns 1-based |
