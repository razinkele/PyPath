# EwE 6.7.144 oracle numbers: LT2022, Ecosim scenario 16 + time-series dataset 35

These numbers come from a headless run of the **unmodified** EwE 6.7.144 core (tag `v6.7.144`, commit 2165a98), built for net10.0 and run on a Windows runner with the Access Database Engine x64. The run (2026-11-10) used `LT2022_0.5ST_final7.eweaccdb`, sha256 d923ef28…43bd.

Following roadmap decision D3, only numbers are committed here; the harness and EwE stay outside the repo. Only the rows the tests read are kept. The full run output is in the private oracle repository and can be regenerated in a few minutes.

| File | Rows | Content |
|---|---|---|
| `ecosim_biomass_monthly.csv` | months 1, 15, 17, 19 | `cEcosimGroupOutput.Biomass(t)`: biomass at the end of month t |
| `ecosim_feeding_time_monthly.csv` | months 15–20 | `FeedingTime(t)`: Ftime after the end-of-month update, i.e. the value used in month t + 1 |

EwE computes in single precision. In this run, `UseVarPQ` is False at run time (EwE's `ReadSafe` bug; deviation R1).
