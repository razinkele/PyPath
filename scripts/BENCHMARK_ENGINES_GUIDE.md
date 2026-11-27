# Timing PyPath and EwE 6 on the same models

Both engines run on one Windows PC, one after the other, and read the same EwE
database (`.eweaccdb`): EwE 6 through `scripts/ewe6_timer`, PyPath through
`scripts/benchmark_engines.py database`. `benchmark_engines.py report` then
prints milliseconds per simulated year for each.

| Model | File | Ecosim scenario |
|---|---|---|
| LT2022 | `Data/LT2022_0.5ST_final7.eweaccdb` | 16 |
| Black Sea | `Data/BlackSea_EwEF_testcase.eweaccdb` | 1 |
| Curonian Lagoon, Central Baltic, Lithuanian Coast | PyPath examples, written as databases (step 2) | 1 |

All commands run in PowerShell from the repository folder,
`C:\PyPath_optimize\PyPath-git`. `$py` is your PyPath environment's Python:

```powershell
$py = "C:\Users\burak\anaconda3\envs\pypath\python.exe"
```

## 1. Install, once

1. **EwE 6, 64-bit**, from https://ecopath.org (Downloads). Note its install
   folder, the one that holds `EwECore.dll`.
2. **Microsoft Access Database Engine, 64-bit** (EwE 6 and PyPath both read
   `.eweaccdb` through it), then `& $py -m pip install pyodbc`.
3. **.NET SDK** 8 or newer, https://dotnet.microsoft.com/download.
4. Build the timer (see `scripts/ewe6_timer/README.md`):
   ```powershell
   cd scripts\ewe6_timer
   dotnet build -c Release -p:EweDir="C:\Program Files\Ecopath with Ecosim 40 years 6.7.0.18865_64-bit"
   cd ..\..
   ```
   Use your own EwE folder in `EweDir`.

## 2. Write the example models as EwE databases

```powershell
mkdir bench
foreach ($m in "Curonian_Lagoon", "Central_Baltic", "Lithuanian_Coast") {
  & $py scripts\convert_for_engines.py --example $m --out "bench\$m.eweaccdb"
}
```

Each run prints how closely PyPath's run of the new database follows its run of
the example. Open one of the files in EwE 6 once to check that it loads.

## 3. Time EwE 6

```powershell
$ewe = "C:\Program Files\Ecopath with Ecosim 40 years 6.7.0.18865_64-bit"
$timer = "scripts\ewe6_timer\bin\Release\net48\EweTimer.exe"
& $timer --ewe-dir $ewe --model Data\LT2022_0.5ST_final7.eweaccdb --name LT2022 --scenario 16 --out bench\ewe6.csv --biomass-out bench\ewe6_biomass.csv
& $timer --ewe-dir $ewe --model Data\BlackSea_EwEF_testcase.eweaccdb --name BlackSea --scenario 1 --out bench\ewe6.csv --biomass-out bench\ewe6_biomass.csv
foreach ($m in "Curonian_Lagoon", "Central_Baltic", "Lithuanian_Coast") {
  & $timer --ewe-dir $ewe --model "bench\$m.eweaccdb" --name $m --scenario 1 --out bench\ewe6.csv --biomass-out bench\ewe6_biomass.csv
}
```

## 4. Time PyPath on the same databases

```powershell
& $py scripts\benchmark_engines.py database --database Data\LT2022_0.5ST_final7.eweaccdb --scenario 16 --out bench\pypath_LT2022.json
& $py scripts\benchmark_engines.py database --database Data\BlackSea_EwEF_testcase.eweaccdb --scenario 1 --out bench\pypath_BlackSea.json
foreach ($m in "Curonian_Lagoon", "Central_Baltic", "Lithuanian_Coast") {
  & $py scripts\benchmark_engines.py database --database "bench\$m.eweaccdb" --scenario 1 --out "bench\pypath_$m.json"
}
```

Close other programs while steps 3 and 4 run.

## 5. The tables

```powershell
& $py scripts\benchmark_engines.py report bench\pypath_LT2022.json bench\pypath_BlackSea.json bench\pypath_Curonian_Lagoon.json bench\pypath_Central_Baltic.json bench\pypath_Lithuanian_Coast.json --ewe-csv bench\ewe6.csv
```

It prints the Ecopath solve time and the Ecosim time per simulated year (the
slope over runs of 10, 20 and 40 years) for each engine and model.
`bench\ewe6_biomass.csv` holds EwE 6's biomass of each group at the end of a
10-year run, to compare with PyPath's and confirm that both ran the same case.
