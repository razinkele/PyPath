# EweTimer: time EwE 6's engine for `benchmark_engines.py`

EweTimer loads an EwE model with EwE 6's own engine (`EwECore.dll` from your
EwE 6 installation), times the Ecopath solve and Ecosim runs of several
lengths, and writes the CSV that `scripts/benchmark_engines.py report
--ewe-csv` reads (`model,what,years,repeat,seconds`). Optionally it writes each
group's biomass at the end of the shortest run, to check that EwE 6 and PyPath
ran the same case.

EwE 6's Ecosim always integrates with RK4, one step a month, the same scheme
as PyPath's EwE rule.

## Build (once)

Needs the .NET SDK (8 or newer, https://dotnet.microsoft.com/download) and
EwE 6 (64-bit) installed. In this folder:

```powershell
dotnet build -c Release -p:EweDir="C:\Program Files\Ecopath with Ecosim 40 years 6.7.0.18865_64-bit"
```

`EweDir` is the folder that holds `EwECore.dll`; use your installation's
folder. The program is built for .NET Framework 4.8, as EwE 6's desktop
release is. If a newer EwE release is built for .NET 10, change
`<TargetFramework>` in `EweTimer.csproj` to `net10.0-windows`.

## Run

```powershell
.\bin\Release\net48\EweTimer.exe --ewe-dir "C:\Program Files\Ecopath with Ecosim 40 years 6.7.0.18865_64-bit" `
    --model "C:\PyPath_optimize\PyPath-git\Data\LT2022_0.5ST_final7.eweaccdb" `
    --name LT2022 --scenario 1 --years 10 20 40 --repeats 5 --warmup 3 `
    --out ewe6.csv --biomass-out ewe6_end_biomass.csv
```

Run it once per model; rows are appended to the same CSV. Close other
programs while it runs, and run PyPath's side (`benchmark_engines.py
database`) on the same machine.

The model file is not changed: the run length is set in memory only.
