// EweTimer: times EwE 6's engine for scripts/benchmark_engines.py.
//
// Loads an EwE model with EwE's own core (EwECore.dll), times the Ecopath
// solve and Ecosim runs of several lengths, and appends rows
//     model,what,years,repeat,seconds
// to a CSV that `benchmark_engines.py report --ewe-csv` reads:
//   what = ecopath        repeat 0 is the first call, 1.. the timed repeats
//   what = ecosim_warmup  untimed-in-the-fit runs first (.NET compiles in tiers)
//   what = ecosim         repeats 1.., one row per run length per repeat
// With --biomass-out it also writes each group's biomass at the end of the
// shortest run, to check that EwE 6 and PyPath ran the same case.
//
// Usage:
//   EweTimer --ewe-dir "C:\Program Files\Ecopath with Ecosim 40 years 6.7.0.18865_64-bit" ^
//            --model "C:\models\LT2022.eweaccdb" --name LT2022 --scenario 1 ^
//            --years 10 20 40 --repeats 5 --warmup 3 --out ewe6.csv
//
// Ecosim in EwE 6 always integrates with RK4, one step a month.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;

internal sealed class Options
{
    public string EweDir;
    public string Model;
    public string Name;
    public int Scenario = 1;
    public List<int> Years = new List<int> { 10, 20, 40 };
    public int Repeats = 5;
    public int Warmup = 3;
    public string Out = "ewe6.csv";
    public string BiomassOut;
    public int Ecospace = 0;            // Ecospace scenario ID; 0 = time Ecosim
    public List<int> EcospaceYears = new List<int> { 2, 5, 10 };
    public int Threads = 0;             // 0 = EwE's own thread settings

    public static Options Parse(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException("missing value for " + a);
            switch (a)
            {
                case "--ewe-dir": o.EweDir = Next(); break;
                case "--model": o.Model = Next(); break;
                case "--name": o.Name = Next(); break;
                case "--scenario": o.Scenario = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--repeats": o.Repeats = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--warmup": o.Warmup = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--out": o.Out = Next(); break;
                case "--biomass-out": o.BiomassOut = Next(); break;
                case "--ecospace": o.Ecospace = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--threads": o.Threads = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                case "--ecospace-years":
                    o.EcospaceYears = new List<int>();
                    while (i + 1 < args.Length && !args[i + 1].StartsWith("--"))
                        o.EcospaceYears.Add(int.Parse(args[++i], CultureInfo.InvariantCulture));
                    break;
                case "--years":
                    o.Years = new List<int>();
                    while (i + 1 < args.Length && !args[i + 1].StartsWith("--"))
                        o.Years.Add(int.Parse(args[++i], CultureInfo.InvariantCulture));
                    break;
                case "-h":
                case "--help":
                    throw new ArgumentException("help");
                default:
                    throw new ArgumentException("unknown option " + a);
            }
        }
        if (string.IsNullOrEmpty(o.EweDir) || string.IsNullOrEmpty(o.Model))
            throw new ArgumentException("--ewe-dir and --model are required");
        if (!File.Exists(Path.Combine(o.EweDir, "EwECore.dll")))
            throw new ArgumentException("EwECore.dll not found in " + o.EweDir);
        if (!File.Exists(o.Model))
            throw new ArgumentException("model not found: " + o.Model);
        if (o.Years.Count == 0 || o.Repeats < 1)
            throw new ArgumentException("need at least one run length and one repeat");
        o.Years.Sort();
        o.EcospaceYears.Sort();
        if (string.IsNullOrEmpty(o.Name))
            o.Name = Path.GetFileNameWithoutExtension(o.Model);
        return o;
    }
}

internal static class Program
{
    private static int Main(string[] args)
    {
        Options o;
        try
        {
            o = Options.Parse(args);
        }
        catch (ArgumentException e)
        {
            if (e.Message != "help") Console.Error.WriteLine("Error: " + e.Message);
            Console.Error.WriteLine(
                "Usage: EweTimer --ewe-dir <EwE 6 folder> --model <file> [--name N] [--scenario <ScenarioID>]\n" +
                "                [--years 10 20 40] [--repeats 5] [--warmup 3] [--out ewe6.csv]\n" +
                "                [--biomass-out end_biomass.csv]\n" +
                "                [--ecospace <EcospaceScenarioID> [--ecospace-years 2 5 10] [--threads N]]");
            return 2;
        }

        // Load EwE's assemblies from its own folder (they are not copied here).
        string eweDir = Path.GetFullPath(o.EweDir);
        AppDomain.CurrentDomain.AssemblyResolve += (sender, e) =>
        {
            string name = new AssemblyName(e.Name).Name;
            foreach (string ext in new[] { ".dll", ".exe" })
            {
                string p = Path.Combine(eweDir, name + ext);
                if (File.Exists(p)) return Assembly.LoadFrom(p);
            }
            return null;
        };

        try
        {
            return Run(o);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("Error: " + e);
            return 1;
        }
    }

    // Kept out of Main so that EwE's types resolve after AssemblyResolve is set.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int Run(Options o)
    {
        var rows = new List<string>();
        var core = new EwECore.cCore();
        // EwE reports why a step failed through its message publisher. The
        // handler's enum types are in different namespaces in different EwE
        // builds, so they are taken from its constructor by reflection.
        var handlers = new List<EwECore.cMessageHandler>();
        try
        {
            ConstructorInfo ctor = null;
            foreach (var c in typeof(EwECore.cMessageHandler).GetConstructors())
                if (c.GetParameters().Length == 4) ctor = c;
            if (ctor != null)
            {
                var ps = ctor.GetParameters();
                var method = typeof(Program).GetMethod(nameof(OnCoreMessage), BindingFlags.NonPublic | BindingFlags.Static);
                var callback = Delegate.CreateDelegate(ps[0].ParameterType, method);
                object anyType = Enum.Parse(ps[2].ParameterType, "Any");
                foreach (string source in new[] { "DataSource", "Core", "Ecopath", "Ecosim", "Ecospace" })
                {
                    object src = Enum.Parse(ps[1].ParameterType, source);
                    var h = (EwECore.cMessageHandler)ctor.Invoke(new object[] { callback, src, anyType, null });
                    handlers.Add(h);
                    core.Messages.AddMessageHandler(h);
                }
            }
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("(EwE's messages will not be shown: " + e.Message + ")");
        }
        // The plug-in manager is in EwECore in newer EwE builds and in
        // EwEPlugin.dll in 6.7.0; create it by reflection to work with both.
        var pmProp = core.GetType().GetProperty("PluginManager");
        if (pmProp != null && pmProp.CanWrite && pmProp.GetValue(core, null) == null)
            pmProp.SetValue(core, Activator.CreateInstance(pmProp.PropertyType), null);
        try
        {
            if (!core.LoadModel(Path.GetFullPath(o.Model)))
                throw new InvalidOperationException("EwE could not load the model " + o.Model +
                    (Messages.Count > 0 ? ". EwE said:\n  " + string.Join("\n  ", Messages) : " (EwE gave no message)"));

            // Ecopath: the solve alone. Repeat 0 is the first call.
            for (int r = 0; r <= o.Repeats; r++)
            {
                Collect();
                var sw = Stopwatch.StartNew();
                if (!core.RunEcopath())
                    throw new InvalidOperationException("Ecopath failed");
                sw.Stop();
                rows.Add(Row(o.Name, "ecopath", 0, r, sw.Elapsed.TotalSeconds));
            }
            Console.WriteLine($"{o.Name}: Ecopath solve {rows[rows.Count - 1].Split(',')[4]} s (last repeat)");

            // --scenario is the scenario's ID in the database (EcosimScenario.ScenarioID),
            // as PyPath numbers scenarios; EwE's core counts them 1..n.
            int index = 0;
            var listed = new List<string>();
            for (int i = 1; i <= core.nEcosimScenarios; i++)
            {
                var sc = core.get_EcosimScenarios(i);
                listed.Add($"ID {sc.DBID} \"{sc.Name}\"");
                if (sc.DBID == o.Scenario) index = i;
            }
            if (index == 0)
                throw new InvalidOperationException(
                    $"the model has no Ecosim scenario with ID {o.Scenario}; it has: {string.Join(", ", listed)}");
            Console.WriteLine($"{o.Name}: Ecosim scenario ID {o.Scenario} \"{core.get_EcosimScenarios(index).Name}\" (number {index} of {core.nEcosimScenarios})");
            if (!core.LoadEcosimScenario(index))
                throw new InvalidOperationException("could not load Ecosim scenario " + o.Scenario +
                    (Messages.Count > 0 ? ". EwE said:\n  " + string.Join("\n  ", Messages) : ""));

            int originalYears = core.EcosimModelParameters.NumberYears;
            if (o.Ecospace > 0)
            {
                try { TimeEcospaceRuns(core, o, rows); }
                finally { try { SetYears(core, originalYears); } catch { } }
            }
            else
            try
            {
                // Warm-up runs at the shortest length: .NET compiles in tiers,
                // so a method is fast only after it has been called many times.
                SetYears(core, o.Years[0]);
                for (int w = 1; w <= o.Warmup; w++)
                {
                    double s = TimeEcosim(core);
                    rows.Add(Row(o.Name, "ecosim_warmup", o.Years[0], w, s));
                }

                // Timed runs: every length in each repeat, so slow periods of the
                // machine fall on all lengths alike.
                bool biomassWritten = false;
                for (int r = 1; r <= o.Repeats; r++)
                {
                    foreach (int n in o.Years)
                    {
                        SetYears(core, n);
                        double s = TimeEcosim(core);
                        rows.Add(Row(o.Name, "ecosim", n, r, s));
                        Console.WriteLine($"{o.Name}: Ecosim {n} years, repeat {r}: {s:F4} s");
                        if (!biomassWritten && n == o.Years[0] && !string.IsNullOrEmpty(o.BiomassOut))
                        {
                            WriteEndBiomass(core, o);
                            biomassWritten = true;
                        }
                    }
                }
            }
            finally
            {
                // Leave the model's run length as it was (the file is not saved).
                try { SetYears(core, originalYears); } catch { }
            }
        }
        finally
        {
            try { core.CloseModel(); } catch { }
        }

        bool header = !File.Exists(o.Out) || new FileInfo(o.Out).Length == 0;
        using (var w = new StreamWriter(o.Out, append: true, encoding: new UTF8Encoding(false)))
        {
            if (header) w.WriteLine("model,what,years,repeat,seconds");
            foreach (string row in rows) w.WriteLine(row);
        }
        Console.WriteLine($"Wrote {rows.Count} rows to {Path.GetFullPath(o.Out)}");
        return 0;
    }

    // Ecospace: runs of each length (years), EwE's result writers off so that
    // the time is the model's, not the writing of map and area files.
    private static void TimeEcospaceRuns(EwECore.cCore core, Options o, List<string> rows)
    {
        int maxYears = o.EcospaceYears[o.EcospaceYears.Count - 1];
        // Ecospace runs no longer than the Ecosim scenario
        if (core.EcosimModelParameters.NumberYears < maxYears) SetYears(core, maxYears);
        int index = 0;
        var listed = new List<string>();
        for (int i = 1; i <= core.nEcospaceScenarios; i++)
        {
            var sc = core.get_EcospaceScenarios(i);
            listed.Add($"ID {sc.DBID} \"{sc.Name}\"");
            if (sc.DBID == o.Ecospace) index = i;
        }
        if (index == 0)
            throw new InvalidOperationException(
                $"the model has no Ecospace scenario with ID {o.Ecospace}; it has: " +
                (listed.Count > 0 ? string.Join(", ", listed) : "none"));
        if (!core.LoadEcospaceScenario(index))
            throw new InvalidOperationException("could not load Ecospace scenario " + o.Ecospace +
                (Messages.Count > 0 ? ". EwE said:\n  " + string.Join("\n  ", Messages) : ""));
        var mp = core.EcospaceModelParameters;
        if (o.Threads > 0)
        {
            mp.nGridSolverThreads = o.Threads;
            mp.nSpaceThreads = o.Threads;
            mp.nEffortDistThreads = o.Threads;
        }
        int disabled = 0;
        for (int w = 1; w <= mp.nResultWriters; w++)
        {
            object writer = mp.ResultWriter(w);
            var en = writer?.GetType().GetProperty("Enabled");
            if (en != null && en.CanWrite && (bool)en.GetValue(writer, null))
            {
                en.SetValue(writer, false, null);
                disabled++;
            }
        }
        // Autosave of Ecospace inputs and results off. The enum is in different
        // namespaces in different EwE builds: take it from the property.
        try
        {
            var autosave = core.GetType().GetProperty("Autosave");
            var kind = autosave?.GetIndexParameters()[0].ParameterType;
            if (autosave != null && autosave.CanWrite)
                foreach (string t in new[] { "Ecospace", "EcospaceResults" })
                    autosave.SetValue(core, false, new[] { Enum.Parse(kind, t) });
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("(could not turn Ecospace autosave off: " + e.Message + ")");
        }
        Console.WriteLine($"{o.Name}: Ecospace scenario ID {o.Ecospace} \"{core.get_EcospaceScenarios(index).Name}\": " +
            $"{core.EcospaceBasemap.InRow} x {core.EcospaceBasemap.InCol} map, {mp.NumberOfTimeStepsPerYear} time steps a year, " +
            $"threads grid {mp.nGridSolverThreads} space {mp.nSpaceThreads}, " +
            $"result writers turned off {disabled}");

        SetEcospaceYears(core, o.EcospaceYears[0]);
        for (int w = 1; w <= o.Warmup; w++)
            rows.Add(Row(o.Name, "ecospace_warmup", o.EcospaceYears[0], w, TimeEcospace(core)));
        for (int r = 1; r <= o.Repeats; r++)
        {
            foreach (int n in o.EcospaceYears)
            {
                SetEcospaceYears(core, n);
                double s = TimeEcospace(core);
                rows.Add(Row(o.Name, "ecospace", n, r, s));
                Console.WriteLine($"{o.Name}: Ecospace {n} years ({core.nEcospaceTimeSteps} steps), repeat {r}: {s:F3} s");
            }
        }
    }

    private static void SetEcospaceYears(EwECore.cCore core, int years)
    {
        var mp = core.EcospaceModelParameters;
        if (Math.Abs(mp.TotalTime - years) > 1e-6) mp.TotalTime = years;
        if (Math.Abs(core.EcospaceModelParameters.TotalTime - years) > 1e-6)
            throw new InvalidOperationException("EwE did not accept an Ecospace run of " + years + " years");
    }

    private static double TimeEcospace(EwECore.cCore core)
    {
        Collect();
        var sw = Stopwatch.StartNew();
        EwECore.cCore.EcoSpaceInterfaceDelegate none = null;
        if (!core.RunEcospace(ref none, false))
            throw new InvalidOperationException("Ecospace failed" +
                (Messages.Count > 0 ? ". EwE said:\n  " + string.Join("\n  ", Messages) : ""));
        sw.Stop();
        return sw.Elapsed.TotalSeconds;
    }

    private static readonly List<string> Messages = new List<string>();

    private static void OnCoreMessage(ref EwECore.cMessage message)
    {
        if (message != null && !string.IsNullOrWhiteSpace(message.Message))
            Messages.Add($"[{message.Source}] {message.Message}");
    }

    private static void SetYears(EwECore.cCore core, int years)
    {
        if (core.EcosimModelParameters.NumberYears != years)
            core.EcosimModelParameters.NumberYears = years;
        if (core.EcosimModelParameters.NumberYears != years)
            throw new InvalidOperationException("EwE did not accept an Ecosim run length of " + years + " years");
    }

    private static double TimeEcosim(EwECore.cCore core)
    {
        Collect();
        var sw = Stopwatch.StartNew();
        if (!core.RunEcosim())
            throw new InvalidOperationException("Ecosim failed");
        sw.Stop();
        return sw.Elapsed.TotalSeconds;
    }

    private static void WriteEndBiomass(EwECore.cCore core, Options o)
    {
        int last = core.nEcosimTimeSteps;
        bool header = !File.Exists(o.BiomassOut) || new FileInfo(o.BiomassOut).Length == 0;
        using (var w = new StreamWriter(o.BiomassOut, append: true, encoding: new UTF8Encoding(false)))
        {
            if (header) w.WriteLine("model,years,group,biomass");
            for (int i = 1; i <= core.nGroups; i++)
            {
                // VB.NET properties with an argument are get_X(...) methods in C#
                string name = core.get_EcopathGroupInputs(i).Name.Replace(",", ";");
                float b = core.get_EcosimGroupOutputs(i).get_Biomass(last);
                w.WriteLine(string.Join(",", o.Name, core.EcosimModelParameters.NumberYears.ToString(CultureInfo.InvariantCulture),
                    name, b.ToString("R", CultureInfo.InvariantCulture)));
            }
        }
    }

    private static void Collect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    private static string Row(string model, string what, int years, int repeat, double seconds) =>
        string.Join(",", model.Replace(",", ";"), what, years.ToString(CultureInfo.InvariantCulture),
            repeat.ToString(CultureInfo.InvariantCulture), seconds.ToString("R", CultureInfo.InvariantCulture));
}
