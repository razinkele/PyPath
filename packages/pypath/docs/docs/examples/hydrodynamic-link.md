# Ecospace on a Hydrodynamic Model's Boxes

A hydrodynamic box model (here in SHYFEM's formats) gives three things an
Ecospace run can use: the **mesh** it computes on, the **fields** of each
box (temperature, salinity), and the **water flows** between the boxes.
This page takes them in that order, on small sample files it writes
itself, so every block runs as it stands.

!!! note "What to expect"
    - The mesh becomes the map: one patch per box, or one per element.
    - Fields become a habitat capacity map, which says *where* a group
      lives. A capacity for every month can be added for chosen groups; it
      also says *when*.
    - Flows move the groups you choose. With the default settings water
      only mixes them between boxes, and on a map where the boxes are
      alike that changes nothing. With throughflow, rivers flush them
      towards the sea: a loss the Ecopath balance does not hold.

## Sample files

Use your own files in their place: a `.grd` mesh whose element type is the
box number, and the box model's `boxes_2d.txt`.

??? example "A mesh of four boxes and two years of box output"

    ```python
    import tempfile
    from pathlib import Path

    import numpy as np

    work = Path(tempfile.mkdtemp())

    # A channel of 8 x 2 squares, two triangles each. Boxes 1 to 4 run from
    # the river end to the sea; the element type is the box number.
    box_of_column = [1, 1, 1, 2, 2, 3, 4, 4]
    depth_of_box = {1: 2.0, 2: 3.0, 3: 6.0, 4: 15.0}
    lines, node = [], {}
    for iy in range(3):
        for ix in range(9):
            node[(ix, iy)] = len(node) + 1
            depth = depth_of_box[box_of_column[min(ix, 7)]]
            lines.append(
                f"1 {node[(ix, iy)]} 0 {21.0 + 0.03 * ix:.4f} "
                f"{55.30 + 0.03 * iy:.4f} {depth}"
            )
    number = 0
    for iy in range(2):
        for ix in range(8):
            a, b = node[(ix, iy)], node[(ix + 1, iy)]
            c, d = node[(ix + 1, iy + 1)], node[(ix, iy + 1)]
            for tri in ((a, b, c), (a, c, d)):
                number += 1
                lines.append(
                    f"2 {number} {box_of_column[ix]} 3 {tri[0]} {tri[1]} {tri[2]}"
                )
    (work / "lagoon.grd").write_text("\n".join(lines) + "\n")


    def write_box_output(path, volume):
        """Two years of daily records: box means, then the section flows.

        Salinity rises from the river end to the sea; in box 3, the strait,
        it also rises in summer (1 in winter, 5 in summer)."""
        days = np.datetime64("2010-01-01") + np.arange(1, 731)
        out = ["#      idbox file_type     nvers", "#     473226         2         7"]
        for n, day in enumerate(days):
            season = 1.0 + 0.5 * np.sin(2 * np.pi * (n - 80) / 365.0)
            temp = 10.0 - 9.0 * np.cos(2 * np.pi * (n - 30) / 365.0)
            salt = (0.1, 0.6, 3.0 - 2.0 * np.cos(2 * np.pi * (n - 30) / 365.0), 7.0)
            river = 60.0 * season
            out += [
                "# time record",
                f"   {86400.0 * (n + 1):.2f}      {day}::00:00:00",
                "#      boxes       nvars",
                "          4           3",
                "# box      temp      salt        volume",
            ]
            out += [
                f"   {b}  {temp:.4f}  {salt[b - 1]:.4f}  {volume[b - 1]:.6E}"
                for b in (1, 2, 3, 4)
            ]
            out += [
                "#   sections",
                "          5",
                "# section  from    to         total      positive      negative",
            ]
            # between boxes: the river's water one way, an exchange both ways
            for s, (frm, to, back) in enumerate(
                ((1, 2, 150.0), (2, 3, 250.0), (3, 4, 400.0)), 1
            ):
                out.append(
                    f"   {s}   {frm}   {to}  {river:.4E}  "
                    f"{river + back * season:.4E}  {back * season:.4E}"
                )
            out.append(f"   4  -2   1  {river:.4E}  {river:.4E}  0.0000E+00")  # river
            out.append(f"   5   4  -1  {river:.4E}  {river + 2000.0:.4E}  2.0000E+03")
        path.write_text("\n".join(out) + "\n")
    ```

## 1. The mesh as the map

`EcospaceGrid.from_shyfem_grd` reads the mesh. `patches="type"` gives one
patch per box, `patches="element"` one per triangle. Areas, borders and
depths come from the mesh; a file in degrees is measured on the sphere.

```python
from pypath.spatial import EcospaceGrid

boxes = EcospaceGrid.from_shyfem_grd(work / "lagoon.grd", patches="type")
elements = EcospaceGrid.from_shyfem_grd(work / "lagoon.grd", patches="element")

print(boxes.n_patches, "boxes of", boxes.patch_areas.round(1), "km2")
print("depth (m):", boxes.cell_metadata["depth"].round(1).tolist())
print(elements.n_patches, "elements")
```

A `.grd` file names no coordinate system. For a mesh in metres pass
`units="m"` and, to place it on a map, `crs="EPSG:..."`.

The sample box output needs the boxes' volumes, so it is written now:

```python
volume = boxes.patch_areas * 1e6 * boxes.cell_metadata["depth"].to_numpy()
write_box_output(work / "boxes_2d.txt", volume)
```

## 2. Box fields as layers and a capacity map

`read_box_series` reads the box model's output; `monthly=True` keeps only
the monthly means, which is what a run of many years needs.
`shyfem_box_layers` puts them on a grid: each patch takes its box's value,
so the same series serves the box grid and the element grid.

One start date serves everything from here on: it is the date of the run's
first month.

```python
from pypath.io.shyfem import read_box_series
from pypath.spatial import shyfem_box_layers

START = "2010-01-01"
series = read_box_series(work / "boxes_2d.txt", monthly=True)
layers = shyfem_box_layers(series, boxes, start=START)

print(layers.layer_names)
print("salinity by box, January:", layers.get_layer_at_time("salt", 0.0).round(1))
print("salinity by box, July:   ", layers.get_layer_at_time("salt", 6 / 12).round(1))
print(
    "temperature, January and July:",
    layers.get_layer_at_time("temp", 0.0)[0].round(1),
    layers.get_layer_at_time("temp", 6 / 12)[0].round(1),
)
```

A capacity map is made from layers and response functions. Here two marine
groups need salt: unsuitable below 0.5, fully suitable from 6.

```python
from pypath.examples import example_ecosim_scenario
from pypath.spatial import capacity_from_layers, create_linear_response

YEARS = range(1, 3)
scenario = example_ecosim_scenario("Central_Baltic", years=YEARS)
names = [str(n).strip() for n in scenario.params.spname]
n_rows = scenario.params.NUM_GROUPS + 1  # the arrays' rows; 0 is the first group

marine = create_linear_response(0.5, 6.0)
capacity = capacity_from_layers(
    layers,
    {"Ad. Cod": {"salt": marine}, "Ad. Sprat": {"salt": marine}},
    group_names=names[1:] + ["-"],
)
print("cod capacity by box:", capacity[names.index("Ad. Cod") - 1].round(2))
```

Ecospace uses the *pattern* of a group's capacity over the map, not its
level. A field that is nearly the same in every box, as temperature often
is, therefore does little as a static map; salinity does a lot.

This map is one for the whole run: the mean over the months. Section 6
gives a group a capacity for every month.

## 3. Which groups the water can carry

Under the EwE rule, which EwE models get by default, plankton and often
detritus are *fast* groups: every step sets their biomass to its local
equilibrium, so nothing can move them. `integrate_groups` returns a copy of
the scenario in which the groups you name are integrated instead.

```python
import numpy as np

from pypath.core.ecosim import integrate_groups, substep_check

fast = [names[i] for i in np.flatnonzero(np.asarray(scenario.params.NoIntegrate)[1:]) + 1]
print("fast groups:", fast)

ready = integrate_groups(scenario, fast, substeps=4)
check = substep_check(ready, years=YEARS)
print(check)
```

An integrated group needs sub-steps for its turnover, and the number cannot
be read off its P/B: `substep_check` runs the scenario at its sub-steps and
at twice as many and reports the largest difference. Raise `substeps` until
it converges. A pool that turns over thousands of times a year (detritus,
in some models) is better left fast: leave it out of the list.

## 4. Water flows

`WaterTransport` holds the monthly flows between the boxes and the groups
they carry. It is applied once a month, after the local dynamics.

```python
from pypath.spatial import WaterTransport

carried = [names.index(g) - 1 for g in fast]  # rows: 0 is the first group

exchange = WaterTransport.from_shyfem_box_series(series, groups=carried, start=START)
throughflow = WaterTransport.from_shyfem_box_series(
    series, groups=carried, start=START, throughflow=True
)
print("diluting boundaries:", throughflow.open_boundaries)
```

There are two ways to use the flows:

| | What moves | Effect |
|---|---|---|
| Exchange only (default) | the water that goes both ways between two boxes | mixes the carried groups; their total is kept |
| `throughflow=True` | all the water; a river brings none of the group | flushes the carried groups towards the sea |

Both directions of every section are kept. With net flows alone a tracer
does not follow an estuary's exchange.

## 5. The runs

```python
from pypath.spatial import EcospaceParams, rsim_run_spatial


def run(transport, monthly=None):
    ecospace = EcospaceParams(
        grid=boxes,
        habitat_preference=np.ones((n_rows, boxes.n_patches)),
        habitat_capacity=capacity,
        dispersal_rate=np.zeros(n_rows),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
        water_transport=transport,
        monthly_capacity=monthly,
    )
    return rsim_run_spatial(ready, ecospace=ecospace, years=YEARS)


still = run(None)
mixed = run(exchange)
flushed = run(throughflow)

phyto = names.index("spring phytoplankton")
for label, out in (("no transport", still), ("exchange", mixed), ("throughflow", flushed)):
    density = out.out_Biomass_spatial[-1, phyto] / boxes.patch_areas
    print(
        f"{label:13s} total {out.out_Biomass[-1, phyto]:.3f}   "
        f"density by box / mean {np.round(density / density.mean(), 2)}"
    )
```

What the three lines show:

- **No transport.** The boxes differ a little, because cod and sprat live
  only where the water is salt and their prey follows.
- **Exchange.** The water evens the plankton out between the boxes; its
  total barely moves. On a map where every box is alike, exchange changes
  nothing at all.
- **Throughflow.** The river end is diluted and the plankton's total falls.
  This is a flushing loss the model's Ecopath balance does not contain, so
  the food web above the plankton responds to it. Slow pools are flushed
  hardest.

## 6. A capacity for every month

The map of section 2 holds for the whole run. `MonthlyCapacity` gives the
groups you name a capacity for each month instead, made from the same
layers and the same kind of response. Here a marine copepod needs salt, and
the strait is salt only in summer:

```python
from pypath.spatial import MonthlyCapacity

copepod = {"Pseudocalanus sp.": {"salt": marine}}
group_names = names[1:] + ["-"]

# the base map: the mean over the months, joined to the map of section 2
base = capacity_from_layers(layers, copepod, group_names=group_names)
row = names.index("Pseudocalanus sp.") - 1
capacity[row] = base[row]

by_month = MonthlyCapacity.from_layers(layers, copepod, group_names=group_names)
print("the copepod in the strait, by month:", by_month.values[:12, 0, 2].round(2))

seasonal = run(exchange, by_month)  # the whole chain in one run
steady = run(exchange)

strait = 2
for label, out in (("the mean map", steady), ("a map per month", seasonal)):
    biomass = out.out_Biomass_spatial[13:, row + 1, strait]
    print(
        f"{label:16s} the copepod in the strait, year 2: "
        f"{biomass.min():.4f} to {biomass.max():.4f}"
    )
```

What the month's capacity does, and what the base map keeps:

- The run is calibrated once, on the base map (`habitat_capacity`). Each
  month the group's capacity in a patch, over the peak of its base map, is
  that patch's feeding efficiency. So here the *level* matters, where a
  static map's does not.
- The base map still sets where the group starts and, unless you ask, how
  it moves between patches: diffusion keeps pulling the group back to the
  base pattern. With `movement=True` (`MonthlyCapacity.from_layers(...,
  movement=True)`) movement reads the month's map, as it does in EwE: the
  group leaves a patch that has turned poor, and water exchange with
  `rest="capacity"` spreads it by the month's capacity. It matters most at
  high dispersal rates. In a three-zone test with a dispersal rate of 5 per
  year, a year after the map was mirrored the density ratio between the
  outer zones was 1.42 with movement on the base map (which asks for 1.75)
  and 0.66 with movement on the month's map (which asks for 0.57).
- Multi-stanza groups cannot take a monthly capacity; give them a static
  map, as cod and sprat have here.

!!! warning "Mind the scale"
    The model's Ecopath balance stands for the conditions at the peak of the
    base map. A response far below 1 for months, such as a temperature
    optimum reached only in summer, starves the group for the rest of the
    year. Shape a response so that it is near 1 under the conditions the
    model describes.

## Runs longer than the flows, and other sources

A run may cover months the flows do not, before their first month or after
their last. `beyond` says what to do there: `"refuse"` (the default),
`"repeat_year"`, `"hold"`, or `"mean_year"`, each calendar month's mean over
all the years of the flows.

```python
longer = WaterTransport.from_shyfem_box_series(
    series, groups=carried, start="2005-01-01", beyond="mean_year"
)
print("the flows begin in month", longer.first_month, "of that run")
```

Flows need not come from SHYFEM's box output.
`WaterTransport.from_link_flows` takes any series of one-way flows between
zones, daily or finer, and `pypath.io.aquabc.read_aquabc_flows` reads them
from an AQUABC input folder:

```python
times = np.datetime64("2010-01-01", "s") + np.arange(365) * np.timedelta64(1, "D")
links = WaterTransport.from_link_flows(
    times,
    np.tile([210.0, 150.0, 60.0], (365, 1)),  # m3/s through each link
    upstream=[1, 2, -2],  # a zone below 1 is the outside
    downstream=[2, 1, 1],
    volume=[8e7, 9e7],
    groups=carried,
)
print(links.zones, links.boundary_id)
```

## On the mesh's elements

The same series and flows serve the element grid: each element takes its
box's field, and the water moves each box's total while the elements keep
their shares. Diffusion needs one adjustment, because its rate is an
exchange rate between neighbouring patches and so depends on their size:

```python
from pypath.spatial import dispersal_rate_for_grid

print("a rate of 1 on the boxes is", dispersal_rate_for_grid(1.0, boxes, elements))
```

See the [Spatial](../api/spatial.md) and [I/O](../api/io.md) references for
every option.
