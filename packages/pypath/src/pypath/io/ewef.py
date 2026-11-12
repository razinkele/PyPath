"""Import EwE-F models (Ecopath with Ecosim in Fortran) from their input files.

EwE-F (Akoğlu et al. 2015, doi:10.5194/gmd-8-2687-2015) reads a model from a
Fortran namelist, ``*_filenames.nml``, that names its CSV input files:

- Ecopath (``ecopath-f``): ``InputData_fname`` (groups and basic inputs),
  ``DietComp_fname``, ``DetFate_fname`` and ``GrowthParam_fname`` (multi-stanza
  growth). :func:`read_ewef` turns them into :class:`~pypath.core.params.RpathParams`.
- Ecosim (``ecosim-f``): ``GroupInfo_fname`` (per-group Ecosim settings) and
  ``Vulnerability_fname``. :func:`read_ewef_ecosim` turns them into
  ``scenario_overrides`` for :func:`pypath.core.ecosim.rsim_scenario`, and
  :func:`ecosim_scenario_from_ewef` builds the scenario.

EwE-F writes its Ecopath result to an HDF5 file; :func:`read_ewef_results`
reads it (needs ``h5py``) for comparisons.

The paths in a namelist are relative to EwE-F's run directory
(``testcases/<Model>/...``); a file that is not found there is looked up by
name next to the namelist. ``-999`` is EwE-F's missing-value marker.

Where the two programs differ, the importer says so in a warning rather than
guessing: EwE-F settings PyPath does not apply (handling time, predator effect
on feeding time, density-dependent catchability, nutrient limitation, fishing
forcing of type 4 or 5) and diet links without a vulnerability (EwE-F skips
them in Ecosim; PyPath, like EwE 6, keeps them at the default vulnerability).
"""

from __future__ import annotations

import csv
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pypath.core.params import RpathParams, create_rpath_params

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
EWEF_MISSING = -999.0

# EwE-F organism type -> PyPath type (consumer, producer, detritus)
_TYPE_MAP = {2: 0, 1: 1, 0: 2}

# Ecopath input columns (EwE-F header label, lower-case) -> field
_ECOPATH_COLUMNS = {
    "b": "Biomass",
    "p/b": "PB",
    "q/b": "QB",
    "ee": "EE",
    "p/q": "ProdCons",
    "unass. q": "Unassim",
    "detritus import": "DetInput",
    "landings": "landings",
    "discards": "discards",
    "biomass accumulation": "BioAcc",
    "organism type": "org_type",
    "split group?": "split",
    "stanza no": "stanza_no",
    "age start": "age_start",
    "leading stanza?": "leading",
}

# Ecosim group settings (EwE-F header label starts with) -> field
_ECOSIM_COLUMNS = [
    ("max rel. p/b", "pb_max"),
    ("feeding time adjust", "ftime_adjust"),
    ("max rel. feeding time", "ftime_max"),
    ("feeding time", "ftime"),
    ("fraction of other mortality", "mo_pred"),
    ("predator effect on feeding time", "pred_effect"),
    ("density-dep. catchability", "qmax_qo"),
    ("qbmax/qbo", "qbmax_qbo"),
    ("switching power", "switch_power"),
    ("advected", "advected"),
]


# --- files ----------------------------------------------------------------------


def _find_namelist(path: PathLike) -> Path:
    """The EwE-F namelist given directly or as the only one in a directory."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        found = sorted(p.glob("*filenames.nml")) or sorted(p.glob("*.nml"))
        if len(found) == 1:
            return found[0]
        if found:
            raise FileNotFoundError(
                f"More than one EwE-F namelist in {p}: {[f.name for f in found]}"
            )
    raise FileNotFoundError(f"No EwE-F namelist (*_filenames.nml) at {p}")


def read_ewef_namelist(path: PathLike) -> Dict[str, Any]:
    """Settings of an EwE-F ``&filenames`` namelist as a dict.

    Strings are returned without quotes, ``.true.``/``.false.`` as bool and
    numbers as int or float.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [line.split("!", 1)[0] for line in text.splitlines()]
    body = "\n".join(lines)
    match = re.search(r"&\w+(.*?)^\s*/\s*$", body, flags=re.S | re.M)
    if match:
        body = match.group(1)
    values: Dict[str, Any] = {}
    pattern = r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s,]+)"
    for key, raw in re.findall(pattern, body):
        if raw[:1] in "\"'":
            value: Any = raw[1:-1]
        elif raw.lower() in (".true.", ".t."):
            value = True
        elif raw.lower() in (".false.", ".f."):
            value = False
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw.replace("d", "e").replace("D", "e"))
                except ValueError:
                    value = raw
        values[key] = value
    return values


def _resolve(nml_path: Path, name: Any) -> Optional[Path]:
    """A file a namelist names: as written, else by name next to the namelist."""
    if not name or not isinstance(name, str):
        return None
    for candidate in (
        nml_path.parent / name,
        nml_path.parent / Path(name).name,
        Path(name),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"EwE-F file {name!r} named in {nml_path.name} not found next to it"
    )


def _read_table(path: Path, first_label: str) -> Tuple[List[str], List[List[str]]]:
    """Header and rows of an EwE-F CSV: the rows after the line whose first
    cell is ``first_label`` (the lines before it give counts)."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = [r for r in csv.reader(fh)]
    for i, row in enumerate(rows):
        if row and row[0].strip().lower() == first_label.lower():
            header = [c.strip() for c in row]
            data = [r for r in rows[i + 1 :] if r and r[0].strip()]
            return header, data
    raise ValueError(f"{path.name}: no header row starting with {first_label!r}")


def _number(cell: Any) -> float:
    """A CSV cell as float; EwE-F's -999 marker and blanks become NaN."""
    try:
        value = float(str(cell).strip())
    except (TypeError, ValueError):
        return np.nan
    return np.nan if value <= EWEF_MISSING + 0.5 else value


def _stanza_group_name(stages: List[str], number: int) -> str:
    """Name shared by a split group's stages ('Anchovy 0-0+', 'Anchovy 1-1+'
    -> 'Anchovy'), else 'Stanza <n>'."""
    prefix = stages[0]
    for name in stages[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix.rstrip(" -_(")
    return prefix if prefix and prefix not in stages else f"Stanza {number}"


# --- Ecopath --------------------------------------------------------------------


def read_ewef(
    path: PathLike, compute_stanzas: bool = True, fleet_name: str = "Fishery"
) -> RpathParams:
    """Read an EwE-F Ecopath model into RpathParams.

    Parameters
    ----------
    path : str or Path
        The ``ecopath-f`` namelist (``*_filenames.nml``) or its directory.
    compute_stanzas : bool
        Fill in the biomass and Q/B of non-leading multi-stanza groups from
        the leading stanza (:func:`pypath.core.stanzas.rpath_stanzas`), as
        EwE-F's Ecopath does; EwE-F's inputs leave them at -999. Default True.
    fleet_name : str
        Name of the single fleet that carries the ``Landings`` and
        ``Discards`` columns (EwE-F has no fleets). Default "Fishery".

    Returns
    -------
    RpathParams
        Ready for :func:`pypath.core.ecopath.rpath`.

    Notes
    -----
    EwE-F organism types 2/1/0 become PyPath types 0/1/2 (consumer,
    producer, detritus); ``P/Q`` is ``ProdCons``, ``Unass. Q`` is
    ``Unassim``, ``Detritus Import`` is ``DetInput`` and ``Biomass
    Accumulation`` is ``BioAcc``. Stanza ages are in months, as in EwE-F; each
    stage's total mortality is its P/B. EwE-F adds a group's discards to
    every detritus pool; here the fleet's discards go to the first detritus
    group (the test cases have none). An ``Export`` share of detritus fate is
    not routed to any detritus group.
    """
    nml_path = _find_namelist(path)
    nml = read_ewef_namelist(nml_path)
    input_path = _resolve(nml_path, nml.get("InputData_fname"))
    diet_path = _resolve(nml_path, nml.get("DietComp_fname"))
    fate_path = _resolve(nml_path, nml.get("DetFate_fname"))
    growth_path = _resolve(nml_path, nml.get("GrowthParam_fname"))
    if input_path is None or diet_path is None:
        raise ValueError(
            f"{nml_path.name} names no InputData_fname/DietComp_fname: "
            "is it an ecopath-f namelist?"
        )

    header, rows = _read_table(input_path, "Groups")
    fields = {}
    for j, label in enumerate(header[1:], start=1):
        key = _ECOPATH_COLUMNS.get(label.lower())
        if key:
            fields[key] = j
    names = [r[0].strip() for r in rows]
    data = {
        key: [_number(r[j]) if j < len(r) else np.nan for r in rows]
        for key, j in fields.items()
    }
    org = [int(v) if np.isfinite(v) else 2 for v in data["org_type"]]
    types = [_TYPE_MAP.get(t, 0) for t in org]
    landings = np.nan_to_num(np.asarray(data.get("landings", [0] * len(names))))
    discards = np.nan_to_num(np.asarray(data.get("discards", [0] * len(names))))
    has_fleet = bool(np.any(landings > 0) or np.any(discards > 0))

    # multi-stanza groups
    split = [
        bool(np.isfinite(v) and v > 0) for v in data.get("split", [0] * len(names))
    ]
    stanza_no = data.get("stanza_no", [np.nan] * len(names))
    split_numbers = sorted({int(stanza_no[i]) for i, s in enumerate(split) if s})
    split_stages = {
        n: [names[i] for i, s in enumerate(split) if s and int(stanza_no[i]) == n]
        for n in split_numbers
    }
    split_names = {
        n: _stanza_group_name(stages, n) for n, stages in split_stages.items()
    }
    stgroups = [
        split_names[int(stanza_no[i])] if s else None for i, s in enumerate(split)
    ]

    groups = names + ([fleet_name] if has_fleet else [])
    all_types = types + ([3] if has_fleet else [])
    params = create_rpath_params(
        groups,
        all_types,
        stgroups=(stgroups + [None] * has_fleet) if split_numbers else None,
    )
    model = params.model
    n_bio = len(names)
    for col in ("Biomass", "PB", "QB", "EE", "ProdCons", "Unassim", "BioAcc"):
        if col in data:
            model.loc[: n_bio - 1, col] = data[col]
    if "DetInput" in data:
        det_input = [v if t == 2 else np.nan for v, t in zip(data["DetInput"], types)]
        model.loc[: n_bio - 1, "DetInput"] = det_input
    if has_fleet:
        model.loc[: n_bio - 1, fleet_name] = landings
        model.loc[: n_bio - 1, f"{fleet_name}.disc"] = discards

    _fill_diet(params, diet_path, names)
    det_groups = [g for g, t in zip(names, types) if t == 2]
    if fate_path is not None:
        _fill_detritus_fate(params, fate_path, names, det_groups)
    if has_fleet and det_groups:
        for det in det_groups:
            model.loc[n_bio, det] = 1.0 if det == det_groups[0] else 0.0

    if split_numbers:
        _fill_stanzas(params, data, names, split_stages, split_names, growth_path)
        if compute_stanzas:
            from pypath.core.stanzas import rpath_stanzas

            params = rpath_stanzas(params)
    params.compat = "ewe"  # an EwE model (roadmap decision D1)
    return params


def _fill_diet(params: RpathParams, diet_path: Path, names: List[str]) -> None:
    header, rows = _read_table(diet_path, "Prey/Pred")
    predators = header[1:]
    diet = params.diet
    prey_index = {str(g).strip(): i for i, g in enumerate(diet["Group"])}
    for col in diet.columns[1:]:
        diet[col] = 0.0
    unknown = [p for p in predators if p and p not in diet.columns]
    if unknown:
        raise ValueError(f"{diet_path.name}: predators not in the model: {unknown}")
    for row in rows:
        prey = row[0].strip()
        if prey not in prey_index:
            raise ValueError(f"{diet_path.name}: prey {prey!r} not in the model")
        for j, pred in enumerate(predators, start=1):
            if pred and j < len(row):
                value = _number(row[j])
                diet.loc[prey_index[prey], pred] = (
                    0.0 if not np.isfinite(value) else value
                )


def _fill_detritus_fate(
    params: RpathParams, fate_path: Path, names: List[str], det_groups: List[str]
) -> None:
    header, rows = _read_table(fate_path, "Groups")
    model = params.model
    row_of = {g: i for i, g in enumerate(names)}
    export_col = next((j for j, c in enumerate(header) if c.lower() == "export"), None)
    exported = []
    for row in rows:
        group = row[0].strip()
        if group not in row_of:
            raise ValueError(f"{fate_path.name}: group {group!r} not in the model")
        for j, col in enumerate(header[1:], start=1):
            if col in det_groups and j < len(row):
                value = _number(row[j])
                model.loc[row_of[group], col] = 0.0 if not np.isfinite(value) else value
        if export_col is not None and export_col < len(row):
            if _number(row[export_col]) > 0:
                exported.append(group)
    if exported:
        warnings.warn(
            f"EwE-F detritus fate exports part of the flows of {exported}; "
            "that share is not routed to any detritus group",
            UserWarning,
            stacklevel=3,
        )


def _fill_stanzas(
    params: RpathParams,
    data: Dict[str, List[float]],
    names: List[str],
    split_stages: Dict[int, List[str]],
    split_names: Dict[int, str],
    growth_path: Optional[Path],
) -> None:
    growth: Dict[str, Dict[str, float]] = {}
    if growth_path is not None:
        header, rows = _read_table(growth_path, "Groups")
        labels = [h.lower() for h in header]
        for row in rows:
            values = {
                labels[j]: _number(row[j]) for j in range(1, min(len(row), len(labels)))
            }
            growth[row[0].strip()] = values

    def _growth(stages: List[str], key_start: str, default: float) -> float:
        for stage in stages:
            if stage in growth:
                for label, value in growth[stage].items():
                    if label.startswith(key_start) and np.isfinite(value):
                        return value
        return default

    st = params.stanzas
    row_of = {g: i for i, g in enumerate(names)}
    stgroups = st.stgroups.set_index("StanzaGroup")
    records = []
    for number, stages in split_stages.items():
        name = split_names[number]
        stgroups.loc[name, "VBGF_Ksp"] = _growth(stages, "k", np.nan)
        stgroups.loc[name, "RecPower"] = _growth(stages, "rec", 1.0)
        stgroups.loc[name, "BAB"] = _growth(stages, "ba/b", 0.0)
        stgroups.loc[name, "Wmat"] = _growth(stages, "wmat", np.nan)
        ordered = sorted(stages, key=lambda g: data["age_start"][row_of[g]])
        firsts = [int(data["age_start"][row_of[g]]) for g in ordered]
        for k, group in enumerate(ordered):
            i = row_of[group]
            records.append(
                {
                    "StGroupNum": int(stgroups.loc[name, "StGroupNum"]),
                    "StanzaGroup": name,
                    "StanzaNum": k + 1,
                    "GroupNum": i + 1,
                    "Group": group,
                    "First": firsts[k],
                    "Last": firsts[k + 1] - 1 if k + 1 < len(firsts) else 999,
                    # EwE takes a stage's total mortality from its P/B
                    "Z": data["PB"][i],
                    "Leading": bool(
                        np.isfinite(data["leading"][i]) and data["leading"][i] > 0
                    ),
                }
            )
    st.stgroups = stgroups.reset_index()[list(st.stgroups.columns)]
    st.stindiv = pd.DataFrame(records)


# --- Ecosim ---------------------------------------------------------------------


def read_ewef_ecosim(path: PathLike, params: RpathParams) -> Dict[str, Any]:
    """Read EwE-F Ecosim settings as ``scenario_overrides``.

    Parameters
    ----------
    path : str or Path
        The ``ecosim-f`` namelist (``*_filenames.nml``) or its directory.
    params : RpathParams
        The model (from :func:`read_ewef`); groups are matched by name.

    Returns
    -------
    dict
        ``vv_overrides`` {(prey, predator): vulnerability}, and per group
        ``pb_maxs`` (producers), ``ftime_max``, ``ftime_adjust``,
        ``mo_pred`` and ``switch_power`` {group: value}, with 0-based model
        indices, for :func:`pypath.core.ecosim.rsim_scenario`.
    """
    nml_path = _find_namelist(path)
    nml = read_ewef_namelist(nml_path)
    names = [str(g).strip() for g in params.model["Group"]]
    types = list(params.model["Type"])
    index = {g: i for i, g in enumerate(names)}
    overrides: Dict[str, Any] = {}
    unapplied: List[str] = []

    info_path = _resolve(nml_path, nml.get("GroupInfo_fname"))
    if info_path is not None:
        header, rows = _read_table(info_path, "Groups")
        fields = {}
        for j, label in enumerate(header[1:], start=1):
            low = label.lower()
            for start, key in _ECOSIM_COLUMNS:
                if low.startswith(start) and key not in fields:
                    fields[key] = j
                    break
        values: Dict[str, Dict[int, float]] = {k: {} for k in fields}
        for row in rows:
            group = row[0].strip()
            if group not in index:
                raise ValueError(f"{info_path.name}: group {group!r} not in the model")
            for key, j in fields.items():
                value = _number(row[j]) if j < len(row) else np.nan
                if np.isfinite(value):
                    values[key][index[group]] = value
        for key, target in (
            ("ftime_max", "ftime_max"),
            ("ftime_adjust", "ftime_adjust"),
            ("mo_pred", "mo_pred"),
            ("switch_power", "switch_power"),
        ):
            if values.get(key):
                overrides[target] = values[key]
        pb = {g: v for g, v in values.get("pb_max", {}).items() if types[g] == 1}
        if pb:
            overrides["pb_maxs"] = pb

        def _differs(key, default, label):
            off = {
                names[g]: v
                for g, v in values.get(key, {}).items()
                if not np.isclose(v, default)
            }
            if off:
                unapplied.append(f"{label} {off}")

        _differs("ftime", 1.0, "initial feeding time")
        _differs("pred_effect", 0.0, "predator effect on feeding time")
        _differs("qmax_qo", 1.0, "density-dependent catchability Qmax/Qo")
        _differs(
            "qbmax_qbo",
            1000.0,
            "QBmax/QBo (handling time; PyPath's handling term stays at its default)",
        )

    vul_path = _resolve(nml_path, nml.get("Vulnerability_fname"))
    if vul_path is not None:
        overrides["vv_overrides"] = _read_vulnerabilities(vul_path, params, index)

    if _resolve(nml_path, nml.get("NutrientForcing_fname")) is not None:
        unapplied.append(
            "nutrient limitation of primary production (NutrientForcing_fname, "
            f"NutBaseFreeProp={nml.get('NutBaseFreeProp')}, "
            f"NutPBmax={nml.get('NutPBmax')})"
        )
    relax = nml.get("relax")
    if isinstance(relax, (int, float)) and not np.isclose(relax, 0.5):
        unapplied.append(f"relax={relax} (PyPath relaxes fast groups with 0.5)")
    steps = nml.get("StepsPerMonth")
    if isinstance(steps, int) and steps != 1:
        unapplied.append(
            f"StepsPerMonth={steps} (PyPath takes one RK4 step per month under "
            "the EwE fast-group rule)"
        )
    forcing_path = _resolve(nml_path, nml.get("Forcing_fname"))
    if forcing_path is not None:
        types_line = _forcing_types(forcing_path)
        if any(t in (4, 5) for t in types_line):
            unapplied.append(
                f"fishing forcing of type 4/5 in {forcing_path.name} (forced F or "
                "Z); add it with RsimFishing.ForcedFRate"
            )
    if unapplied:
        warnings.warn(
            "EwE-F Ecosim settings PyPath does not apply: " + "; ".join(unapplied),
            UserWarning,
            stacklevel=2,
        )
    return overrides


def _read_vulnerabilities(
    vul_path: Path, params: RpathParams, index: Dict[str, int]
) -> Dict[Tuple[int, int], float]:
    header, rows = _read_table(vul_path, "Prey/Pred")
    predators = header[1:]
    diet = params.diet.set_index(params.diet["Group"].astype(str).str.strip())
    vv: Dict[Tuple[int, int], float] = {}
    missing = []
    for row in rows:
        prey = row[0].strip()
        if prey not in index:
            raise ValueError(f"{vul_path.name}: prey {prey!r} not in the model")
        for j, pred in enumerate(predators, start=1):
            if not pred or j >= len(row):
                continue
            if pred not in index:
                raise ValueError(f"{vul_path.name}: predator {pred!r} not in the model")
            value = _number(row[j])
            if np.isfinite(value):
                vv[(index[prey], index[pred])] = value
            elif (
                pred in diet.columns
                and prey in diet.index
                and float(np.nan_to_num(diet.loc[prey, pred])) > 0
            ):
                missing.append(f"{pred} eats {prey}")
    if missing:
        warnings.warn(
            f"EwE-F gives no vulnerability for diet links {missing}: EwE-F's Ecosim "
            "skips them; PyPath keeps them at the default vulnerability, as EwE 6 does",
            UserWarning,
            stacklevel=3,
        )
    return vv


def _forcing_types(path: Path) -> List[int]:
    """The ``types`` line of an EwE-F fishing forcing file."""
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        cells = line.replace(",", " ").split()
        if cells and cells[0].lower() == "types":
            out = []
            for c in cells[1:]:
                try:
                    out.append(int(float(c)))
                except ValueError:
                    pass
            return out
    return []


def ecosim_scenario_from_ewef(
    ecopath_path: PathLike,
    ecosim_path: PathLike,
    years: Optional[range] = None,
    fast_group_rule: Optional[str] = None,
):
    """An Ecosim scenario from an EwE-F model's Ecopath and Ecosim inputs.

    ``years`` defaults to EwE-F's run length ``tf`` (years) from the Ecosim
    namelist. Fishing stays at the Ecopath rates, as in EwE-F for forcing of
    type 1.
    """
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario

    params = read_ewef(ecopath_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = rpath(params)
    overrides = read_ewef_ecosim(ecosim_path, params)
    if years is None:
        tf = read_ewef_namelist(_find_namelist(ecosim_path)).get("tf")
        n = int(tf) if isinstance(tf, (int, float)) and tf >= 1 else 10
        years = range(1, max(n, 2) + 1)
    return rsim_scenario(
        balanced,
        params,
        years=years,
        scenario_overrides=overrides,
        fast_group_rule=fast_group_rule,
    )


# --- results ----------------------------------------------------------------------


def read_ewef_results(
    path: PathLike, groups: Optional[List[str]] = None
) -> pd.DataFrame:
    """EwE-F's Ecopath result (the ``ep_data`` table of its HDF5 file).

    Returns one row per group with ``Biomass``, ``PB``, ``QB``, ``EE``,
    ``ProdCons``, production, consumption, respiration and the other
    ``ep_data`` fields; -999 becomes NaN. ``groups`` names the rows (EwE-F
    stores none). Needs ``h5py``.
    """
    try:
        import h5py
    except ImportError as e:  # optional dependency
        raise ImportError("read_ewef_results needs h5py (conda install h5py)") from e
    with h5py.File(str(path), "r") as fh:
        table = fh["ep_data"][()]
    df = pd.DataFrame({name: table[name] for name in table.dtype.names})
    df = df.rename(
        columns={
            "biomass": "Biomass",
            "PoB": "PB",
            "QoB": "QB",
            "PoQ": "ProdCons",
            "unass_Q": "Unassim",
        }
    )
    for col in df.columns:
        if df[col].dtype.kind == "f":
            df[col] = df[col].astype(float).where(df[col] > EWEF_MISSING + 0.5)
    if groups is not None:
        if len(groups) != len(df):
            raise ValueError(f"{len(groups)} group names for {len(df)} EwE-F groups")
        df.index = list(groups)
    return df
