"""Ecosim utilities for UI enhancements and result formatting."""
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


def format_timeseries_for_export(
    output: dict,
    include_biomass: bool = True,
    include_catch: bool = True,
    include_effort: bool = True,
    annual_summary: bool = False,
) -> pd.DataFrame:
    """Format Ecosim simulation output for CSV export."""
    if not output or "monthly_biomass" not in output:
        raise ValueError("Output must contain monthly_biomass key")
    
    months = output.get("time", None)
    if months is None:
        n_months = len(output["monthly_biomass"])
        months = np.arange(n_months) / 12.0
    else:
        months = np.asarray(months)
    
    if annual_summary:
        years = np.floor(months).astype(int)
        n_years = len(np.unique(years))
        data = {"Year": np.arange(n_years)}
    else:
        data = {"Time_months": np.arange(len(months))}
    
    if include_biomass and "monthly_biomass" in output:
        biomass = np.asarray(output["monthly_biomass"])
        n_groups = biomass.shape[1] if len(biomass.shape) > 1 else 1
        
        if annual_summary:
            years_unique = np.unique(np.floor(months).astype(int))
            for g in range(n_groups):
                group_biomass = biomass[:, g] if len(biomass.shape) > 1 else biomass
                annual_vals = []
                for year in years_unique:
                    year_mask = np.floor(months) == year
                    annual_vals.append(np.mean(group_biomass[year_mask]))
                data[f"Biomass_Group{g}"] = annual_vals
        else:
            for g in range(n_groups):
                group_biomass = biomass[:, g] if len(biomass.shape) > 1 else biomass
                data[f"Biomass_Group{g}"] = group_biomass
    
    if include_catch and "monthly_catch" in output and output["monthly_catch"] is not None:
        catch = np.asarray(output["monthly_catch"])
        if len(catch.shape) > 1:
            n_fleets = catch.shape[1]
            if annual_summary:
                years_unique = np.unique(np.floor(months).astype(int))
                for f in range(n_fleets):
                    annual_vals = []
                    for year in years_unique:
                        year_mask = np.floor(months) == year
                        annual_vals.append(np.sum(catch[year_mask, f]))
                    data[f"Catch_Fleet{f}"] = annual_vals
            else:
                for f in range(n_fleets):
                    data[f"Catch_Fleet{f}"] = catch[:, f]
    
    if include_effort and "effort" in output and output["effort"] is not None:
        effort = np.asarray(output["effort"])
        if len(effort.shape) > 1:
            n_fleets = effort.shape[1]
            if annual_summary:
                years_unique = np.unique(np.floor(months).astype(int))
                for f in range(n_fleets):
                    annual_vals = []
                    for year in years_unique:
                        year_mask = np.floor(months) == year
                        annual_vals.append(np.mean(effort[year_mask, f]))
                    data[f"Effort_Fleet{f}"] = annual_vals
            else:
                for f in range(n_fleets):
                    data[f"Effort_Fleet{f}"] = effort[:, f]
    
    return pd.DataFrame(data)


def validate_ecosim_output(output: dict) -> Tuple[bool, List[str]]:
    """Validate Ecosim simulation output structure."""
    errors = []
    
    if not isinstance(output, dict):
        return False, ["Output must be a dictionary"]
    
    if "monthly_biomass" not in output:
        errors.append("Output missing monthly_biomass key")
    elif not isinstance(output["monthly_biomass"], (np.ndarray, list)):
        errors.append("monthly_biomass must be array-like")
    elif len(np.asarray(output["monthly_biomass"]).shape) != 2:
        errors.append("monthly_biomass must be 2D array (time, groups)")
    
    for key in ["monthly_catch", "effort", "time"]:
        if key in output and output[key] is not None:
            if not isinstance(output[key], (np.ndarray, list)):
                errors.append(f"{key} must be array-like or None")
    
    return len(errors) == 0, errors


def get_timeseries_summary(
    output: dict,
    group_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate summary statistics for each group's time-series."""
    is_valid, errors = validate_ecosim_output(output)
    if not is_valid:
        raise ValueError(f"Invalid output: {'; '.join(errors)}")
    
    biomass = np.asarray(output["monthly_biomass"])
    if len(biomass.shape) == 1:
        biomass = biomass.reshape(-1, 1)
    
    n_groups = biomass.shape[1]
    
    if group_names is None:
        group_names = [f"Group{i}" for i in range(n_groups)]
    elif len(group_names) != n_groups:
        raise ValueError(f"Expected {n_groups} group names, got {len(group_names)}")
    
    summary_data = {
        "Group": [],
        "Initial_Biomass": [],
        "Final_Biomass": [],
        "Min_Biomass": [],
        "Max_Biomass": [],
        "Mean_Biomass": [],
        "StdDev_Biomass": [],
        "Range": [],
    }
    
    for g in range(n_groups):
        group_biomass = biomass[:, g]
        summary_data["Group"].append(group_names[g])
        summary_data["Initial_Biomass"].append(float(group_biomass[0]))
        summary_data["Final_Biomass"].append(float(group_biomass[-1]))
        summary_data["Min_Biomass"].append(float(np.min(group_biomass)))
        summary_data["Max_Biomass"].append(float(np.max(group_biomass)))
        summary_data["Mean_Biomass"].append(float(np.mean(group_biomass)))
        summary_data["StdDev_Biomass"].append(float(np.std(group_biomass)))
        summary_data["Range"].append(
            float(np.max(group_biomass) - np.min(group_biomass))
        )
    
    return pd.DataFrame(summary_data)


def detect_group_trends(
    output: dict,
    window_size: int = 12,
) -> pd.DataFrame:
    """Detect biomass trends (increasing/decreasing) for each group."""
    is_valid, errors = validate_ecosim_output(output)
    if not is_valid:
        raise ValueError(f"Invalid output: {'; '.join(errors)}")
    
    biomass = np.asarray(output["monthly_biomass"])
    if len(biomass.shape) == 1:
        biomass = biomass.reshape(-1, 1)
    
    n_groups = biomass.shape[1]
    trends_data = {
        "Group": [],
        "Trend": [],
        "Slope": [],
        "Significance": [],
    }
    
    for g in range(n_groups):
        group_biomass = biomass[:, g]
        
        x = np.arange(len(group_biomass))
        slope = np.polyfit(x, group_biomass, 1)[0]
        
        if abs(slope) < 1e-6:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        mean_biomass = np.mean(group_biomass)
        relative_slope = (slope / mean_biomass) if mean_biomass > 0 else 0
        
        if abs(relative_slope) < 0.001:
            significance = "low"
        elif abs(relative_slope) < 0.01:
            significance = "medium"
        else:
            significance = "high"
        
        trends_data["Group"].append(f"Group{g}")
        trends_data["Trend"].append(trend)
        trends_data["Slope"].append(float(slope))
        trends_data["Significance"].append(significance)
    
    return pd.DataFrame(trends_data)
