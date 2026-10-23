"""Results export and formatting utilities for PyPath Shiny app."""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path


def format_model_summary(params, include_diet: bool = True) -> pd.DataFrame:
    """Format model parameters as a summary table for export."""
    if not hasattr(params, "model"):
        raise ValueError("params must have a model attribute")
    
    model = params.model.copy()
    
    for col in ["Group", "Type", "Biomass", "PB"]:
        if col not in model.columns:
            model[col] = np.nan
    
    return model


def create_results_metadata(
    model_summary: Optional[pd.DataFrame] = None,
    simulation_output: Optional[dict] = None,
    user_notes: str = "",
) -> Dict:
    """Create metadata for results export."""
    from datetime import datetime
    
    metadata = {
        "export_timestamp": datetime.now().isoformat(),
        "notes": user_notes,
        "model_info": {},
        "simulation_info": {},
    }
    
    if model_summary is not None:
        metadata["model_info"]["num_groups"] = len(model_summary)
        if "Type" in model_summary.columns:
            type_col = model_summary["Type"]
            metadata["model_info"]["consumers"] = int((type_col == 0).sum())
            metadata["model_info"]["producers"] = int((type_col == 1).sum())
            metadata["model_info"]["detritus"] = int((type_col == 2).sum())
            metadata["model_info"]["fleets"] = int((type_col == 3).sum())
        if "Biomass" in model_summary.columns:
            metadata["model_info"]["total_biomass"] = float(model_summary["Biomass"].sum())
    
    if simulation_output is not None:
        if "monthly_biomass" in simulation_output:
            biomass = simulation_output["monthly_biomass"]
            biomass_arr = np.asarray(biomass)
            metadata["simulation_info"]["timesteps"] = len(biomass_arr)
            metadata["simulation_info"]["num_groups"] = (
                biomass_arr.shape[1] if len(biomass_arr.shape) > 1 else 1
            )
        if "monthly_catch" in simulation_output and simulation_output["monthly_catch"] is not None:
            catch = simulation_output["monthly_catch"]
            catch_arr = np.asarray(catch)
            metadata["simulation_info"]["num_fleets"] = (
                catch_arr.shape[1] if len(catch_arr.shape) > 1 else 1
            )
    
    return metadata


def validate_export_data(
    model_summary: Optional[pd.DataFrame] = None,
    simulation_output: Optional[dict] = None,
) -> Tuple[bool, List[str]]:
    """Validate data before export."""
    errors = []
    
    if model_summary is not None:
        if not isinstance(model_summary, pd.DataFrame):
            errors.append("model_summary must be a DataFrame")
        elif model_summary.empty:
            errors.append("model_summary is empty")
        elif "Group" not in model_summary.columns:
            errors.append("model_summary missing Group column")
    
    if simulation_output is not None:
        if not isinstance(simulation_output, dict):
            errors.append("simulation_output must be a dictionary")
        elif "monthly_biomass" not in simulation_output:
            errors.append("simulation_output missing monthly_biomass key")
    
    return len(errors) == 0, errors


def export_results_to_json(
    model_summary: Optional[pd.DataFrame] = None,
    simulation_output: Optional[dict] = None,
    metadata: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> str:
    """Export results to JSON format."""
    is_valid, errors = validate_export_data(model_summary, simulation_output)
    if not is_valid:
        raise ValueError(f"Invalid export data: {'; '.join(errors)}")
    
    data = {}
    
    if metadata is not None:
        data["metadata"] = metadata
    
    if model_summary is not None:
        data["model"] = model_summary.to_dict(orient="records")
    
    if simulation_output is not None:
        sim_data = {}
        for key in ["monthly_biomass", "monthly_catch", "effort", "time"]:
            if key in simulation_output and simulation_output[key] is not None:
                arr = np.asarray(simulation_output[key])
                sim_data[key] = arr.tolist()
        if sim_data:
            data["simulation"] = sim_data
    
    json_str = json.dumps(data, indent=2, default=str)
    
    if output_path:
        Path(output_path).write_text(json_str, encoding="utf-8")
        return output_path
    
    return json_str


def prepare_csv_export_tables(
    model_summary: Optional[pd.DataFrame] = None,
    simulation_output: Optional[dict] = None,
    annual_summary: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Prepare multiple CSV export tables."""
    from pypath_shiny.pages.ecosim_utils import (
        format_timeseries_for_export,
        get_timeseries_summary,
    )
    
    tables = {}
    
    if model_summary is not None:
        tables["model"] = model_summary.copy()
    
    if simulation_output is not None:
        try:
            timeseries = format_timeseries_for_export(simulation_output, annual_summary=False)
            tables["timeseries"] = timeseries
            
            if annual_summary:
                annual = format_timeseries_for_export(simulation_output, annual_summary=True)
                tables["annual"] = annual
            
            summary = get_timeseries_summary(simulation_output)
            tables["summary"] = summary
        except Exception:
            pass
    
    return tables


def format_results_for_download(
    model_summary: Optional[pd.DataFrame] = None,
    simulation_output: Optional[dict] = None,
    format_type: str = "csv",
) -> Tuple[str, str]:
    """Format results for browser download."""
    is_valid, errors = validate_export_data(model_summary, simulation_output)
    if not is_valid:
        raise ValueError(f"Invalid export data: {'; '.join(errors)}")
    
    if format_type == "json":
        content = export_results_to_json(model_summary, simulation_output)
        filename = "pypath_results.json"
    
    elif format_type == "html":
        html_parts = []
        html_parts.append("<html><head><title>PyPath Results</title></head><body>")
        
        if model_summary is not None:
            html_parts.append("<h2>Model Summary</h2>")
            html_parts.append(model_summary.to_html())
        
        if simulation_output is not None:
            html_parts.append("<h2>Simulation Results</h2>")
            html_parts.append("<p>Simulation completed successfully.</p>")
        
        html_parts.append("</body></html>")
        content = "\n".join(html_parts)
        filename = "pypath_results.html"
    
    elif format_type == "csv":
        tables = prepare_csv_export_tables(model_summary, simulation_output)
        csv_parts = []
        for table_name, df in tables.items():
            csv_parts.append(f"# {table_name.upper()}")
            csv_parts.append(df.to_csv(index=False))
            csv_parts.append("")
        content = "\n".join(csv_parts)
        filename = "pypath_results.csv"
    
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
    
    return content, filename


def create_download_bundle(
    model_summary: Optional[pd.DataFrame] = None,
    simulation_output: Optional[dict] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict[str, str], str]:
    """Create a bundle of export files for multi-file download."""
    files = {}
    
    if model_summary is not None:
        files["model.csv"] = model_summary.to_csv(index=False)
    
    if simulation_output is not None:
        from pypath_shiny.pages.ecosim_utils import (
            format_timeseries_for_export,
            get_timeseries_summary,
        )
        
        timeseries = format_timeseries_for_export(simulation_output)
        files["timeseries.csv"] = timeseries.to_csv(index=False)
        
        annual = format_timeseries_for_export(simulation_output, annual_summary=True)
        files["annual_summary.csv"] = annual.to_csv(index=False)
        
        summary = get_timeseries_summary(simulation_output)
        files["group_summary.csv"] = summary.to_csv(index=False)
    
    if metadata is not None:
        metadata_json = json.dumps(metadata, indent=2, default=str)
        files["metadata.json"] = metadata_json
    
    return files, "pypath_results.zip"
