"""Ecopath model page module."""

from typing import List, Union

import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui

from pypath.core.ecopath import Rpath, rpath

# pypath imports (path setup handled by app/__init__.py)
from pypath.core.params import RpathParams, create_rpath_params

# Import shared utilities
from .utils import (
    COLUMN_TOOLTIPS,
    NO_DATA_STYLE,
    STANZA_STYLE,
    create_cell_styles,
    format_dataframe_for_display,
    is_balanced_model,
)
from .validation import (
    validate_biomass,
    validate_ee,
    validate_model_parameters,
    validate_pb,
)

# Configuration imports
try:
    from app.config import DEFAULTS, PLOTS, THRESHOLDS
except ModuleNotFoundError:
    from config import DEFAULTS, PLOTS, THRESHOLDS


def _get_groups_from_model(model: Union[Rpath, RpathParams]) -> List[str]:
    """Safely extract group names from Rpath or RpathParams object.

    This helper function provides a unified interface for getting group names
    from either a balanced Rpath model or an unbalanced RpathParams object,
    handling their different internal structures.

    Parameters
    ----------
    model : Union[Rpath, RpathParams]
        Either a balanced Rpath object or unbalanced RpathParams object

    Returns
    -------
    List[str]
        List of group names in the order they appear in the model

    Raises
    ------
    ValueError
        If the model object doesn't have recognizable group information

    Notes
    -----
    **Rpath objects** store group names directly in the `Group` attribute.
    **RpathParams objects** store them in `model` DataFrame under 'Group' column.

    Examples
    --------
    >>> from pypath.core.params import create_rpath_params
    >>> params = create_rpath_params(['Fish', 'Plankton'], [0, 1])
    >>> groups = _get_groups_from_model(params)
    >>> groups
    ['Fish', 'Plankton']
    """
    if hasattr(model, "Group"):
        # It's a balanced Rpath object
        return list(model.Group)
    elif hasattr(model, "model") and "Group" in model.model.columns:
        # It's an RpathParams object
        return list(model.model["Group"])
    else:
        raise ValueError("Cannot determine group names from model object")


def _recreate_params_from_model(model: Rpath) -> RpathParams:
    """Recreate RpathParams from a balanced Rpath model.

    This function reconstructs an editable RpathParams object from a balanced
    Rpath model, allowing users to modify parameters and re-balance. The
    reconstruction preserves biomass, production, consumption, and diet
    information from the balanced model.

    Parameters
    ----------
    model : Rpath
        A balanced Rpath model object with computed parameters

    Returns
    -------
    RpathParams
        Unbalanced parameter object with values populated from the balanced model.
        This can be edited and re-balanced.

    Raises
    ------
    ValueError
        If model doesn't have required attributes (Group, type, etc.)

    Notes
    -----
    **Reconstructed Parameters:**
    - Biomass, PB, QB, EE, Unassim, BioAcc from balanced model
    - Diet matrix from DC (diet composition) matrix
    - Group types preserved

    **Not Reconstructed:**
    - Fishing catches (landings/discards) - can be edited afterward
    - Multi-stanza parameters - would need separate handling

    The resulting RpathParams object is unbalanced and will need to be
    re-balanced with `rpath()` after any modifications.

    Examples
    --------
    >>> from pypath.core.params import create_rpath_params
    >>> from pypath.core.ecopath import rpath
    >>> params = create_rpath_params(['Fish', 'Plankton'], [0, 1])
    >>> # ... set parameters ...
    >>> balanced = rpath(params)
    >>> # Now recreate editable params
    >>> params_copy = _recreate_params_from_model(balanced)
    >>> # Modify and re-balance
    >>> params_copy.model.loc[0, 'PB'] = 1.5
    >>> new_balanced = rpath(params_copy)
    """
    # Create basic params structure
    groups = _get_groups_from_model(model)

    # Get types
    if hasattr(model, "type"):
        types = list(model.type)
    elif hasattr(model, "model") and "Type" in model.model.columns:
        types = list(model.model["Type"])
    else:
        raise ValueError("Cannot determine types from model object")

    params = create_rpath_params(groups, types)

    # Fill in the balanced parameter values
    params.model["Biomass"] = model.Biomass
    params.model["PB"] = model.PB
    params.model["QB"] = model.QB
    params.model["EE"] = model.EE
    params.model["Unassim"] = model.Unassim
    params.model["BioAcc"] = model.BA

    # Set types
    params.model["Type"] = types

    # Reconstruct diet matrix from DC (diet composition)
    # DC is (ngroups + 1, nliving) where last row is import
    nliving = model.NUM_LIVING
    for i in range(model.NUM_GROUPS):
        for j in range(nliving):
            if i < nliving:  # Living groups eat
                params.diet.iloc[i, j + 1] = model.DC[i, j]

    # Note: Fishing catches would need to be reconstructed from Landings/Discards
    # For now, we'll leave them as-is (they can be edited later)

    return params


# Public helper to convert DataGrid edits to numeric values
def _convert_input_to_numeric(new_value):
    """Convert a new_value from a DataGrid edit to a numeric value.

    Treat explicit zeros (0 or '0') as valid numeric zero. Treat blank
    strings and None as np.nan.
    """
    if new_value is None:
        return np.nan
    if isinstance(new_value, str) and new_value.strip() == "":
        return np.nan
    try:
        return float(new_value)
    except (ValueError, TypeError):
        # Let caller handle exceptions for invalid numeric formats
        raise


def ecopath_ui():
    """Ecopath model page UI."""
    return ui.page_fluid(
        ui.h2("Ecopath Mass-Balance Model", class_="mb-4"),
        ui.layout_sidebar(
            # Sidebar for model setup
            ui.sidebar(
                # Run Model section at the top
                ui.h5("Run Model"),
                ui.input_action_button(
                    "btn_balance", "Balance Model", class_="btn-success w-100"
                ),
                ui.download_button(
                    "download_params",
                    "Download Parameters",
                    class_="btn-outline-secondary w-100 mt-2",
                ),
                ui.tags.hr(),
                # Collapsible Model Setup section
                ui.tags.details(
                    ui.tags.summary(
                        ui.tags.strong("Model Setup"),
                        style="cursor: pointer; padding: 5px 0;",
                    ),
                    ui.div(
                        # Model name
                        ui.input_text("eco_name", "Model Name", value="My Ecosystem"),
                        ui.tags.hr(),
                        # Group definition section
                        ui.h6("Define Groups"),
                        ui.input_text_area(
                            "group_names",
                            "Group Names (one per line)",
                            value="Phytoplankton\nZooplankton\nSmall Fish\nLarge Fish\nDetritus\nFleet",
                            rows=6,
                        ),
                        ui.input_text_area(
                            "group_types",
                            "Group Types (one per line: 1=producer, 0=consumer, 2=detritus, 3=fleet)",
                            value="1\n0\n0\n0\n2\n3",
                            rows=6,
                        ),
                        ui.input_action_button(
                            "btn_create_params",
                            "Create Parameter Template",
                            class_="btn-primary w-100 mt-3",
                        ),
                        ui.tags.hr(),
                        # File upload
                        ui.h6("Or Load from File"),
                        ui.input_file(
                            "upload_params",
                            "Upload Parameters (CSV)",
                            accept=[".csv"],
                            multiple=False,
                        ),
                        style="padding-top: 10px;",
                    ),
                ),
                width=300,
            ),
            # Main content area
            ui.navset_card_tab(
                ui.nav_panel(
                    "Model Parameters",
                    ui.h4("Basic Parameters", class_="mt-3"),
                    # Legend for cell styling
                    ui.div(
                        ui.tags.span(
                            ui.tags.span(
                                "",
                                style="display: inline-block; width: 16px; height: 16px; background-color: #f0f0f0; border: 1px solid #ccc; margin-right: 4px; vertical-align: middle;",
                            ),
                            " No data (was 9999)",
                            style="margin-right: 16px; font-size: 0.85em; color: #666;",
                        ),
                        ui.tags.span(
                            ui.tags.span(
                                "",
                                style="display: inline-block; width: 16px; height: 16px; background-color: #fff9e6; border-bottom: 2px dashed #f0ad4e; border-left: 1px solid #ccc; border-right: 1px solid #ccc; border-top: 1px solid #ccc; margin-right: 4px; vertical-align: middle;",
                            ),
                            " Has remark (from EwE file)",
                            style="font-size: 0.85em; color: #666;",
                        ),
                        class_="mb-2",
                    ),
                    ui.output_data_frame("model_params_table"),
                    # Parameter help section
                    ui.div(
                        ui.tags.details(
                            ui.tags.summary(
                                ui.tags.i(class_="bi bi-info-circle me-2"),
                                "Parameter Descriptions",
                                style="cursor: pointer; color: #0066cc;",
                            ),
                            ui.div(
                                ui.tags.dl(
                                    ui.tags.dt("Group"),
                                    ui.tags.dd(
                                        "Name of the functional group (species or group of species)"
                                    ),
                                    ui.tags.dt("Type"),
                                    ui.tags.dd(
                                        "Group type: 0=Consumer, 1=Producer, 2=Detritus, 3=Fleet"
                                    ),
                                    ui.tags.dt("Biomass"),
                                    ui.tags.dd(
                                        "Biomass (t/km²) - standing stock of the group"
                                    ),
                                    ui.tags.dt("PB"),
                                    ui.tags.dd(
                                        "Production/Biomass ratio (1/year) - turnover rate"
                                    ),
                                    ui.tags.dt("QB"),
                                    ui.tags.dd(
                                        "Consumption/Biomass ratio (1/year) - feeding rate (grey for producers/detritus)"
                                    ),
                                    ui.tags.dt("EE"),
                                    ui.tags.dd(
                                        "Ecotrophic Efficiency (0-1) - fraction of production used in the system"
                                    ),
                                    ui.tags.dt("Unassim"),
                                    ui.tags.dd(
                                        "Unassimilated consumption (0-1) - fraction of food not assimilated (grey for producers/detritus)"
                                    ),
                                    ui.tags.dt("BioAcc"),
                                    ui.tags.dd(
                                        "Biomass accumulation rate (t/km²/year) - change in biomass over time"
                                    ),
                                    class_="row",
                                ),
                                class_="mt-3 p-3 border rounded",
                            ),
                        ),
                        class_="mt-2",
                    ),
                    # Show remarks panel if any exist
                    ui.output_ui("remarks_panel"),
                ),
                ui.nav_panel(
                    "Diet Matrix",
                    ui.h4("Diet Composition", class_="mt-3"),
                    ui.p(
                        "Enter diet fractions (columns must sum to 1.0 for each predator)"
                    ),
                    ui.output_data_frame("diet_matrix_table"),
                ),
                ui.nav_panel(
                    "Fisheries",
                    ui.h4("Landings & Discards", class_="mt-3"),
                    ui.output_data_frame("fisheries_table"),
                ),
                ui.nav_panel(
                    "Model Results",
                    ui.h4("Balanced Model Results", class_="mt-3"),
                    ui.output_ui("balance_status"),
                    ui.output_data_frame("model_results_table"),
                ),
                ui.nav_panel(
                    "Multi-Stanza",
                    ui.h4("Multi-Stanza Groups", class_="mt-3"),
                    ui.p(
                        "Multi-stanza groups link age-structured life stages (e.g., juvenile/adult) "
                        "that share growth and mortality parameters.",
                        class_="text-muted",
                    ),
                    ui.output_ui("stanza_status"),
                    ui.h5("Stanza Group Configuration", class_="mt-3"),
                    ui.output_data_frame("stanza_groups_table"),
                    ui.h5("Individual Life Stages", class_="mt-3"),
                    ui.output_data_frame("stanza_indiv_table"),
                ),
                ui.nav_panel(
                    "Diagnostics",
                    ui.h4("Model Diagnostics", class_="mt-3"),
                    ui.output_ui("diagnostics_output"),
                    ui.layout_columns(
                        ui.output_plot("trophic_level_plot"),
                        ui.output_plot("ee_plot"),
                        col_widths=[6, 6],
                    ),
                ),
            ),
        ),
    )


def ecopath_server(
    input: Inputs, output: Outputs, session: Session, model_data: reactive.Value
):
    """Ecopath model page server logic."""

    # Reactive values for this page
    params = reactive.Value(None)
    balanced_model = reactive.Value(None)

    # Watch for changes in model_data (from imports or other sources)
    @reactive.effect
    def _sync_model_data():
        """Sync local params with shared model_data."""
        imported = model_data.get()
        if imported is not None:
            # Check if it's an RpathParams (not a balanced Rpath model)
            if hasattr(imported, "model") and hasattr(imported, "diet"):
                # It's RpathParams - use it
                params.set(imported)
                n_groups = len(imported.model)
                n_diet = imported.diet.iloc[:, 1:].notna().sum().sum()
                ui.notification_show(
                    f"Loaded model: {n_groups} groups, {n_diet} diet values",
                    type="message",
                )
            elif hasattr(imported, "NUM_GROUPS"):
                # It's a balanced Rpath model - recreate params from balanced values
                recreated_params = _recreate_params_from_model(imported)
                params.set(recreated_params)
                ui.notification_show(
                    f"Loaded balanced model: {imported.NUM_GROUPS} groups",
                    type="message",
                )

    @reactive.effect
    @reactive.event(input.btn_create_params)
    def _create_params():
        """Create parameter template from group definitions."""
        try:
            # Parse group names
            names = [n.strip() for n in input.group_names().split("\n") if n.strip()]
            types_str = [
                t.strip() for t in input.group_types().split("\n") if t.strip()
            ]

            if len(names) != len(types_str):
                ui.notification_show(
                    f"Number of names ({len(names)}) must match number of types ({len(types_str)})",
                    type="error",
                )
                return

            types = [int(t) for t in types_str]

            # Create parameters
            new_params = create_rpath_params(names, types)
            params.set(new_params)

            ui.notification_show(
                f"Created parameter template with {len(names)} groups", type="message"
            )
        except Exception as e:
            ui.notification_show(f"Error creating parameters: {str(e)}", type="error")

    def add_header_tooltips(columns: list) -> list:
        """Create column definitions with tooltips for DataGrid headers."""
        col_defs = []
        for col in columns:
            tooltip = COLUMN_TOOLTIPS.get(col, f"{col} parameter")
            col_defs.append(
                {
                    "id": col,
                    "name": col,
                    "title": tooltip,  # Tooltip on hover
                }
            )
        return col_defs

    @output
    @render.data_frame
    def model_params_table():
        """Render editable model parameters table."""
        p = params.get()
        if p is None:
            return render.DataGrid(
                pd.DataFrame({"Message": ["Load or create a model first"]})
            )

        # Select key columns for display
        display_cols = [
            "Group",
            "Type",
            "Biomass",
            "PB",
            "QB",
            "EE",
            "Unassim",
            "BioAcc",
        ]
        cols = [c for c in display_cols if c in p.model.columns]

        df = p.model[cols].copy()

        # Get remarks if available
        remarks_df = (
            p.remarks if hasattr(p, "remarks") and p.remarks is not None else None
        )

        # Get stanza group names if available
        stanza_groups = None
        if hasattr(p, "stanzas") and p.stanzas is not None:
            stindiv = (
                p.stanzas.stindiv
            )  # StanzaParams is a dataclass, access attribute directly
            if stindiv is not None and len(stindiv) > 0:
                stanza_groups = (
                    stindiv["Group"].tolist() if "Group" in stindiv.columns else []
                )

        # Format for display: handle 9999 values, round to 3 decimals, mark cells with remarks
        formatted_df, no_data_mask, remarks_mask, stanza_mask = (
            format_dataframe_for_display(
                df, decimal_places=3, remarks_df=remarks_df, stanza_groups=stanza_groups
            )
        )
        styles = create_cell_styles(
            formatted_df, no_data_mask, remarks_mask, stanza_mask
        )

        return render.DataGrid(
            formatted_df, editable=True, filters=False, styles=styles, width="100%"
        )

    @output
    @render.ui
    def remarks_panel():
        """Show remarks if any exist in the model."""
        p = params.get()
        if p is None:
            return ui.div()  # Return empty div instead of None

        remarks_df = (
            p.remarks if hasattr(p, "remarks") and p.remarks is not None else None
        )
        if remarks_df is None:
            return ui.p(
                ui.tags.i(class_="bi bi-info-circle me-1"),
                "No remarks available. Remarks are imported from EwE database files.",
                class_="text-muted small mt-3",
            )

        # Build list of non-empty remarks
        remarks_list = []
        for idx, row in remarks_df.iterrows():
            group_name = str(row.get("Group", f"Row {idx}"))  # Ensure string
            for col in remarks_df.columns:
                if col != "Group":
                    remark = row.get(col, "")
                    if isinstance(remark, str) and remark.strip():
                        remarks_list.append(
                            {
                                "group": group_name,
                                "parameter": str(col),
                                "remark": str(remark.strip()),
                            }
                        )

        if not remarks_list:
            return ui.p(
                ui.tags.i(class_="bi bi-info-circle me-1"),
                "No remarks found in this model.",
                class_="text-muted small mt-3",
            )

        # Show remarks count
        return ui.p(
            ui.tags.i(class_="bi bi-chat-quote me-1"),
            f"Model has {len(remarks_list)} remarks.",
            class_="text-muted small mt-3",
        )

    @output
    @render.data_frame
    def diet_matrix_table():
        """Render editable diet matrix."""
        p = params.get()
        if p is None:
            return render.DataGrid(
                pd.DataFrame({"Message": ["Load or create a model first"]})
            )

        # Format for display: handle 9999 values and round to 3 decimals
        df = p.diet.copy()
        formatted_df, no_data_mask, remarks_mask, _ = format_dataframe_for_display(
            df, decimal_places=3
        )
        styles = create_cell_styles(formatted_df, no_data_mask, remarks_mask)

        return render.DataGrid(
            formatted_df, editable=True, filters=False, styles=styles
        )

    @output
    @render.data_frame
    def fisheries_table():
        """Render fisheries (landings/discards) table."""
        p = params.get()
        if p is None:
            return render.DataGrid(
                pd.DataFrame({"Message": ["Load or create a model first"]})
            )

        model_df = p.model

        # Find fleet columns by looking for columns that are also Type==3 groups
        fleet_groups = model_df[model_df["Type"] == 3]["Group"].tolist()

        # Also check for columns that look like fleet names (not standard params)
        standard_cols = {
            "Group",
            "Type",
            "Biomass",
            "PB",
            "QB",
            "EE",
            "ProdCons",
            "BioAcc",
            "Unassim",
            "DetInput",
            "Detritus",
        }
        potential_fleets = [
            c
            for c in model_df.columns
            if c not in standard_cols and not c.endswith(".disc")
        ]

        # Use fleet groups if available, otherwise use potential fleet columns
        if fleet_groups:
            fleet_names = fleet_groups
        elif potential_fleets:
            fleet_names = potential_fleets
        else:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No fleets defined in the model."]})
            )

        # Build a DataFrame with Group + fleet landings/discards
        living_groups = model_df[model_df["Type"] < 2]

        data = {"Group": living_groups["Group"].tolist()}
        for fleet in fleet_names:
            # Landings
            if fleet in model_df.columns:
                data[f"{fleet}_Land"] = living_groups.index.map(
                    lambda idx: (
                        model_df.at[idx, fleet]
                        if pd.notna(model_df.at[idx, fleet])
                        else None
                    )
                ).tolist()
            else:
                data[f"{fleet}_Land"] = [None] * len(living_groups)

            # Discards
            disc_col = f"{fleet}.disc"
            if disc_col in model_df.columns:
                data[f"{fleet}_Disc"] = living_groups.index.map(
                    lambda idx: (
                        model_df.at[idx, disc_col]
                        if pd.notna(model_df.at[idx, disc_col])
                        else None
                    )
                ).tolist()
            else:
                data[f"{fleet}_Disc"] = [None] * len(living_groups)

        df = pd.DataFrame(data)
        # Format for display: handle 9999 values and round to 3 decimals
        formatted_df, no_data_mask, remarks_mask, _ = format_dataframe_for_display(
            df, decimal_places=3
        )
        styles = create_cell_styles(formatted_df, no_data_mask, remarks_mask)

        return render.DataGrid(
            formatted_df, editable=True, filters=False, styles=styles
        )

    # === Multi-Stanza Functions ===

    @output
    @render.ui
    def stanza_status():
        """Show multi-stanza status."""
        p = params.get()
        if p is None:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "Load a model to see multi-stanza information.",
                class_="alert alert-info",
            )

        # Check if stanza data exists
        has_stanzas = (
            hasattr(p, "stanzas")
            and p.stanzas is not None
            and p.stanzas.n_stanza_groups > 0
        )

        if not has_stanzas:
            return ui.div(
                ui.tags.i(class_="bi bi-info-circle me-2"),
                "This model has no multi-stanza groups defined. "
                "Multi-stanza groups are used to model age-structured populations "
                "(e.g., juvenile and adult life stages of the same species).",
                class_="alert alert-info",
            )

        n_groups = int(p.stanzas.n_stanza_groups)
        n_stages = int(len(p.stanzas.stindiv)) if p.stanzas.stindiv is not None else 0

        return ui.div(
            ui.tags.i(class_="bi bi-check-circle-fill text-success me-2"),
            f"Model has {n_groups} multi-stanza group(s) with {n_stages} total life stages.",
            class_="alert alert-success",
        )

    @output
    @render.data_frame
    def stanza_groups_table():
        """Render stanza groups configuration table."""
        p = params.get()
        if p is None:
            return render.DataGrid(pd.DataFrame({"Message": ["Load a model first"]}))

        has_stanzas = (
            hasattr(p, "stanzas")
            and p.stanzas is not None
            and p.stanzas.stgroups is not None
            and len(p.stanzas.stgroups) > 0
        )

        if not has_stanzas:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No multi-stanza groups in this model"]})
            )

        df = p.stanzas.stgroups.copy()

        # Format for display
        formatted_df, no_data_mask, _, _ = format_dataframe_for_display(
            df, decimal_places=3
        )
        styles = create_cell_styles(formatted_df, no_data_mask, None)

        return render.DataGrid(formatted_df, styles=styles)

    @output
    @render.data_frame
    def stanza_indiv_table():
        """Render individual stanza life stages table."""
        p = params.get()
        if p is None:
            return render.DataGrid(pd.DataFrame({"Message": ["Load a model first"]}))

        has_stanzas = (
            hasattr(p, "stanzas")
            and p.stanzas is not None
            and p.stanzas.stindiv is not None
            and len(p.stanzas.stindiv) > 0
        )

        if not has_stanzas:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No multi-stanza life stages in this model"]})
            )

        df = p.stanzas.stindiv.copy()

        # Reorder columns for better display
        preferred_order = [
            "StanzaGroup",
            "Group",
            "StanzaNum",
            "First",
            "Last",
            "Z",
            "Leading",
        ]
        cols = [c for c in preferred_order if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]

        # Format for display
        formatted_df, no_data_mask, _, _ = format_dataframe_for_display(
            df, decimal_places=3
        )
        styles = create_cell_styles(formatted_df, no_data_mask, None)

        return render.DataGrid(formatted_df, styles=styles)

    # Track cell edits from DataGrids and update params
    @reactive.effect
    def _handle_model_params_edit():
        """Handle edits to model parameters table."""
        edit = input.model_params_table_cell_edit()
        if edit is None:
            return

        p = params.get()
        if p is None:
            return

        row = edit["row"]
        col_name = edit["column"]
        new_value = edit["value"]
        group_name = (
            p.model.loc[row, "Group"] if "Group" in p.model.columns else f"Row {row}"
        )

        # Update the params
        if col_name in p.model.columns and col_name != "Group":
            try:
                # Convert value
                numeric_value = _convert_input_to_numeric(new_value)

                # Validate based on column type
                is_valid = True
                error_msg = None

                if col_name == "Biomass" and not np.isnan(numeric_value):
                    is_valid, error_msg = validate_biomass(numeric_value, group_name)
                elif col_name == "PB" and not np.isnan(numeric_value):
                    group_type = (
                        p.model.loc[row, "Type"] if "Type" in p.model.columns else None
                    )
                    is_valid, error_msg = validate_pb(
                        numeric_value, group_name, group_type
                    )
                elif col_name == "EE" and not np.isnan(numeric_value):
                    is_valid, error_msg = validate_ee(numeric_value, group_name)

                if is_valid:
                    p.model.loc[row, col_name] = numeric_value
                    ui.notification_show(
                        f"Updated {col_name} for {group_name}",
                        type="message",
                        duration=2,
                    )
                else:
                    ui.notification_show(
                        f"Invalid value for {col_name}: {error_msg}",
                        type="warning",
                        duration=5,
                    )

            except (ValueError, TypeError):
                ui.notification_show(
                    f"Invalid numeric value for {col_name}: '{new_value}'",
                    type="error",
                    duration=4,
                )

    @reactive.effect
    def _handle_diet_matrix_edit():
        """Handle edits to diet matrix table."""
        edit = input.diet_matrix_table_cell_edit()
        if edit is None:
            return

        p = params.get()
        if p is None:
            return

        row = edit["row"]
        col_name = edit["column"]
        new_value = edit["value"]
        prey_name = (
            p.diet.loc[row, "Group"] if "Group" in p.diet.columns else f"Row {row}"
        )

        # Update the diet matrix
        if col_name in p.diet.columns and col_name != "Group":
            try:
                # Convert value
                numeric_value = float(new_value) if new_value else 0.0

                # Validate diet proportion (0-1)
                if numeric_value < 0:
                    ui.notification_show(
                        f"Diet proportion cannot be negative: {numeric_value:.3f}",
                        type="error",
                        duration=4,
                    )
                    return

                if numeric_value > 1:
                    ui.notification_show(
                        f"Diet proportion cannot exceed 1.0: {numeric_value:.3f}",
                        type="warning",
                        duration=4,
                    )
                    return

                p.diet.loc[row, col_name] = numeric_value
                ui.notification_show(
                    f"Updated diet: {prey_name} → {col_name}",
                    type="message",
                    duration=2,
                )

            except (ValueError, TypeError):
                ui.notification_show(
                    f"Invalid numeric value for diet: '{new_value}'",
                    type="error",
                    duration=4,
                )

    @reactive.effect
    @reactive.event(input.btn_balance)
    def _balance_model():
        """Balance the Ecopath model."""
        p = params.get()
        if p is None:
            ui.notification_show("Create parameters first", type="warning")
            return

        try:
            # Set defaults for missing values
            if "BioAcc" not in p.model.columns:
                p.model["BioAcc"] = DEFAULTS.ba_consumers
            else:
                p.model["BioAcc"] = p.model["BioAcc"].fillna(DEFAULTS.ba_consumers)

            if "Unassim" not in p.model.columns:
                p.model["Unassim"] = DEFAULTS.unassim_consumers
            else:
                p.model["Unassim"] = p.model["Unassim"].fillna(
                    DEFAULTS.unassim_consumers
                )

            if "DetInput" not in p.model.columns:
                p.model["DetInput"] = 0.0
            else:
                p.model["DetInput"] = p.model["DetInput"].fillna(0.0)

            # For living groups, set a default Unassim if needed
            living_mask = p.model["Type"] < 2
            p.model.loc[living_mask & (p.model["Unassim"] == 0), "Unassim"] = (
                DEFAULTS.unassim_consumers
            )

            # Set detritus fate columns if missing
            det_groups = p.model[p.model["Type"] == 2]["Group"].tolist()
            if det_groups:
                for det in det_groups:
                    if det not in p.model.columns:
                        # Add the detritus fate column
                        p.model[det] = np.nan

                    # Set default detritus fate for living/detritus groups
                    n_det = len(det_groups)
                    for idx in p.model.index:
                        gtype = p.model.loc[idx, "Type"]
                        if gtype < 3:  # Not a fleet
                            if pd.isna(p.model.loc[idx, det]):
                                p.model.loc[idx, det] = 1.0 / n_det
                        else:
                            p.model.loc[idx, det] = np.nan

            # Validate model parameters before balancing
            is_valid, validation_errors = validate_model_parameters(
                p.model,
                check_groups=True,
                check_biomass=True,
                check_pb=True,
                check_ee=False,  # EE is calculated, not input
            )

            if not is_valid:
                # Show first error in notification
                error_summary = (
                    validation_errors[0]
                    if len(validation_errors) == 1
                    else f"{len(validation_errors)} validation errors found. First error:\n{validation_errors[0]}"
                )
                ui.notification_show(error_summary, type="error", duration=10)
                return

            # Balance the model
            model = rpath(p, eco_name=input.eco_name())
            balanced_model.set(model)
            model_data.set(model)

            ui.notification_show("Model balanced successfully!", type="message")

        except Exception as e:
            import traceback

            traceback.print_exc()
            ui.notification_show(f"Error balancing model: {str(e)}", type="error")

    @output
    @render.ui
    def balance_status():
        """Show balance status."""
        model = balanced_model.get()
        if model is None:
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-circle me-2"),
                "Model not yet balanced. Enter parameters and click 'Balance Model'.",
                class_="alert alert-info",
            )

        # Check for issues - convert to int for display
        ee_issues = int(np.sum((model.EE > 1) | (model.EE < 0)))

        if ee_issues > 0:
            return ui.div(
                ui.tags.i(class_="bi bi-exclamation-triangle me-2"),
                f"Model balanced with warnings: {ee_issues} groups have EE outside [0,1]",
                class_="alert alert-warning",
            )

        # Use model name or default
        model_name = model.eco_name if model.eco_name else "Ecopath"
        return ui.div(
            ui.tags.i(class_="bi bi-check-circle me-2"),
            f"Model '{model_name}' balanced successfully!",
            class_="alert alert-success",
        )

    @output
    @render.data_frame
    def model_results_table():
        """Display balanced model results with formatting."""
        model = balanced_model.get()
        if model is None:
            return render.DataGrid(
                pd.DataFrame({"Message": ["Balance the model to see results"]})
            )

        # Get the summary DataFrame
        df = model.summary()

        # Get stanza group names if available (from params)
        p = params.get()
        stanza_groups = None
        if p is not None and hasattr(p, "stanzas") and p.stanzas is not None:
            stindiv = (
                p.stanzas.stindiv
            )  # StanzaParams is a dataclass, access attribute directly
            if stindiv is not None and len(stindiv) > 0:
                stanza_groups = (
                    stindiv["Group"].tolist() if "Group" in stindiv.columns else []
                )

        # Format for display: handle 9999 values, round decimals, convert Type, mark stanza groups
        formatted_df, no_data_mask, _, stanza_mask = format_dataframe_for_display(
            df, decimal_places=3, stanza_groups=stanza_groups
        )

        # Create styles - check mask values carefully using bool() conversion
        styles = []

        # Style no-data cells and stanza cells
        for row_idx in range(len(formatted_df)):
            for col_idx, col in enumerate(formatted_df.columns):
                is_no_data = (
                    bool(no_data_mask.iloc[row_idx][col])
                    if col in no_data_mask.columns
                    else False
                )
                is_stanza = (
                    bool(stanza_mask.iloc[row_idx][col])
                    if (stanza_mask is not None and col in stanza_mask.columns)
                    else False
                )

                if is_no_data:
                    styles.append(
                        {
                            "location": "body",
                            "rows": row_idx,
                            "cols": col_idx,
                            "style": NO_DATA_STYLE,
                        }
                    )
                elif is_stanza:
                    styles.append(
                        {
                            "location": "body",
                            "rows": row_idx,
                            "cols": col_idx,
                            "style": STANZA_STYLE,
                        }
                    )

        # Add special styling for calculated columns (EE, GE, TL)
        calculated_cols = ["EE", "GE", "TL"]
        col_positions = {c: i for i, c in enumerate(formatted_df.columns)}
        for col in calculated_cols:
            if col in formatted_df.columns:
                col_idx = col_positions[col]
                for row_idx in range(len(formatted_df)):
                    is_no_data = (
                        bool(no_data_mask.iloc[row_idx][col])
                        if col in no_data_mask.columns
                        else False
                    )
                    is_stanza = (
                        bool(stanza_mask.iloc[row_idx][col])
                        if (stanza_mask is not None and col in stanza_mask.columns)
                        else False
                    )
                    if not is_no_data and not is_stanza:
                        styles.append(
                            {
                                "location": "body",
                                "rows": row_idx,
                                "cols": col_idx,
                                "style": {
                                    "background-color": "#f0fff0"
                                },  # Light green for calculated values
                            }
                        )

        return render.DataGrid(formatted_df, filters=False, styles=styles, width="100%")

    @output
    @render.ui
    def diagnostics_output():
        """Display model diagnostics."""
        model = balanced_model.get()
        if model is None:
            return ui.p("Balance the model to see diagnostics.", class_="text-muted")

        # Calculate diagnostics - convert numpy values to Python types
        total_biomass = float(np.sum(model.Biomass[: model.NUM_LIVING]))
        total_production = float(
            np.sum(model.Biomass[: model.NUM_LIVING] * model.PB[: model.NUM_LIVING])
        )

        return ui.div(
            ui.layout_columns(
                ui.value_box(
                    "Total Groups",
                    str(model.NUM_GROUPS),
                    showcase=ui.tags.i(class_="bi bi-diagram-3"),
                ),
                ui.value_box(
                    "Living Groups",
                    str(model.NUM_LIVING),
                    showcase=ui.tags.i(class_="bi bi-heart"),
                ),
                ui.value_box(
                    "Total Biomass",
                    f"{total_biomass:.2f}",
                    showcase=ui.tags.i(class_="bi bi-box"),
                ),
                ui.value_box(
                    "Total Production",
                    f"{total_production:.2f}",
                    showcase=ui.tags.i(class_="bi bi-arrow-up-circle"),
                ),
                col_widths=[3, 3, 3, 3],
            ),
        )

    @output
    @render.plot
    def trophic_level_plot():
        """Plot trophic levels."""
        import matplotlib.pyplot as plt

        model = balanced_model.get()
        if model is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No model data", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(PLOTS.default_width, PLOTS.default_height))

        # Get group names safely
        all_groups = _get_groups_from_model(model)
        num_living_dead = (
            model.NUM_LIVING + model.NUM_DEAD
            if is_balanced_model(model)
            else len(all_groups)
        )
        groups = all_groups[:num_living_dead]
        tl = model.TL[:num_living_dead]

        colors = [
            (
                "#2ecc71"
                if t == 1
                else (
                    "#3498db"
                    if t < THRESHOLDS.type_threshold_consumer_toppred
                    else "#e74c3c"
                )
            )
            for t in tl
        ]

        ax.barh(groups, tl, color=colors)
        ax.set_xlabel("Trophic Level")
        ax.set_title("Trophic Levels by Group")
        ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig

    @output
    @render.plot
    def ee_plot():
        """Plot ecotrophic efficiency."""
        import matplotlib.pyplot as plt

        model = balanced_model.get()
        if model is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No model data", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(PLOTS.default_width, PLOTS.default_height))

        # Get group names safely
        all_groups = _get_groups_from_model(model)
        num_living_dead = (
            model.NUM_LIVING + model.NUM_DEAD
            if is_balanced_model(model)
            else len(all_groups)
        )
        groups = all_groups[:num_living_dead]
        ee = model.EE[:num_living_dead]

        colors = ["#2ecc71" if 0 <= e <= 1 else "#e74c3c" for e in ee]

        ax.barh(groups, ee, color=colors)
        ax.set_xlabel("Ecotrophic Efficiency")
        ax.set_title("Ecotrophic Efficiency by Group")
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="EE=1")
        ax.set_xlim(0, max(1.1, max(ee) * 1.1))

        plt.tight_layout()
        return fig

    @render.download(filename="pypath_params.csv")
    def download_params():
        """Download parameters as CSV."""
        p = params.get()
        if p is not None:
            return p.model.to_csv(index=False)
        return ""
