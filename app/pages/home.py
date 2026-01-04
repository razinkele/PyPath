"""Home page module."""

from shiny import Inputs, Outputs, Session, reactive, render, ui
from pypath.core.params import create_rpath_params, StanzaParams
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
import pandas as pd
import numpy as np
import warnings

# Import centralized configuration
try:
    from app.config import DEFAULTS
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    app_dir = Path(__file__).parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    from config import DEFAULTS


def home_ui():
    """Home page UI."""
    return ui.page_fluid(
        ui.div(
            # Hero section
            ui.div(
                ui.p(
                    "A Python implementation of Ecopath with Ecosim and Ecospace for ecosystem modeling",
                    class_="lead"
                ),
                ui.tags.hr(class_="my-4"),
                ui.p(
                    "PyPath provides tools for building mass-balance food web models (Ecopath), "
                    "running dynamic ecosystem simulations (Ecosim), and spatial modeling with "
                    "irregular grids (Ecospace). This dashboard allows you to create models, "
                    "run simulations, and visualize results interactively."
                ),
                ui.div(
                    ui.input_action_button(
                        "btn_start_ecopath",
                        "Start with Ecopath →",
                        class_="btn-primary btn-lg me-2"
                    ),
                    ui.input_action_button(
                        "btn_load_example",
                        "Load Example Model",
                        class_="btn-outline-secondary btn-lg"
                    ),
                    class_="mt-4"
                ),
                class_="p-5 mb-4 bg-light rounded-3"
            ),

            # What's New section
            ui.div(
                ui.h2(
                    ui.tags.i(class_="bi bi-star-fill text-warning me-2"),
                    "What's New in PyPath",
                    class_="mb-3"
                ),
                ui.card(
                    ui.card_body(
                        ui.layout_columns(
                            ui.div(
                                ui.h5(
                                    ui.tags.i(class_="bi bi-geo-alt text-success me-2"),
                                    "Irregular Grid Support",
                                    class_="mb-2"
                                ),
                                ui.p(
                                    "Upload custom polygon geometries for realistic spatial modeling! "
                                    "ECOSPACE now supports shapefiles, GeoJSON, and GeoPackage formats.",
                                    class_="mb-1"
                                ),
                                ui.p(
                                    ui.tags.strong("Try it: "),
                                    "Navigate to the Ecospace page and upload ",
                                    ui.tags.code("examples/coastal_grid_example.geojson"),
                                    class_="text-muted small"
                                ),
                            ),
                            ui.div(
                                ui.h5(
                                    ui.tags.i(class_="bi bi-shuffle text-primary me-2"),
                                    "Diet Rewiring",
                                    class_="mb-2"
                                ),
                                ui.p(
                                    "Advanced prey switching behavior! Predators now adapt their diet "
                                    "based on prey availability with configurable switching power.",
                                    class_="mb-1"
                                ),
                                ui.p(
                                    ui.tags.strong("Explore: "),
                                    "Check out the Diet Rewiring Demo page for interactive examples.",
                                    class_="text-muted small"
                                ),
                            ),
                            ui.div(
                                ui.h5(
                                    ui.tags.i(class_="bi bi-lightning text-warning me-2"),
                                    "Enhanced Ecosim",
                                    class_="mb-2"
                                ),
                                ui.p(
                                    "Environmental forcing, optimization tools, and improved "
                                    "multi-stanza support for age-structured populations.",
                                    class_="mb-1"
                                ),
                                ui.p(
                                    ui.tags.strong("Learn more: "),
                                    "See the Advanced Features demos for detailed examples.",
                                    class_="text-muted small"
                                ),
                            ),
                            col_widths=[4, 4, 4]
                        )
                    ),
                    class_="border-success"
                ),
                class_="mb-4"
            ),

            # Feature cards - Row 1
            ui.h2("Features", class_="mb-4"),
            ui.layout_columns(
                # Data Import card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-cloud-download me-2"),
                        "Data Import"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Connect to EcoBase database (350+ models)"),
                            ui.tags.li("Search and download published models"),
                            ui.tags.li("Import EwE database files (.ewemdb, .eweaccdb)"),
                            ui.tags.li("Automatic format conversion"),
                            ui.tags.li("Pre-configured ecosystem models"),
                        ),
                        ui.input_action_button(
                            "btn_goto_import",
                            "Import Model",
                            class_="btn-success mt-3"
                        )
                    ),
                ),
                
                # Ecopath card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-diagram-3 me-2"),
                        "Ecopath Mass Balance"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Define functional groups and food web"),
                            ui.tags.li("Set biomass, P/B, Q/B ratios"),
                            ui.tags.li("Automatic mass-balance calculations"),
                            ui.tags.li("Trophic level computation"),
                            ui.tags.li("Multi-stanza age structure"),
                        ),
                        ui.input_action_button(
                            "btn_goto_ecopath",
                            "Create Ecopath Model",
                            class_="btn-primary mt-3"
                        )
                    ),
                ),
                
                # Ecosim card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-graph-up me-2"),
                        "Ecosim Simulation"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Dynamic ecosystem simulations"),
                            ui.tags.li("Foraging arena functional response"),
                            ui.tags.li("Fishing scenarios and forcing"),
                            ui.tags.li("Environmental forcing effects"),
                            ui.tags.li("Multiple integration methods"),
                        ),
                        ui.input_action_button(
                            "btn_goto_ecosim",
                            "Run Simulation",
                            class_="btn-primary mt-3"
                        )
                    ),
                ),
                col_widths=[4, 4, 4]
            ),
            
            # Feature cards - Row 2
            ui.layout_columns(
                # ECOSPACE card (NEW!)
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-geo-alt me-2"),
                        "ECOSPACE Spatial Modeling",
                        ui.tags.span(
                            "NEW",
                            class_="badge bg-success ms-2"
                        )
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Irregular grid support (GeoJSON, Shapefile)"),
                            ui.tags.li("Realistic spatial geometries"),
                            ui.tags.li("Habitat preference mapping"),
                            ui.tags.li("Dispersal and movement dynamics"),
                            ui.tags.li("Spatial fishing effort allocation"),
                        ),
                        ui.input_action_button(
                            "btn_goto_ecospace",
                            "Explore Spatial",
                            class_="btn-success mt-3"
                        )
                    ),
                ),

                # Analysis card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-diagram-2 me-2"),
                        "Network Analysis"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Food web topology metrics"),
                            ui.tags.li("Ecosystem health indicators"),
                            ui.tags.li("Trophic impact analysis (MTI)"),
                            ui.tags.li("Keystone species identification"),
                            ui.tags.li("Lindeman spine diagrams"),
                        ),
                        ui.input_action_button(
                            "btn_goto_analysis",
                            "Run Analysis",
                            class_="btn-info mt-3"
                        )
                    ),
                ),

                # Results card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-bar-chart me-2"),
                        "Results & Visualization"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Interactive time series plots"),
                            ui.tags.li("Food web diagrams"),
                            ui.tags.li("Biomass and catch trajectories"),
                            ui.tags.li("Export results to CSV/Excel"),
                            ui.tags.li("Comparative scenario analysis"),
                        ),
                        ui.input_action_button(
                            "btn_goto_results",
                            "View Results",
                            class_="btn-primary mt-3"
                        )
                    ),
                ),
                col_widths=[4, 4, 4],
                class_="mt-4"
            ),

            # Feature cards - Row 3
            ui.layout_columns(
                # Advanced Features card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-gear-wide-connected me-2"),
                        "Advanced Features"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Diet rewiring (prey switching)"),
                            ui.tags.li("Environmental forcing functions"),
                            ui.tags.li("Multi-stanza age structure"),
                            ui.tags.li("Optimization and sensitivity analysis"),
                            ui.tags.li("Custom fishing scenarios"),
                        ),
                        ui.p(
                            "Access advanced features via dedicated demo pages.",
                            class_="text-muted small mt-2"
                        )
                    ),
                ),

                # About card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-info-circle me-2"),
                        "About PyPath"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("Python port of R Rpath package"),
                            ui.tags.li("NOAA/EwE ecosystem modeling"),
                            ui.tags.li("Open source (MIT License)"),
                            ui.tags.li("GitHub repository"),
                            ui.tags.li("Active development"),
                        ),
                        ui.input_action_button(
                            "btn_goto_about",
                            "Learn More",
                            class_="btn-secondary mt-3"
                        )
                    ),
                ),

                # Documentation card
                ui.card(
                    ui.card_header(
                        ui.tags.i(class_="bi bi-file-text me-2"),
                        "Documentation"
                    ),
                    ui.card_body(
                        ui.tags.ul(
                            ui.tags.li("User guides and tutorials"),
                            ui.tags.li("API reference documentation"),
                            ui.tags.li("Example models and scripts"),
                            ui.tags.li("Irregular grid guide (NEW)"),
                            ui.tags.li("Video tutorials (coming soon)"),
                        ),
                        ui.p(
                            "See the 'examples/' folder for guides and sample data.",
                            class_="text-muted small mt-2"
                        )
                    ),
                ),
                col_widths=[4, 4, 4],
                class_="mt-4"
            ),
            
            # Quick start section
            ui.h2("Quick Start", class_="mt-5 mb-4"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Step 1: Define Groups"),
                    ui.card_body(
                        ui.p(
                            "Start by defining your ecosystem's functional groups - "
                            "producers, consumers, detritus, and fishing fleets."
                        ),
                        ui.tags.code("create_rpath_params(groups, types)")
                    ),
                ),
                ui.card(
                    ui.card_header("Step 2: Set Parameters"),
                    ui.card_body(
                        ui.p(
                            "Enter biomass, P/B, Q/B ratios, diet composition, "
                            "and fishery catches for each group."
                        ),
                        ui.tags.code("params.model['Biomass'] = ...")
                    ),
                ),
                ui.card(
                    ui.card_header("Step 3: Balance Model"),
                    ui.card_body(
                        ui.p(
                            "Run the mass-balance calculations to solve for "
                            "missing parameters and validate the model."
                        ),
                        ui.tags.code("model = rpath(params)")
                    ),
                ),
                ui.card(
                    ui.card_header("Step 4: Run Simulation"),
                    ui.card_body(
                        ui.p(
                            "Create a scenario and run dynamic Ecosim simulations "
                            "to explore ecosystem dynamics."
                        ),
                        ui.tags.code("rsim_run(scenario)")
                    ),
                ),
                ui.card(
                    ui.card_header(
                        "Step 5: Add Spatial Dynamics",
                        ui.tags.span("Optional", class_="badge bg-info ms-2", style="font-size: 0.7em;")
                    ),
                    ui.card_body(
                        ui.p(
                            "Upload spatial grids and run Ecospace simulations "
                            "with habitat preferences and dispersal."
                        ),
                        ui.tags.code("rsim_run_spatial(ecospace)")
                    ),
                ),
                col_widths=[2, 2, 2, 3, 3]
            ),
            
            class_="container py-4"
        )
    )


def home_server(input: Inputs, output: Outputs, session: Session, model_data: reactive.Value):
    """Home page server logic."""
    
    @reactive.effect
    @reactive.event(input.btn_goto_import)
    def _goto_import():
        ui.update_navs("main_navbar", selected="Data Import")
    
    @reactive.effect
    @reactive.event(input.btn_start_ecopath, input.btn_goto_ecopath)
    def _goto_ecopath():
        ui.update_navs("main_navbar", selected="Ecopath Model")
    
    @reactive.effect
    @reactive.event(input.btn_goto_ecosim)
    def _goto_ecosim():
        ui.update_navs("main_navbar", selected="Ecosim Simulation")

    @reactive.effect
    @reactive.event(input.btn_goto_ecospace)
    def _goto_ecospace():
        ui.update_navs("main_navbar", selected="Ecospace")

    @reactive.effect
    @reactive.event(input.btn_goto_analysis)
    def _goto_analysis():
        ui.update_navs("main_navbar", selected="Analysis")
    
    @reactive.effect
    @reactive.event(input.btn_goto_results)
    def _goto_results():
        ui.update_navs("main_navbar", selected="Results")
    
    @reactive.effect
    @reactive.event(input.btn_goto_about)
    def _goto_about():
        ui.update_navs("main_navbar", selected="About")
    
    @reactive.effect
    @reactive.event(input.btn_load_example)
    def _load_example_model():
        """Load an example marine ecosystem model."""
        try:
            # Create example marine ecosystem model
            groups = [
                'Seals',           # Top predator
                'JuvRoundfish1',   # Juvenile fish
                'AduRoundfish1',   # Adult fish
                'OtherGroundfish', # Groundfish
                'Foragefish1',     # Forage fish
                'Megabenthos',     # Large benthos
                'Zooplankton',     # Zooplankton
                'Phytoplankton',   # Primary producer
                'Detritus',        # Detritus
                'Trawlers',        # Fishing fleet
            ]
            
            types = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3]  # consumer, producer, detritus, fleet

            # Define stanza groups - make JuvRoundfish1 and AduRoundfish1 into multi-stanza group
            stgroups_list = [None, 'Roundfish', 'Roundfish', None, None, None, None, None, None, None]

            params = create_rpath_params(groups, types, stgroups=stgroups_list)

            # Initialize remarks DataFrame (same structure as model)
            remarks_cols = ['Group'] + [col for col in params.model.columns if col != 'Group']
            params.remarks = pd.DataFrame({col: [''] * len(groups) for col in remarks_cols})

            # Populate multi-stanza parameters for the Roundfish stanza group
            if params.stanzas.n_stanza_groups > 0:
                # Set stanza group parameters (von Bertalanffy growth, maturity weight, etc.)
                params.stanzas.stgroups.loc[0, 'VBGF_Ksp'] = 0.4  # von Bertalanffy growth rate
                params.stanzas.stgroups.loc[0, 'VBGF_d'] = 0.66667  # VBGF allometric parameter
                params.stanzas.stgroups.loc[0, 'Wmat'] = 50.0  # Maturity weight (g)
                params.stanzas.stgroups.loc[0, 'BAB'] = 0.0  # Biomass accumulation rate
                params.stanzas.stgroups.loc[0, 'RecPower'] = 1.0  # Recruitment power

                # Set individual stanza parameters (age ranges and mortality)
                # Juvenile: 0-24 months, Z=0.8
                juv_idx = params.stanzas.stindiv[params.stanzas.stindiv['Group'] == 'JuvRoundfish1'].index[0]
                params.stanzas.stindiv.loc[juv_idx, 'StanzaNum'] = 1
                params.stanzas.stindiv.loc[juv_idx, 'First'] = 0  # Start at birth
                params.stanzas.stindiv.loc[juv_idx, 'Last'] = 24  # End at 24 months
                params.stanzas.stindiv.loc[juv_idx, 'Z'] = 0.8  # Total mortality
                params.stanzas.stindiv.loc[juv_idx, 'Leading'] = 0  # Not leading stanza

                # Adult: 24+ months, Z=0.35, leading stanza
                adu_idx = params.stanzas.stindiv[params.stanzas.stindiv['Group'] == 'AduRoundfish1'].index[0]
                params.stanzas.stindiv.loc[adu_idx, 'StanzaNum'] = 2
                params.stanzas.stindiv.loc[adu_idx, 'First'] = 24  # Start at 24 months
                params.stanzas.stindiv.loc[adu_idx, 'Last'] = DEFAULTS.default_months  # End at default months (10 years)
                params.stanzas.stindiv.loc[adu_idx, 'Z'] = 0.35  # Total mortality
                params.stanzas.stindiv.loc[adu_idx, 'Leading'] = 1  # Leading stanza (plus group)

                # Add StanzaGroup column to stindiv for clarity
                params.stanzas.stindiv['StanzaGroup'] = 'Roundfish'

            # Add helpful remarks/tooltips for key parameters
            # Find group indices
            seal_idx = groups.index('Seals')
            juv_idx = groups.index('JuvRoundfish1')
            adu_idx = groups.index('AduRoundfish1')
            phyto_idx = groups.index('Phytoplankton')
            det_idx = groups.index('Detritus')

            params.remarks.loc[seal_idx, 'Biomass'] = 'Low biomass typical for top predator'
            params.remarks.loc[seal_idx, 'EE'] = 'Low EE - top predator, little predation'
            params.remarks.loc[juv_idx, 'Biomass'] = 'Part of Roundfish multi-stanza group'
            params.remarks.loc[juv_idx, 'EE'] = 'High EE due to predation and growth to adult stage'
            params.remarks.loc[adu_idx, 'Biomass'] = 'Leading stanza of Roundfish group'
            params.remarks.loc[adu_idx, 'PB'] = 'Lower P/B for adult stage'
            params.remarks.loc[phyto_idx, 'Type'] = 'Primary producer (Type=1)'
            params.remarks.loc[phyto_idx, 'Biomass'] = 'Autotroph - no QB value needed'
            params.remarks.loc[det_idx, 'Type'] = 'Detritus pool (Type=2)'
            params.remarks.loc[det_idx, 'DetInput'] = 'Import of detritus from outside system'

            # Set model parameters
            biomass_data = {
                'Seals': 0.025, 'JuvRoundfish1': 0.1304, 'AduRoundfish1': 1.39,
                'OtherGroundfish': 7.4, 'Foragefish1': 5.1, 'Megabenthos': 19.765,
                'Zooplankton': 23.0, 'Phytoplankton': 10.0, 'Detritus': 500.0,
            }
            pb_data = {
                'Seals': 0.15, 'JuvRoundfish1': 1.5, 'AduRoundfish1': 0.35,
                'OtherGroundfish': 0.4, 'Foragefish1': 0.7, 'Megabenthos': 0.2,
                'Zooplankton': 30.0, 'Phytoplankton': 200.0,
            }
            qb_data = {
                'Seals': 25.0, 'JuvRoundfish1': 10.0, 'AduRoundfish1': 3.5,
                'OtherGroundfish': 2.0, 'Foragefish1': 5.0, 'Megabenthos': 1.5,
                'Zooplankton': 100.0,
            }
            ee_data = {
                'Seals': 0.1, 'JuvRoundfish1': 0.9, 'AduRoundfish1': 0.8,
                'OtherGroundfish': 0.8, 'Foragefish1': 0.9, 'Megabenthos': 0.6,
                'Zooplankton': 0.9, 'Phytoplankton': 0.8,
            }
            
            for i, group in enumerate(groups):
                if group in biomass_data:
                    params.model.loc[i, 'Biomass'] = biomass_data[group]
                if group in pb_data:
                    params.model.loc[i, 'PB'] = pb_data[group]
                if group in qb_data:
                    params.model.loc[i, 'QB'] = qb_data[group]
                if group in ee_data:
                    params.model.loc[i, 'EE'] = ee_data[group]
            
            # Set defaults - consumers get config value, producers/detritus get 0.0
            params.model['BioAcc'] = 0.0
            params.model.loc[params.model['Type'] == 0, 'Unassim'] = DEFAULTS.unassim_consumers  # Consumers
            params.model.loc[params.model['Type'] == 1, 'Unassim'] = DEFAULTS.unassim_producers  # Producers
            params.model.loc[params.model['Type'] == 2, 'Unassim'] = DEFAULTS.unassim_producers  # Detritus
            params.model.loc[params.model['Type'] == 3, 'BioAcc'] = float('nan')
            params.model.loc[params.model['Type'] == 3, 'Unassim'] = float('nan')
            params.model['Detritus'] = 1.0
            params.model.loc[params.model['Type'] == 3, 'Detritus'] = float('nan')
            
            # Set diet matrix
            prey_names = list(params.diet['Group'])
            n_prey = len(prey_names)
            
            def make_diet(diet_dict):
                diet = [0.0] * n_prey
                for prey, prop in diet_dict.items():
                    if prey in prey_names:
                        diet[prey_names.index(prey)] = prop
                return diet
            
            params.diet['Seals'] = make_diet({'Foragefish1': 0.4, 'AduRoundfish1': 0.3, 'OtherGroundfish': 0.3})
            params.diet['JuvRoundfish1'] = make_diet({'Zooplankton': 0.9, 'Megabenthos': 0.1})
            params.diet['AduRoundfish1'] = make_diet({'Foragefish1': 0.5, 'Zooplankton': 0.3, 'Megabenthos': 0.2})
            params.diet['OtherGroundfish'] = make_diet({'Foragefish1': 0.4, 'Megabenthos': 0.3, 'Zooplankton': 0.3})
            params.diet['Foragefish1'] = make_diet({'Zooplankton': 1.0})
            params.diet['Megabenthos'] = make_diet({'Phytoplankton': 0.3, 'Detritus': 0.7})
            params.diet['Zooplankton'] = make_diet({'Phytoplankton': 0.9, 'Detritus': 0.1})
            params.diet['Phytoplankton'] = [0.0] * n_prey
            
            # Set fishing catches
            catches = {'AduRoundfish1': 0.145, 'OtherGroundfish': 0.38, 'Megabenthos': 0.19,
                       'Seals': 0.002, 'JuvRoundfish1': 0.003, 'Foragefish1': 0.1}
            for group, catch in catches.items():
                if group in groups:
                    idx = groups.index(group)
                    params.model.loc[idx, 'Trawlers'] = catch
            
            # Balance the model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = rpath(params)
            
            # Store the params (not the balanced model) in shared state
            # The ecopath page needs the editable parameters, not the balanced results
            model_data.set(params)
            
            ui.notification_show(
                "Example model loaded with multi-stanza groups (Roundfish) and sample remarks! Navigate to Ecopath tab or explore Advanced Features.",
                type="message",
                duration=7
            )
            
            # Navigate to Ecopath tab
            ui.update_navs("main_navbar", selected="Ecopath Model")
            
        except Exception as e:
            ui.notification_show(f"Error loading example model: {str(e)}", type="error")
