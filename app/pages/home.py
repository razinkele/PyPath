"""Home page module."""

from shiny import Inputs, Outputs, Session, reactive, render, ui


def home_ui():
    """Home page UI."""
    return ui.page_fluid(
        ui.div(
            # Hero section
            ui.div(
                ui.p(
                    "A Python implementation of Ecopath with Ecosim for ecosystem modeling",
                    class_="lead"
                ),
                ui.tags.hr(class_="my-4"),
                ui.p(
                    "PyPath provides tools for building mass-balance food web models (Ecopath) "
                    "and running dynamic ecosystem simulations (Ecosim). This dashboard allows "
                    "you to create models, run simulations, and visualize results interactively."
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
                col_widths=[3, 3, 3, 3]
            ),
            
            class_="container py-4"
        )
    )


def home_server(input: Inputs, output: Outputs, session: Session):
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
