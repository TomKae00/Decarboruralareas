# Define the configuration file
configfile: "config.yaml"

# The "all" rule is used to specify the final target files of the workflow.
rule all:
    input:
        "all_close_potentials.gpkg"

# Rule to process the data using the data_Energieportal2.0.py script
rule fetch_process_data:
    output:
        gpkg="all_close_potentials.gpkg"
    script:
        "data_Energieportal2.0.py"
