import os

configfile: "config.yaml"

#RDIR = os.path.join(config['results_dir'], config["run"])

rule all:
    input:
        expand("output/results_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°_return:{return_temp}_elecprice_{electricity_price}",
            **config['scenario'])
# The "all" rule is used to specify the final target files of the workflow.
#rule all_weather_data:
#    input:
#        expand('results/temperature_series_{selected_system_id}_{year_of_interest}.csv', **config['scenario'])

# on this data is the script built. needed for all the other scripts
rule data_energieportal:
    output:
        selected_system='output/selected_system_{selected_system_id}.gpkg',
        all_close_potentials='output/all_close_potentials_{selected_system_id}.gpkg',
        max_potentials='output/max_potentials_{selected_system_id}.gpkg'
    params:
        system_id='{selected_system_id}'
    script:
        'scripts/data_Energieportal3.py'

# defining the rules to obtain the results of the weather_data
rule weather_data_yearly:
    input:
        cutout="input/cutout_germany/germany_{year_of_interest}.nc",
        selected_system='output/selected_system_{selected_system_id}.gpkg'
    output:
        temp='output/temperature_series_{selected_system_id}_{year_of_interest}.csv'
#    params:
#        system_id='{selected_system_id}',
#        year_of_interest='{year_of_interest}'
    script:
        'scripts/weather_data.py'

rule weather_data_2022:
    input:
        selected_system='output/selected_system_{selected_system_id}.gpkg'
    output:
        temp2022='output/temperature_series2022_{selected_system_id}.csv'
#    params:
#        system_id='{selected_system_id}'
    script:
        'scripts/weather_data.py'

# defining rules for the creation of hourly heat demand
rule heat_demand:
    input:
        temp='output/temperature_series_{selected_system_id}_{year_of_interest}.csv',
        temp2022='output/temperature_series2022_{selected_system_id}.csv'
    output:
        thh='output/thh_series_{selected_system_id}_{year_of_interest}.csv'
#    params:
#        system_id = '{selected_system_id}',
#        year_of_interest='{year_of_interest}'
    script:
        'scripts/heat_demand.py'

# defining rules for the creation of the hourly water_temperature
rule water_temperature:
    output:
        water_temp='output/water_temperature_{year_of_interest}.csv'
#    params:
#        year_of_interest = '{year_of_interest}'
    script:
        'scripts/water_temperature.py'


rule cost_functions:
    input:
        all_close_potentials = 'output/all_close_potentials_{selected_system_id}.gpkg',
        temp='output/temperature_series_{selected_system_id}_{year_of_interest}.csv',
        water_temp='output/water_temperature_final_{year_of_interest}.csv'
    output:
        cop_series='output/cop_series_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.pkl',
        power_law_models='output/power_law_models_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.csv',
        erzeuger_index='output/erzeugerpreisindex_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.csv',
        combined_data='output/combined_data_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.csv'
#    params:
#        selected_system_id = "{selected_system_id}",
#        year_of_interest = '{year_of_interest}',
#        supply_temp= "{supply_temp}"
    script:
        'scripts/cost_functions.py'

rule linearizing_costs:
    input:
        cop_series = 'output/cop_series_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.pkl',
        power_law_models='output/power_law_models_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.csv',
        erzeuger_index='output/erzeugerpreisindex_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.csv',
        combined_data='output/combined_data_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.csv'
    output:
        all_tech='output/all_technologies_dfs_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°_return:{return_temp}.pkl'
#    params:
#        selected_system_id = "{selected_system_id}",
#        year_of_interest = '{year_of_interest}',
#        supply_temp = "{supply_temp}",
#        return_temp = "{return_temp}"
    script:
        'scripts/linearizing_costs.py'

rule main:
    input:
        max_potentials='output/max_potentials_{selected_system_id}.gpkg',
        temp='output/temperature_series_{selected_system_id}_{year_of_interest}.csv',
        thh='output/thh_series_{selected_system_id}_{year_of_interest}.csv',
        cop_series= 'output/cop_series_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°.pkl',
        all_tech='output/all_technologies_dfs_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°_return:{return_temp}.pkl'
    output:
        results='output/results_{selected_system_id}_{year_of_interest}_supply:{supply_temp}°_return:{return_temp}_elecprice_{electricity_price}'
#    params:
#        selected_system_id = "{selected_system_id}",
#        year_of_interest = '{year_of_interest}',
#        supply_temp = "{supply_temp}",
#        return_temp = "{return_temp}",
#        electricity_price = '{electricity_price}'
    script:
        'scripts/main.py'




