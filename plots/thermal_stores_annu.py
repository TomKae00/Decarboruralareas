import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def process_scenario(directory):
    # Regex to parse the directory name for parameters
    pattern = re.compile(r"supply:(\d+\.\d+)_return.*elecprice_(\d+).*")
    match = pattern.search(directory.replace('/', ':'))
    if not match:
        print(f"No match found for directory: {directory}")
        return None

    supply_temp = float(match.group(1))
    electricity_price = int(match.group(2))

    # Construct file paths
    links_file = os.path.join(directory, 'links.csv')
    stores_file = os.path.join(directory, 'stores.csv')
    links_p0_file = os.path.join(directory, 'links-p0.csv')

    # Try reading the data files
    try:
        links = pd.read_csv(links_file)
        stores = pd.read_csv(stores_file)
        links_p0 = pd.read_csv(links_p0_file)
    except FileNotFoundError:
        print(f"Missing files in {directory}.")
        return None

    # Process Links for CAPEX and OPEX
    links_details = links[links['build_opt'] == 1]
    links_capex = links_details['p_nom_opt'] * links_details['capital_cost']
    links_capex_sum = links_capex.sum()

    links_opex_sum = 0
    if not links_p0.empty:
        # Extracting component names from links_details that are in links_p0's columns
        valid_components = links_details['name'].tolist()
        valid_columns = [col for col in links_p0.columns if
                         any(comp in col for comp in valid_components)]
        # Calculate total_p0 by summing only the valid columns
        if valid_columns:
            links_p0['total_p0'] = links_p0[valid_columns].sum(axis=1)

            # Calculate Opex including the cost of electricity
            total_valid_p0_sum = links_p0['total_p0'].sum()
            links_opex = links_details.loc[
                             links_details['name'].isin(valid_components), 'marginal_cost'].sum() * total_valid_p0_sum
            links_opex += electricity_price * total_valid_p0_sum
            links_opex_sum = links_opex

    # Process Stores for CAPEX (assuming no OPEX for stores as per your setup)
    stores_details = stores[stores['e_nom_opt'] > 0.1]
    stores_capex = stores_details['e_nom_opt'] * stores_details['capital_cost']
    stores_capex_sum = stores_capex.sum()

    stores_opex_sum = 0  # Placeholder if you later need to calculate OPEX for stores

    return {
        'electricity_price': electricity_price,
        'supply_temperature': supply_temp,
        'links_capex': links_capex_sum,
        'links_opex': links_opex_sum,
        'stores_capex': stores_capex_sum,
        'stores_opex': stores_opex_sum
    }


def process_all_scenarios(base_directory):
    scenarios = [d for d in os.listdir(base_directory) if
                 os.path.isdir(os.path.join(base_directory, d)) and 'redispatch' not in d]
    results = []

    for scenario in scenarios:
        directory = os.path.join(base_directory, scenario)
        result = process_scenario(directory)
        if result:
            results.append(result)

    grouped_results = {k: [] for k in set(r['supply_temperature'] for r in results)}
    for result in results:
        grouped_results[result['supply_temperature']].append(result)

    arrays_dict = {}
    for temp, group in grouped_results.items():
        sorted_group = sorted(group, key=lambda x: x['electricity_price'])
        arrays_dict[f"CAPEX_Links_{temp}"] = np.array([g['links_capex'] for g in sorted_group])
        arrays_dict[f"OPEX_Links_{temp}"] = np.array([g['links_opex'] for g in sorted_group])
        arrays_dict[f"CAPEX_Stores_{temp}"] = np.array([g['stores_capex'] for g in sorted_group])
        arrays_dict[f"OPEX_Stores_{temp}"] = np.array([0 for _ in sorted_group])  # Static as no OPEX calculation

    return arrays_dict


# Load the data
base_directory1 = 'Data/WP/HWS/'
base_directory2 = 'Data/WP/PTES/'
data_arrays = process_all_scenarios(base_directory1)
data_arrays2 = process_all_scenarios(base_directory2)

"""""
plotting
"""""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "font.size": 34  # Match the font size used in your LaTeX document
})
fontsize_set = 34

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11))

sns.axes_style("white")
sns.set_style("ticks")
sns.set_context("talk")

capex_links_130_HWS = data_arrays['CAPEX_Links_343.15']
capex_stores_130_HWS = data_arrays['CAPEX_Stores_343.15']
opex_links_130_HWS = data_arrays['OPEX_Links_343.15']
opex_stores_130_HWS = data_arrays['OPEX_Stores_343.15']
capex_links_170_HWS = data_arrays['CAPEX_Links_363.15']
capex_stores_170_HWS = data_arrays['CAPEX_Stores_363.15']
opex_links_170_HWS = data_arrays['OPEX_Links_363.15']
opex_stores_170_HWS = data_arrays['OPEX_Stores_363.15']
capex_links_210_HWS = data_arrays['CAPEX_Links_383.15']
capex_stores_210_HWS = data_arrays['CAPEX_Stores_383.15']
opex_links_210_HWS = data_arrays['OPEX_Links_383.15']
opex_stores_210_HWS = data_arrays['OPEX_Stores_383.15']
price = ['130', '170', '210']



# plot details
bar_width = 0.25
epsilon = .025
line_width = 1
opacity = 0.7
pos_bar_positions = np.arange(len(capex_links_130_HWS))
neg_bar_positions = pos_bar_positions + bar_width
neg_neg_bar_positions = neg_bar_positions + bar_width

ax1.grid(axis='y', linestyle='--', linewidth=0.5, color='grey', which='major', zorder=0, alpha=0.5)

# make bar plots
capex_links_130_bar = ax1.bar(pos_bar_positions, capex_links_130_HWS, bar_width - epsilon,
                              color='indianred',
                              edgecolor='indianred',
                              linewidth=line_width,
                              label='CAPEX Erzeuger',
                              zorder=10)
capex_stores_130_bar = ax1.bar(pos_bar_positions, capex_stores_130_HWS, bar_width - epsilon,
                               bottom=capex_links_130_HWS,
                               alpha=opacity,
                               color='blue',
                               edgecolor='blue',
                               linewidth=line_width,
                               #hatch='//',
                               label='CAPEX Speicher',
                               zorder=10)
opex_links_130_bar = ax1.bar(pos_bar_positions, opex_links_130_HWS, bar_width - epsilon,
                             bottom=capex_links_130_HWS + capex_stores_130_HWS,
                             alpha=opacity,
                             color='white',
                             edgecolor='indianred',
                             linewidth=line_width,
                             #hatch='0',
                             label='OPEX Erzeuger',
                             zorder=10)
#    opex_stores_130_bar = plt.bar(pos_bar_positions, opex_stores_130, bar_width - epsilon,
#                                  bottom=capex_links_130 + capex_stores_130 + opex_links_130,
#                                  alpha=opacity,
#                                  color='white',
#                                  edgecolor='blue',
#                                  linewidth=line_width,
#                                  #hatch='/',
#                                  label='Opex HWS',
#                                  zorder=10)
capex_links_170_bar = ax1.bar(neg_bar_positions, capex_links_170_HWS, bar_width - epsilon,
                              color='indianred',
                              edgecolor='indianred',
                              linewidth=line_width,
                              #label='HPV+ Mutations',
                              zorder=10
                              )
capex_stores_170_bar = ax1.bar(neg_bar_positions, capex_stores_170_HWS, bar_width - epsilon,
                               bottom=capex_links_170_HWS,
                               alpha=opacity,
                               color='blue',
                               edgecolor='blue',
                               linewidth=line_width,
                               hatch='/',
                               #label='HPV+ CNA',
                               zorder=10
                               )
opex_links_170_bar = ax1.bar(neg_bar_positions, opex_links_170_HWS, bar_width - epsilon,
                             bottom=capex_links_170_HWS + capex_stores_170_HWS,
                             alpha=opacity,
                             color='white',
                             edgecolor='indianred',
                             linewidth=line_width,
                             hatch='/',
                             #label='HPV+ Both',
                             zorder=10
                             )
#    opex_stores_170_bar = plt.bar(neg_bar_positions, opex_stores_170, bar_width - epsilon,
#                                  bottom=capex_links_170 + capex_stores_170 + opex_links_170,
#                                  alpha=opacity,
#                                  color='white',
#                                  edgecolor='blue',
#                                  linewidth=line_width,
#                                  hatch='/',
#                                  #label='HPV+ Both',
#                                  zorder=10
#                                  )
capex_links_210_bar = ax1.bar(neg_neg_bar_positions, capex_links_210_HWS, bar_width - epsilon,
                              color='indianred',
                              edgecolor='indianred',
                              linewidth=line_width,
                              #label='HPV+ Mutations',
                              zorder=10
                              )
capex_stores_210_bar = ax1.bar(neg_neg_bar_positions, capex_stores_210_HWS, bar_width - epsilon,
                               bottom=capex_links_210_HWS,
                               alpha=opacity,
                               color='blue',
                               edgecolor='blue',
                               linewidth=line_width,
                               hatch='//',
                               #label='HPV+ CNA',
                               zorder=10
                               )
opex_links_210_bar = ax1.bar(neg_neg_bar_positions, opex_links_210_HWS, bar_width - epsilon,
                             bottom=capex_links_210_HWS + capex_stores_210_HWS,
                             alpha=opacity,
                             color='white',
                             edgecolor='indianred',
                             linewidth=line_width,
                             hatch='//',
                             #label='HPV+ Both',
                             zorder=10
                             )
#    opex_stores_210_bar = plt.bar(neg_neg_bar_positions, opex_stores_210, bar_width - epsilon,
#                                  bottom=capex_links_210 + capex_stores_210 + opex_links_210,
#                                  alpha=opacity,
#                                  color='white',
#                                  edgecolor='blue',
#                                  linewidth=line_width,
#                                  hatch='//',
#                                  #label='HPV+ Both',
#                                  zorder=10
#                                  )
ax1.set_xticks(neg_bar_positions, price, rotation=0, fontsize=fontsize_set)
ax1.set_ylabel('Annuitätischen Gesamtkosten [€]', fontsize=fontsize_set)
#    plt.subplots_adjust(right=0.8)

y_max = ax1.get_ylim()[1]  # Get the current maximum y value
ax1.set_ylim(top=9.9e6)

ax1.tick_params(axis='y', labelsize=fontsize_set)  # Set the y-axis tick label size
ax1.tick_params(axis='x', labelsize=fontsize_set)
ax1.yaxis.get_offset_text().set_fontsize(fontsize_set)  # Set the offset text font size


ax1.set_xlabel('Strompreis [€/MWh]', fontsize=fontsize_set)


#    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='grey', which='major')

# First legend: Labels of costs (Positioned at top right, inside of plot)
handles, labels = ax1.get_legend_handles_labels()
first_legend = ax1.legend(handles[:4], labels[:4], loc='upper left', bbox_to_anchor=(0.24, 1),
                          title="Kostengruppe", fontsize=fontsize_set, title_fontsize=fontsize_set-8)
ax1.add_artist(first_legend)

# Second legend: Hatches indicating temperatures (Positioned below the first legend)
legend_elements = [
    mpatches.Patch(facecolor='white', edgecolor='black', label='70/40'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label='90/50'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='110/60')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.98, 0.6), title="VLT/RLT")
ax1.legend(handles=legend_elements, loc='upper left', title="VLT/RLT [°C]", bbox_to_anchor=(0, 1), fontsize=fontsize_set,
            title_fontsize=fontsize_set-8
           )

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

capex_links_130_PTES = data_arrays2['CAPEX_Links_343.15']
capex_stores_130_PTES = data_arrays2['CAPEX_Stores_343.15']
opex_links_130_PTES = data_arrays2['OPEX_Links_343.15']
opex_stores_130_PTES = data_arrays2['OPEX_Stores_343.15']
capex_links_170_PTES = data_arrays2['CAPEX_Links_363.15']
capex_stores_170_PTES = data_arrays2['CAPEX_Stores_363.15']
opex_links_170_PTES = data_arrays2['OPEX_Links_363.15']
opex_stores_170_PTES = data_arrays2['OPEX_Stores_363.15']


# plot details
bar_width = 0.25
epsilon = .025
line_width = 1
opacity = 0.7
pos_bar_positions2 = np.arange(len(capex_links_130_PTES)) + 0.125
print(pos_bar_positions2)
neg_bar_positions2 = pos_bar_positions2 + bar_width
print(neg_bar_positions2)

ax2.grid(axis='y', linestyle='--', linewidth=0.5, color='grey', which='major', zorder=0, alpha=0.5)

# make bar plots
capex_links_130_bar2 = ax2.bar(pos_bar_positions2, capex_links_130_PTES, bar_width - epsilon,
                              color='indianred',
                              edgecolor='indianred',
                              linewidth=line_width,
                              label='CAPEX Erzeuger',
                              zorder=10)
capex_stores_130_bar2 = ax2.bar(pos_bar_positions2, capex_stores_130_PTES, bar_width - epsilon,
                               bottom=capex_links_130_PTES,
                               alpha=opacity,
                               color='blue',
                               edgecolor='blue',
                               linewidth=line_width,
                               #hatch='//',
                               label='CAPEX Speicher',
                               zorder=10)
opex_links_130_bar2 = ax2.bar(pos_bar_positions2, opex_links_130_PTES, bar_width - epsilon,
                             bottom=capex_links_130_PTES + capex_stores_130_PTES,
                             alpha=opacity,
                             color='white',
                             edgecolor='indianred',
                             linewidth=line_width,
                             #hatch='0',
                             label='OPEX Erzeuger',
                             zorder=10)
#    opex_stores_130_bar = plt.bar(pos_bar_positions, opex_stores_130, bar_width - epsilon,
#                                  bottom=capex_links_130 + capex_stores_130 + opex_links_130,
#                                  alpha=opacity,
#                                  color='white',
#                                  edgecolor='blue',
#                                  linewidth=line_width,
#                                  #hatch='/',
#                                  label='Opex HWS',
#                                  zorder=10)
capex_links_170_bar2 = ax2.bar(neg_bar_positions2, capex_links_170_PTES, bar_width - epsilon,
                              color='indianred',
                              edgecolor='indianred',
                              linewidth=line_width,
                              #label='HPV+ Mutations',
                              zorder=10
                              )
capex_stores_170_bar2 = ax2.bar(neg_bar_positions2, capex_stores_170_PTES, bar_width - epsilon,
                               bottom=capex_links_170_PTES,
                               alpha=opacity,
                               color='blue',
                               edgecolor='blue',
                               linewidth=line_width,
                               hatch='/',
                               #label='HPV+ CNA',
                               zorder=10
                               )
opex_links_170_bar2 = ax2.bar(neg_bar_positions2, opex_links_170_PTES, bar_width - epsilon,
                             bottom=capex_links_170_PTES + capex_stores_170_PTES,
                             alpha=opacity,
                             color='white',
                             edgecolor='indianred',
                             linewidth=line_width,
                             hatch='/',
                             #label='HPV+ Both',
                             zorder=10
                             )
#    opex_stores_170_bar = plt.bar(neg_bar_positions, opex_stores_170, bar_width - epsilon,
#                                  bottom=capex_links_170 + capex_stores_170 + opex_links_170,
#                                  alpha=opacity,
#                                  color='white',
#                                  edgecolor='blue',
#                                  linewidth=line_width,
#                                  hatch='/',
#                                  #label='HPV+ Both',
#                                  zorder=10
#                                  )
#capex_links_210_bar2 = ax1.bar(neg_neg_bar_positions, capex_links_210_HWS, bar_width - epsilon,
#                              color='indianred',
#                              edgecolor='indianred',
#                              linewidth=line_width,
#                              #label='HPV+ Mutations',
#                              zorder=10
#                              )
#capex_stores_210_bar = ax1.bar(neg_neg_bar_positions, capex_stores_210_HWS, bar_width - epsilon,
#                               bottom=capex_links_210_HWS,
#                               alpha=opacity,
#                               color='blue',
#                               edgecolor='blue',
#                               linewidth=line_width,
#                               hatch='//',
#                               #label='HPV+ CNA',
#                               zorder=10
#                               )
#opex_links_210_bar = ax1.bar(neg_neg_bar_positions, opex_links_210_HWS, bar_width - epsilon,
#                             bottom=capex_links_210_HWS + capex_stores_210_HWS,
#                             alpha=opacity,
#                             color='white',
#                             edgecolor='indianred',
#                             linewidth=line_width,
#                             hatch='//',
#                             #label='HPV+ Both',
#                             zorder=10
#                             )
#    opex_stores_210_bar = plt.bar(neg_neg_bar_positions, opex_stores_210, bar_width - epsilon,
#                                  bottom=capex_links_210 + capex_stores_210 + opex_links_210,
#                                  alpha=opacity,
#                                  color='white',
#                                  edgecolor='blue',
#                                  linewidth=line_width,
#                                  hatch='//',
#                                  #label='HPV+ Both',
#                                  zorder=10
#                                  )
ax2.set_xticks(neg_bar_positions, price, rotation=0, fontsize=fontsize_set)
#ax2.set_ylabel('Annuitätischen Gesamtkosten [€]', fontsize=fontsize_set)
#    plt.subplots_adjust(right=0.8)

y_max = ax2.get_ylim()[1]  # Get the current maximum y value
ax2.set_ylim(top=9.9e6)
ax2.tick_params(axis='y', labelleft=False)

#ax2.tick_params(axis='y', labelsize=fontsize_set)  # Set the y-axis tick label size
#ax2.tick_params(axis='x', labelsize=fontsize_set)
#ax2.yaxis.get_offset_text().set_fontsize(fontsize_set)  # Set the offset text font size


ax2.set_xlabel('Strompreis [€/MWh]', fontsize=fontsize_set)


plt.grid(axis='y', linestyle='--', linewidth=0.5, color='grey', which='major')


handles, labels = ax2.get_legend_handles_labels()
first_legend = ax2.legend(handles[:4], labels[:4], loc='upper left', bbox_to_anchor=(0.24, 1),
                          title="Kostengruppe", fontsize=fontsize_set, title_fontsize=fontsize_set-8)
ax2.add_artist(first_legend)

# Second legend: Hatches indicating temperatures (Positioned below the first legend)
legend_elements = [
    mpatches.Patch(facecolor='white', edgecolor='black', label='70/40'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label='90/50'),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='110/60')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.98, 0.6), title="VLT/RLT")
ax2.legend(handles=legend_elements, loc='upper left', title="VLT/RLT [°C]", bbox_to_anchor=(0, 1), fontsize=fontsize_set,
            title_fontsize=fontsize_set-8
           )

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)





ax1.set_title('a)', fontsize=fontsize_set, pad=20)
ax2.set_title('b)', fontsize=fontsize_set, pad=20)

#fig.subplots_adjust(left=0.05, right=0.95)

plt.tight_layout()

#plt.savefig('heatmap-E-Kessel.eps', format='eps', dpi=300)
plt.savefig('Kombinierte_Kosten-WP-WWS-PTES.eps', format='eps'#, dpi=300
            )
plt.show()
