import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('pre_screening_HP_Brandenburg_Fernwaerme.29_2018_supply:343.15.csv', index_col=0)

best_performers = data.loc[data.groupby('Technology')['Efficiency-Cost Ratio'].idxmax()]
# Calculating the 5% tolerance for these peak values
best_performers['Tolerance Lower Bound'] = best_performers['Efficiency-Cost Ratio'] * 0.9

# Merging this information back to the original dataset
data_with_tolerance = pd.merge(data, best_performers[['Technology', 'Tolerance Lower Bound']], on='Technology',
                               how='left')
# Adding a column to indicate if a fluid is within the tolerance
data_with_tolerance['Within Tolerance'] = data_with_tolerance['Efficiency-Cost Ratio'] >= data_with_tolerance[
    'Tolerance Lower Bound']

technologies = data['Technology'].unique()
tech_positions = range(len(technologies))

# palette = sns.color_palette("deep", len(data['fluid'].unique()))
# sns.set_palette(palette)

fluid_markers = {
    'R290': '^', 'R134a': 'o', 'R600a': '*', 'R600': 'p', 'R245fa': 's', 'R717': 'X', 'R744': '+', 'R1234yf': 'D'
}

fig, ax = plt.subplots(figsize=(10, 6))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "font.size": 16  # Match the font size used in your LaTeX document
})

# Replotting all points with the corrected palette
for fluid, marker in fluid_markers.items():
    subset = data[data['fluid'] == fluid]
    sns.scatterplot(x="Technology", y="Efficiency-Cost Ratio", data=subset, ax=ax, marker=marker, s=120,
                    color='black', label=fluid, alpha=0.4)

# Redrawing shaded areas with the correct bounds and narrower width
technologies = data['Technology'].unique()  # assuming this is defined as the unique entries of 'Technology'
tech_positions = np.arange(len(technologies))  # numerical positions for each technology
best_performers = data.groupby('Technology')['Efficiency-Cost Ratio'].max().reset_index()

for tech, pos in zip(technologies, tech_positions):
    max_ratio = best_performers[best_performers['Technology'] == tech]['Efficiency-Cost Ratio'].values[0]
    lower_bound = max_ratio * 0.9
    ax.fill_betweenx([lower_bound, max_ratio], pos - 0.3, pos + 0.3, color='grey', alpha=0.3)

# Highlighting points within the shaded areas by plotting them again with larger markers
highlighted_data = data_with_tolerance[data_with_tolerance['Within Tolerance']]

for fluid, marker in fluid_markers.items():
    subset = highlighted_data[highlighted_data['fluid'] == fluid]
    sns.scatterplot(x="Technology", y="Efficiency-Cost Ratio", data=subset, ax=ax, marker=marker, s=150, label=None,
                    color='black', alpha=1, #edgecolor='black', linewidth=0.1
                    )

# Custom x-axis labels
custom_labels = ['Luft', 'Abwärme', 'O.-Gewässer']  # These should match the number and order of technologies
ax.set_xticks(tech_positions)  # Set the positions at which to place the ticks
ax.set_xticklabels(custom_labels)  # Set the custom labels

ax.set_ylim(0, 0.011)

ax.tick_params(axis='x', which='major', pad=10
               )
ax.tick_params(axis='y', which='major', pad=25
               )

plt.xticks(rotation=0, ha='center', fontsize=16
           )
plt.yticks(rotation=0, ha='center', fontsize=16
           )

ax.set_xlabel('', #fontsize=14,
              labelpad=15)
ax.set_ylabel('Effizienz-Kosten-Verhätnis', fontsize=16,
              labelpad=15)

ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.2, color='gray')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Handling the legend to ensure it doesn't duplicate
handles, labels = ax.get_legend_handles_labels()
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)
legend = ax.legend(unique_handles, unique_labels, title='Fluid', bbox_to_anchor=(1.05, 1.02), loc='upper left', fontsize=16
                   )
legend.get_title().set_fontsize('16')

plt.tight_layout()
plt.savefig('HP_pre_screening_70°C.pdf')
plt.show()

