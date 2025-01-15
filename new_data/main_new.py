import utils
import parameters
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

panels, pools, alternates = utils.read_and_seperate_data()
best_betas = {0: (-1, [], []), 1: (-1, [], []), 2: (-1, [], [])}
labels = ['Deschutes', 'Eugene', 'Petaluma']

for i in range(parameters.datasets):
    panels_except_i = panels[:i] + panels[i+1:]
    for subset in utils.generate_subsets():
        loss, betas, cols = utils.compute_loss_and_betas(subset, panels_except_i)
        if loss > best_betas[i][0]:
            best_betas[i] = (loss, betas, cols)

for i in best_betas.keys():
    fv_name = []
    values = []
    for fv in parameters.offsets:
        if fv[0] in best_betas[i][2]:
            fv_name.append(fv[0][:3] + fv[1][:5])
            values.append(best_betas[i][1][utils.get_beta_index(fv[0], fv[1], best_betas[i][2], 0)])
    
    x = np.arange(len(fv_name))  # Create positions for the 10 bars

    # Width of each bar
    width = 0.30  # Bar width (adjustable)

    # Create the figure and axes
    _, ax = plt.subplots()
    ax.bar(x, values, width)
    
    ax.set_xlabel('Feature-Value')
    ax.set_ylabel('Beta')
    ax.set_xticks(x)
    ax.set_xticklabels(fv_name, fontsize = 5)

    # Add a legend
    ax.legend()
    plt.savefig("plots/betas" + labels[i] + ".png", dpi=300, bbox_inches="tight")
    # Show the plot
    plt.show()

    # # Optionally, save the plot



alternates_L1_ERM_loss = {}
alternates_01_ERM_loss = {}
alternates_L1_loss = {}
alternates_01_loss = {}
alternates_empty_loss = {}
alternates_random_loss = {}
alternates_actual_loss = {}
# For each dataset
L1_loss = []
binary_loss = []
empty_loss = []
random_loss = []
actuals_loss = []
best_loss = []
for i in range(parameters.datasets):
    num_alternates = len(alternates[i][0]) // 2
    
    # # # # # # Compute the best set of alternates that minimizes expected risk
    quotas = utils.compute_quotas(panels[i], best_betas[i][2])
    alternate_set_L1, alternates_L1_ERM_loss[i] = utils.get_L1_alternates_set(panels[i], quotas, pools[i], best_betas[i][1], best_betas[i][2], num_alternates)
    alternate_set_01, alternates_01_ERM_loss[i] = utils.get_01_alternates_set(panels[i], quotas, pools[i], best_betas[i][1], best_betas[i][2], num_alternates)
    random_alternate_set = utils.get_random_alternate_set(pools[i], num_alternates)
    best_l = utils.get_best_loss(panels[i], quotas, pools[i], num_alternates)

    # # # # # # # Calculate the risk on real data
    alternates_L1_loss[i] = utils.alternates_real_loss(alternate_set_L1, quotas, panels[i], pools[i])
    alternates_01_loss[i] = utils.alternates_real_loss(alternate_set_01, quotas, panels[i], pools[i])
    alternates_empty_loss[i] = utils.alternates_real_loss([], quotas, panels[i], pools[i])
    alternates_random_loss[i] = utils.alternates_real_loss(random_alternate_set, quotas, panels[i], pools[i])
    alternates_actual_loss[i] = utils.alternates_real_loss(alternates[i][1][:(len(alternates[i][0]) // 2)], quotas, panels[i], pools[i])

    L1_loss.append(alternates_L1_loss[i])
    binary_loss.append(alternates_01_loss[i])
    empty_loss.append(alternates_empty_loss[i])
    random_loss.append(alternates_random_loss[i])
    actuals_loss.append(alternates_actual_loss[i])
    best_loss.append(best_l)

print(labels)
print(L1_loss)
print(empty_loss)
print(random_loss)
print(actuals_loss)
print(binary_loss)
print(best_loss)

# X-axis position for the bars
x = np.arange(len(labels))  # Create positions for the 10 bars

# Width of each bar
width = 0.10  # Bar width (adjustable)

# Create the figure and axes
_, ax = plt.subplots()

# Plot the bars
ax.bar(x - 2.5*width, L1_loss, width, label='l1 loss', color='blue')
ax.bar(x - 1.5*width, actuals_loss, width, label='actuals loss', color='brown')
ax.bar(x - 0.5*width, binary_loss, width, label='0-1 loss', color='black')
ax.bar(x + 0.5*width, empty_loss, width, label='empty loss', color='red')
ax.bar(x + 1.5*width, random_loss, width, label='random loss', color='green')
ax.bar(x + 2.5*width, best_loss, width, label='best loss', color='orange')

# Add labels and title
ax.set_xlabel('Dataset')
ax.set_ylabel('Loss')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add a legend
ax.legend()
plt.savefig("plots/stats_HD.png", dpi=300, bbox_inches="tight")
# Show the plot
plt.show()

# Optionally, save the plot

