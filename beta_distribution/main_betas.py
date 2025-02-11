import utils
import parameters
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

panels, pools, alternates, quotas = utils.read_and_seperate_data()
# best_betas = {0: (-1, [], []), 1: (-1, [], []), 2: (-1, [], [])}
labels = ['Deschutes', 'Eugene', 'Petaluma']

# for i in range(parameters.datasets):
#     panels_except_i = panels[:i] + panels[i+1:]
#     for subset in utils.generate_subsets():
#         loss, betas, cols = utils.compute_loss_and_betas(subset, panels_except_i)
#         if loss > best_betas[i][0]:
#             best_betas[i] = (loss, betas, cols)
     
# print(panels, pools, best_betas)
# print(quotas)
best_betas = utils.compute_loss_and_betas([1]*parameters.features, panels)
print(best_betas)

fv_name = []
values = []
for fv in parameters.offsets:
    if fv[0] in best_betas[2]:
        fv_name.append(fv[0][:3] + fv[1][:5])
        beta_ind = utils.get_beta_index(fv[0], fv[1], best_betas[2])
        value = best_betas[1][beta_ind]
        print(fv[0], fv[1], value)
        values.append(value)
        # values.append(best_betas[1][utils.get_beta_index(fv[0], fv[1], best_betas[2], 0)])

x = np.arange(len(fv_name))  # Create positions for the 10 bars

# Width of each bar
width = 0.30  # Bar width (adjustable)

# Create the figure and axes
_, ax = plt.subplots()
ax.bar(x, values, width)

ax.set_xlabel('Feature-Value')
ax.set_ylabel('Beta')
ax.set_xticks(x)
ax.set_xticklabels(fv_name, fontsize = 8)

# Add a legend
ax.legend()
plt.savefig("plots/betasAll.png", dpi=300, bbox_inches="tight")
plt.clf()
# Show the plot
# plt.show()

alt_set = utils.calculate_alternate_set(panels[0], pools[0], quotas[0], alternates[0][1], 5, 0, best_betas[1], best_betas[2])
print("panel", panels[0])
print("pool", pools[0])
print("alt_set", alt_set)

# for dataset in range(parameters.datasets):
#     for _, num_alternates in enumerate(parameters.alternates_numbers[dataset]):
#         alternate_sets = []
#         for benchmark in range(len(parameters.benchmarks)):
#             # alt_set = utils.calculate_alternate_set(panels[dataset], pools[dataset], quotas, alternates[dataset][1], num_alternates, benchmark, best_betas[dataset][1], best_betas[dataset][2])
#             alt_set = utils.calculate_alternate_set(panels[dataset], pools[dataset], quotas[dataset], alternates[dataset][1], num_alternates, benchmark, best_betas[1], best_betas[2])
#             alternate_sets.append(alt_set)
#         losses = [[] for _ in range(len(parameters.benchmarks))] 
#         for _ in range(parameters.plot_samples):
#             # cur_losses = utils.calculate_loss_sets(alternate_sets[:len(alternate_sets)-1], panels[dataset], pools[dataset], quotas, best_betas[dataset][1], best_betas[dataset][2], num_alternates)
#             cur_losses = utils.calculate_loss_sets(alternate_sets[:len(alternate_sets)-1], panels[dataset], pools[dataset], quotas[dataset], best_betas[1], best_betas[2], num_alternates)
#             for i, loss in enumerate(cur_losses):
#                 losses[i].append(loss)
#         print(losses)
#         utils.plot_violin(losses, labels[dataset], dataset, num_alternates)

