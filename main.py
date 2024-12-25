import utils
import matplotlib.pyplot as plt
import pandas as pd

# Get data from file
dic_panel, dic_pool, datasets = utils.read_and_seperate_data()
# Get rid of ABE as it has missing entries
dic_panel.pop('ABE')
dic_pool.pop('ABE')
# These datasets do not contain dropouts
dic_panel.pop('ABR')
dic_pool.pop('ABR')
dic_panel.pop('ABL')
dic_pool.pop('ABL')
dic_panel.pop('ABT')
dic_pool.pop('ABT')

# Get possible subsets of features satisfying constraints we set in parameters file
possible_subsets = utils.compute_possible_subsets()

# Get the best possible betas for each dataset
best_betas = utils.compute_best_betas(possible_subsets, dic_panel)


alternates_LP_loss = {}
alternates_real_loss = {}
# For each dataset
for dataset in best_betas.keys():
    # Compute the best set of alternates that minimizes expected risk
    alternate_set, alternates_LP_loss[dataset] = utils.get_alternates_set(dataset, dic_panel[dataset], dic_pool[dataset], best_betas[dataset][0], best_betas[dataset][2])
    # Calculate the risk on real data
    alternates_real_loss[dataset] = utils.alternates_real_loss(alternate_set, dic_panel[dataset], dic_pool[dataset], dataset)
    print(dataset)
    print(dic_pool[dataset].iloc[alternate_set])
    print(alternates_LP_loss[dataset])
    print(alternates_real_loss[dataset])

keys = list(alternates_real_loss.keys())
values = list(alternates_real_loss.values())

# Create the histogram
plt.bar(keys, values)

# Add labels and title
plt.xlabel('Datasets')
plt.ylabel('Loss')
plt.title('Losses of datasets')

# Show the plot
plt.show()
# # # Datasets missing entries
# # # ABL, ABQ, ABE

