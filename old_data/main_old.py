import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Get data from file
dic_panel, dic_pool, datasets = utils.read_and_seperate_data()
# Get rid of ABE as it has missing entries
dic_panel.pop('ABE')
dic_pool.pop('ABE')

# Get possible subsets of features satisfying constraints we set in parameters file
# possible_subsets = utils.compute_possible_subsets()
# print(possible_subsets)
# print(len(possible_subsets))
# print(datasets)
# # Get the best possible betas for each dataset
# best_betas = {}
# for dataset in datasets:
#     best_betas[dataset] = utils.compute_best_betas(dataset, possible_subsets, dic_panel)

# print(best_betas)

best_betas = {'AAZ': (10.994617903723451, np.array([0.81491756, 0.21599543, 0.71561624, 0.95082553, 0.38917863,
       0.75486802, 0.89979976, 0.89098193, 0.72493785, 0.21665237,
       0.04007717, 0.90539031, 0.03503562]), ['A', 'L', 'E', 'F']), 'ABD': (10.966747790292422, np.array([0.13526757, 0.58915096, 0.32955146, 0.72096194, 0.59137953,
       0.31750356, 0.51118665, 0.98979682, 0.29618711, 0.46482134,
       0.53734841]), ['A', 'L', 'E']), 'ABI': (10.417914868914327, np.array([0.41785794, 0.2268939 , 0.03059184, 0.16737364, 0.87760642,
       0.84214276, 0.98609235, 0.01183816, 0.21268985, 0.23471963]), ['L', 'E', 'F']), 'ABC': (11.449443942104535, np.array([0.33374647, 0.53384566, 0.28758704, 0.46754152, 0.88556928,
       0.27969378, 0.61398119, 0.41031308, 0.28440046, 0.5543978 ]), ['L', 'E', 'F']), 'ABP': (9.819796120840763, np.array([0.10701088, 0.5044351 , 0.75627148, 0.38437044, 0.45333935,
       0.96483384, 0.41799032, 0.05484178, 0.17597149, 0.16578087,
       0.4556885 , 0.27477465, 0.97030468]), ['A', 'L', 'E', 'F']), 'ABG': (10.632898915128122, np.array([0.32137556, 0.39581956, 0.25578936, 0.39102675, 0.82396849,
       0.24247681, 0.53572852, 0.32297231, 0.3660115 , 0.75031958]), ['L', 'E', 'F']), 'ABJ': (10.731558118529142, np.array([0.08762778, 0.87966113, 0.05282974, 0.35761469, 0.27003931,
       0.90866981, 0.42733027, 0.05100176]), ['A', 'E', 'F']), 'ABS': (10.961231687781462, np.array([0.2670797 , 0.46984915, 0.27035901, 0.74593541, 0.4317361 ,
       0.34673916, 0.41268232, 0.79919141]), ['A', 'E', 'F']), 'ABX': (10.670157762679505, np.array([0.04392962, 0.07033228, 0.25246778, 0.10201349, 0.29403845,
       0.15959129, 0.48769898, 0.59084471, 0.07591103, 0.59065938]), ['L', 'E', 'F']), 'ABZ': (10.862008095482091, np.array([0.10974424, 0.53591136, 0.29236602, 0.4667654 , 0.91785274,
       0.25291775, 0.61203086, 0.38912396, 0.34030998, 0.71102673]), ['L', 'E', 'F']), 'ABU': (10.649341415766148, np.array([0.94108274, 0.667234  , 0.92911257, 0.24244549, 0.31372357,
       0.60261331, 0.6276313 , 0.57551322, 0.69122117, 0.77013995,
       0.53810636]), ['A', 'L', 'F']), 'ABY': (10.728030627310176, np.array([0.76473529, 0.63205491, 0.57532707, 0.15188637, 0.57453478,
       0.29915616, 0.21412661, 0.24401571]), ['A', 'E', 'F']), 'ABK': (10.823819469977249, np.array([0.46582638, 0.25269477, 0.12975881, 0.36257032, 0.82610253,
       0.51765157, 0.35515585, 0.74310962]), ['A', 'E', 'F']), 'ABQ': (9.878431657664622, np.array([0.2641457 , 0.01504812, 0.0104761 , 0.51516454, 0.85288827,
       0.18019741, 0.57072451, 0.07971924, 0.95450619, 0.86431875,
       0.91311337]), ['A', 'L', 'E']), 'ACA': (11.11260486890954, np.array([0.77313419, 0.50217364, 0.79210327, 0.80081533, 0.15902349,
       0.99198717, 0.59011983, 0.99869453, 0.68344331, 0.08613915,
       0.40027969, 0.77148841, 0.31700433]), ['A', 'L', 'E', 'F']), 'ACB': (10.557603737083527, np.array([0.66002181, 0.53992901, 0.28077712, 0.97468136, 0.46029516,
       0.22249176, 0.43098762, 0.88139568, 0.2657541 , 0.5776713 ,
       0.46735829, 0.55495801, 0.87680818]), ['A', 'L', 'E', 'F'])}



alternates_L1_ERM_loss = {}
alternates_01_ERM_loss = {}
alternates_L1_loss = {}
alternates_01_loss = {}
alternates_empty_loss = {}
alternates_random_loss = {}
# For each dataset
labels = []
L1_loss = []
binary_loss = []
empty_loss = []
random_loss = []
for dataset in best_betas.keys():
    # # # # # # Compute the best set of alternates that minimizes expected risk
    alternate_set_L1, alternates_L1_ERM_loss[dataset] = utils.get_L1_alternates_set(dataset, dic_panel[dataset], dic_pool[dataset], best_betas[dataset][1], best_betas[dataset][2])
    alternate_set_01, alternates_01_ERM_loss[dataset] = utils.get_01_alternates_set(dataset, dic_panel[dataset], dic_pool[dataset], best_betas[dataset][1], best_betas[dataset][2])
    random_alternate_set = utils.get_random_alternate_set(dataset, dic_pool, len(alternate_set_L1))
    print(alternate_set_L1, len(alternate_set_L1))
    # # # # # # Calculate the risk on real data
    alternates_L1_loss[dataset] = utils.alternates_real_loss(alternate_set_L1, dic_panel[dataset], dic_pool[dataset], dataset)
    alternates_01_loss[dataset] = utils.alternates_real_loss(alternate_set_01, dic_panel[dataset], dic_pool[dataset], dataset)
    alternates_empty_loss[dataset] = utils.alternates_real_loss([], dic_panel[dataset], dic_pool[dataset], dataset)
    alternates_random_loss[dataset] = utils.alternates_real_loss(random_alternate_set, dic_panel[dataset], dic_pool[dataset], dataset)

    labels.append(dataset)
    L1_loss.append(alternates_L1_loss[dataset])
    binary_loss.append(alternates_01_loss[dataset])
    empty_loss.append(alternates_empty_loss[dataset])
    random_loss.append(alternates_random_loss[dataset])

print(labels)
print(L1_loss)
print(empty_loss)
print(random_loss)
print(binary_loss)

# X-axis position for the bars
x = np.arange(len(labels))  # Create positions for the 10 bars

# Width of each bar
width = 0.2  # Bar width (adjustable)

# Create the figure and axes
_, ax = plt.subplots()

# Plot the bars
ax.bar(x - 1.5*width, L1_loss, width, label='l1 loss', color='blue')
ax.bar(x - width/2, binary_loss, width, label='0-1 loss', color='black')
ax.bar(x + width/2, empty_loss, width, label='empty loss', color='red')
ax.bar(x + 1.5*width, random_loss, width, label='random loss', color='green')

# Add labels and title
ax.set_xlabel('Dataset')
ax.set_ylabel('Loss')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add a legend
ax.legend()
plt.savefig("plots/stats_chris.png", dpi=300, bbox_inches="tight")
# Show the plot
plt.show()

# Optionally, save the plot

