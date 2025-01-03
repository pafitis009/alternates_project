import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
# These datasets contain too few features
dic_panel.pop('ABB')
dic_pool.pop('ABB')
dic_panel.pop('ABA')
dic_pool.pop('ABA')

# # Get possible subsets of features satisfying constraints we set in parameters file
possible_subsets = utils.compute_possible_subsets()

print(possible_subsets)
print(len(possible_subsets))
print(datasets)

# Get the best possible betas for each dataset
# best_betas = {}
# for dataset in datasets:
#     best_betas[dataset] = utils.compute_best_betas(dataset, possible_subsets, dic_panel)

# print(best_betas)

best_betas = {'AAZ': (1.814517190205722, np.array([0.95134424, 0.58324395, 0.33456948, 0.77348371, 0.59978618,
       0.30998149, 0.49835878, 0.99160934, 0.30477797, 0.99466538,
       0.72480074, 0.50266754, 0.92517261]), ['A', 'L', 'E', 'F']), 'ABD': (1.8139787842795516, np.array([0.99836291, 0.73226675, 0.41800515, 0.98886551, 0.58050668,
       0.31135914, 0.48972416, 0.99989984, 0.30018272, 0.95817517,
       0.6377026 , 0.39952666, 0.78938686]), ['A', 'L', 'E', 'F']), 'ABI': (1.7177268811897641, np.array([0.94400984, 0.66905076, 0.43876005, 0.9999999 , 0.58021844,
       0.25972777, 0.48122587, 0.98628266, 0.22661199, 0.7760205 ,
       0.56523644, 0.53995369, 0.99673228]), ['A', 'L', 'E', 'F']), 'ABC': (1.8491568725582432, np.array([0.9999999 , 0.65306933, 0.36668943, 0.97571989, 0.57587871,
       0.31108797, 0.50488338, 0.9999999 , 0.2992223 , 0.83275969,
       0.56379759, 0.52591605, 0.99994742]), ['A', 'L', 'E', 'F']), 'ABP': (1.653319669895932, np.array([0.9999999 , 0.59007642, 0.36584897, 0.9999999 , 0.51539657,
       0.2900673 , 0.41859627, 0.9999999 , 0.26571878, 0.9999999 ,
       0.60696664, 0.37635668, 0.9999999 ]), ['A', 'L', 'E', 'F']), 'ABG': (1.738675808613591, np.array([0.95945558, 0.66434679, 0.3674918 , 0.99905961, 0.45452593,
       0.29350673, 0.44608175, 0.9999999 , 0.27704027, 0.9288336 ,
       0.57177178, 0.50934681, 0.9999999 ]), ['A', 'L', 'E', 'F']), 'ABJ': (1.7711057425755128, np.array([0.69641971, 0.73034049, 0.40925652, 0.99999919, 0.95199989,
       0.53489603, 0.9056321 , 0.52796636, 0.5188874 , 0.87936114,
       0.63399303, 0.36229982, 0.65427355]), ['A', 'L', 'E', 'F']), 'ABS': (1.7912990917216705, np.array([0.81268673, 0.66555454, 0.53189397, 0.79687659, 0.78653439,
       0.96503863, 0.4921973 , 0.39247646, 0.99616475, 0.30832736]), ['A', 'D', 'F', 'G']), 'ABX': (1.7338721808171773, np.array([0.94455365, 0.61995525, 0.34321362, 0.81805711, 0.68377851,
       0.72875159, 0.90929753, 0.80493488, 0.59744106, 0.20609091]), ['A', 'D', 'F', 'G']), 'ABZ': (1.7940709523982845, np.array([0.73180143, 0.53857772, 0.5126273 , 0.99967579, 0.72798121,
       0.94859202, 0.75351248, 0.5196523 , 0.99498   , 0.27489013]), ['A', 'D', 'F', 'G']), 'ABU': (1.5980174485690022, np.array([0.9999999 , 0.50532417, 0.35868695, 0.71713297, 0.54223396,
       0.96295861, 0.9929651 , 0.53855467, 0.9999999 , 0.17904773]), ['A', 'D', 'F', 'G']), 'ABY': (1.7381921562433142, np.array([0.79410847, 0.64506035, 0.67603857, 0.44839792, 0.83608343,
       0.83269659, 0.51467996, 0.79038106, 0.76569529, 0.53758403,
       0.9999999 , 0.30794706]), ['L', 'D', 'F', 'G']), 'ABK': (1.7652618577550385, np.array([0.86758321, 0.75236666, 0.37856659, 0.9999999 , 0.55880667,
       0.26028818, 0.43960905, 0.99999219, 0.29044982, 0.99980697,
       0.57341972, 0.48853801, 0.99916542]), ['A', 'L', 'E', 'F']), 'ABQ': (1.6661944688459354, np.array([0.61453984, 0.95954205, 0.51336188, 0.86666165, 0.52509621,
       0.30140772, 0.48438765, 0.99944572, 0.32354044, 0.99873881,
       0.77501348, 0.37491736, 0.87318758]), ['A', 'L', 'E', 'F']), 'ACA': (1.818170992082351, np.array([0.82274147, 0.73619695, 0.41567279, 0.97866696, 0.59370316,
       0.27634529, 0.49062295, 0.99971348, 0.29412225, 0.95756275,
       0.60356494, 0.50411439, 0.99772801]), ['A', 'L', 'E', 'F']), 'ACB': (1.6584512744118942, np.array([0.58506381, 0.81395119, 0.54565805, 0.9999999 , 0.99966503,
       0.50141784, 0.47909158, 0.38230263, 0.93139574, 0.30895629]), ['A', 'D', 'F', 'G'])}

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
    # alternate_set_01, alternates_01_ERM_loss[dataset] = utils.get_01_alternates_set(dataset, dic_panel[dataset], dic_pool[dataset], best_betas[dataset][1], best_betas[dataset][2])
    random_alternate_set = utils.get_random_alternate_set(dataset, dic_pool, len(alternate_set_L1))
    print(alternate_set_L1, len(alternate_set_L1))
    # # # # # # Calculate the risk on real data
    alternates_L1_loss[dataset] = utils.alternates_real_loss(alternate_set_L1, dic_panel[dataset], dic_pool[dataset], dataset)
    # alternates_01_loss[dataset] = utils.alternates_real_loss(alternate_set_01, dic_panel[dataset], dic_pool[dataset], dataset)
    alternates_empty_loss[dataset] = utils.alternates_real_loss([], dic_panel[dataset], dic_pool[dataset], dataset)
    alternates_random_loss[dataset] = utils.alternates_real_loss(random_alternate_set, dic_panel[dataset], dic_pool[dataset], dataset)

    labels.append(dataset)
    L1_loss.append(alternates_L1_loss[dataset])
    # binary_loss.append(alternates_01_loss[dataset])
    empty_loss.append(alternates_empty_loss[dataset])
    random_loss.append(alternates_random_loss[dataset])
    
print(labels)
print(L1_loss)
print(empty_loss)
print(random_loss)
# L1_loss = [0.10818713450292397, 0.21978021978021978, 2.006393123784428, 0, 1.7610950938265986, 0.9962647295889081, 2.0801844354475936, 1.2063547563547565, 1.5861940335178875, 0.8173332414711725, 1.2205862384741692, 3.0190520048278677, 1.6016956162117453, 3.347826439400068, 0.2591093117408907, 1.368259189640768]
# empty_loss = [0.22274652147610408, 0.22863122032133354, 3.1336825184854264, 0, 4.3038135461459, 1.8114790184736753, 2.1971434997750787, 1.9637774262774264, 2.767933192787882, 1.3864119596878215, 3.3371692962210204, 3.513592894888267, 1.6712242765723242, 4.35142159812317, 0.8437918290859467, 3.507530980228348]

# X-axis position for the bars
x = np.arange(len(labels))  # Create positions for the 10 bars

# Width of each bar
width = 0.2  # Bar width (adjustable)

# Create the figure and axes
_, ax = plt.subplots()

# Plot the bars
ax.bar(x - 1.5*width, L1_loss, width, label='l1 loss', color='blue')
# ax.bar(x - width/2, binary_loss, width, label='0-1 loss', color='green')
ax.bar(x + width/2, empty_loss, width, label='empty loss', color='red')
ax.bar(x + 1.5*width, random_loss, width, label='random loss', color='yellow')

# Add labels and title
ax.set_xlabel('Dataset')
ax.set_ylabel('Loss')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add a legend
ax.legend()
plt.savefig("plots/stats.png", dpi=300, bbox_inches="tight")
# Show the plot
plt.show()

# Optionally, save the plot

