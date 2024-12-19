import utils
import matplotlib.pyplot as plt
# Read data and split them into seperate datasets where people in the same datasets
# must have values for the same exact features

# dic_panel is a dictionary that contains all people of certain dataset from the panel
# dic_pool is a dictionary that contains all people of certain dataset from the pool
# data_panel is a dictionary that contains all people from panels
# dic_panel is a dictionary that contains all people from pools
dic_panel, dic_pool, data_panel, data_pool = utils.read_and_seperate_data()
# Estimate beta values 
# gets subsets of {0,1}^{# of features} that the betas will be based on. 
# We set some constraints such as they have to be between some values 
# (number_of_minimum_features and number_of_maximum_features) and they
# have to be compatible with at least some number of datasets (at least number_of_minimum_datasets)
possible_subsets = utils.compute_possible_subsets(data_panel)

# print(possible_subsets)

temp_dic = {}
entry = next(iter(possible_subsets))
temp_dic[entry] = possible_subsets[entry]

# get estimates for each vector of betas we decided to use
beta_estimates = utils.estimate_dropout(temp_dic)

# TODO: Need to find which subset of betas to choose in order to compute the maximum likelihood
best_features_for_dataset = {}

# compute_beta_statistics(possible_subsets)

# Get sample dropout sets 
# computes dropout probabilities for each dataset, and each beta we choose for that dataset
# and gives dropout sets 
dropouts = utils.get_sample_dropouts(beta_estimates, dic_panel, 10)

# # Computes the best alternate set for the data set & the choice of betas - consequently the dropout sets
alternates = 1
quotas = utils.compute_exact_quotas(dic_panel, beta_estimates)

for dataset in quotas.keys():
    best_alternates = utils.compute_best_alternates(quotas[dataset], dic_panel[dataset], dic_pool[dataset], dropouts[dataset][best_features_for_dataset[dataset]], alternates)

# Datasets missing entries
# ABL, ABQ, ABE

