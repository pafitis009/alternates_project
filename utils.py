import numpy as np
import pandas as pd
import parameters
import math
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.optimize import minimize
from scipy.optimize import Bounds

def read_and_seperate_data():
    file_path = "cleaned_anonymized_data.csv"  # Replace with your CSV file path
    data = pd.read_csv(file_path)
    
    data_per_group_panel = data[data['STATUS'].isin(["Selected", "Selected, dropped out"])]
    data_per_group_pool = data[data['STATUS'].isin(["Not selected"])]

    unique_categories = data['DATA_ID'].unique()

    dic_panel = {}
    dic_pool = {}

    for category in unique_categories:
        subset = data_per_group_panel[data_per_group_panel['DATA_ID'] == category]
        dic_panel[category] = subset
    
    for category in unique_categories:
        subset = data_per_group_pool[data_per_group_pool['DATA_ID'] == category]
        dic_pool[category] = subset
    
    return dic_panel, dic_pool, data_per_group_panel, data_per_group_pool

def compute_possible_subsets(data):
    def binary_strings(n = parameters.number_of_features):
        for i in range(2**n):
            st = format(i, f'0{n}b')
            if st.count('1') >= parameters.number_of_minimum_features and st.count('1') <= parameters.number_of_maximum_features:
                yield st
    def check_non_empty_dataframe(binary_string, df):
        columns_to_check = [col for col, bit in zip(df.columns, binary_string) if bit == '1']
        filtered_df = df.dropna(subset=columns_to_check)
        return filtered_df
    
    dic = {}
    for st in binary_strings():
        st = '11' + st + '1'
        filtered = check_non_empty_dataframe(st, data)
        unique_categories = filtered['DATA_ID'].unique()
        if len(unique_categories) >= parameters.number_of_minimum_datasets:
            dic[st] = filtered
    return dic

def estimate_dropout_for_subset(data, st):
    columns_to_drop = [col for col, bit in zip(data.columns, st) if bit == '0']
    data = data.drop(columns=columns_to_drop)
    filtered_data = data.drop(columns=['DATA_ID', 'Number'])  # Drop unnecessary columns

    filtered_data['y'] = filtered_data['STATUS'].apply(lambda x: 1 if x == "Selected" else 0)
    filtered_data = filtered_data.drop(columns=['STATUS'])  # Drop the STATUS column after mapping

    # Prepare the feature matrix and target vector
    feature_matrix = filtered_data.drop(columns=['y']).values
    target_vector = filtered_data['y'].values

    # Calculate the total number of unique values per feature
    unique_values_per_feature = [set({value for value in feature_matrix[:, i] if value != -1}) for i in range(feature_matrix.shape[1])]
    num_parameters = 1 + sum([len(x) for x in unique_values_per_feature])  # Beta dimension
    
    feature_value_map = {}
    value_feature_map = {}
    offset = 1  # Start after beta_0
    for i, unique_values in enumerate(unique_values_per_feature):
        for value in unique_values:  # Adjust this loop if features aren't integer-encoded
            feature_value_map[(i, value)] = offset
            value_feature_map[offset] = (i, value)
            offset += 1
    # print(feature_matrix)
    # print(target_vector)
    # Define the log-likelihood function
    def log_likelihood(beta, X, y):
        log_likelihood_value = 0
        for v, y_v in zip(X, y):
            # Calculate the linear combination: beta_0 + sum(beta_{f_i, v_i})
            beta_sum = beta[0] 
            # print(v, y_v)
            for i, v_i in enumerate(v):
                if v_i == -1:
                    continue
                beta_sum *= beta[feature_value_map[(i, v_i)]]
            
            # Calculate the likelihood components

            if y_v == 1:
                log_likelihood_value += (np.log(beta_sum))
            else:
                log_likelihood_value += np.log(1 - beta_sum)
        
        return -log_likelihood_value  # Negate for maximization

    # # Initial beta values
    beta_initial = np.random.rand(num_parameters)

    # Scale and shift to the range [-1, -0.5]
    cnt = 0
    cnt_0 = [0]*num_parameters
    cnt_1 = [0]*num_parameters
    for i in range(len(feature_matrix)):
        if target_vector[i] == 0:
            cnt+= 1
            for j, f in enumerate(feature_matrix[i]):
                if f == -1:
                    continue
                cnt_0[feature_value_map[(j,f)]] += 1
        else:
            for j, f in enumerate(feature_matrix[i]):
                if f == -1:
                    continue
                cnt_1[feature_value_map[(j,f)]] += 1

    bounds = Bounds([0.00001] * num_parameters, [0.9999999] * num_parameters)

    # Optimize the log-likelihood function
    result = minimize(
        log_likelihood,
        beta_initial,
        args=(feature_matrix, target_vector),
        method='L-BFGS-B',
        bounds = bounds,
        options={'disp': True}
    )

    # # Extract the optimal beta
    optimal_beta = result.x
    print("Optimal beta:", optimal_beta)
    # print(unique_values_per_feature)
    # for i, val in enumerate(optimal_beta):
    #     if i == 0:
    #         print("Beta 0: " + str(val))
    #     else:
    #         if val == 0.9999999:
    #             print(value_feature_map[i], 1, cnt_1[i], cnt_0[i])
    #         else:
    #             print(value_feature_map[i], val, cnt_1[i], cnt_0[i])
    return (optimal_beta, value_feature_map, feature_value_map)

def estimate_dropout(possible_subsets):
    dic_betas = {}
    for st in possible_subsets.keys():
        betas, mpvf, mpfv = estimate_dropout_for_subset(possible_subsets[st], st)
        dic_betas[st] = (betas, mpvf, mpfv)
    return dic_betas
def compute_beta_statistics(possible_subsets):
    statistics = {}
    for subset in possible_subsets.keys():
        temp_dic = {}
        temp_dic[subset] = possible_subsets[subset]
        beta_estimates = estimate_dropout(temp_dic)
        col_names = ['Number','STATUS','A','L','D','E','F','G','H','I','B','J','K','C','M','N','DATA_ID']
        cnt = 0
        ones = {}
        betas, mpvf, _ = beta_estimates[subset]
        for i in range(2, 16):
            if subset[i] == '1':
                ones[cnt] = i
                cnt += 1
        for i, beta in enumerate(betas):
            if i == 0:
                feature_pair = 'beta_0'
            else:
                feature_pair = (col_names[ones[mpvf[i][0]]], mpvf[i][0])
            
            if feature_pair not in statistics.keys():
                statistics[feature_pair] = [beta]
            else:
                statistics[feature_pair].append(beta)

        print(statistics)

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each key's values as a series of points
    for i, (key, val_list) in enumerate(statistics.items()):
        plt.scatter([str(key)] * len(val_list), val_list, label=str(key), s=100)  # Adjust marker size with `s`

    # Add labels and title
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title('Scatter Plot of Values per Key')
    plt.ylim(0, 1)  # Since values are between 0 and 1

    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()


def get_sample_dropouts(beta_estimates, dic_panel, num_samples):
    final_samples = {}
    for dataset in dic_panel.keys():
        data = dic_panel[dataset]
        samples_dataset = {}
        for st in beta_estimates.keys():
            columns_to_check = [col for col, flag in zip(data.columns, st) if flag == '1']
            contains_na = data[columns_to_check].isna().any().any()
            if contains_na:
                samples_dataset[st] = []
                break
            columns_to_drop = [col for col, bit in zip(data.columns, st) if bit == '0']
            new_data = data.drop(columns=columns_to_drop)
            new_data = new_data.drop(columns=['DATA_ID', 'Number', 'STATUS'])
            probs = []
            for _,row in new_data.iterrows():
                cur_p = 1
                idx = 0
                for col in new_data.columns:
                    cur_p = cur_p * beta_estimates[st][0][beta_estimates[st][2][(idx, row[col])]]
                    idx += 1
                probs.append(cur_p)
            samples = np.random.binomial(1, 1 - np.array(probs), size=(num_samples, len(probs)))
            samples_dataset[st] = samples
        final_samples[dataset] = samples_dataset
    return final_samples

def compute_st(panel):
    return ''.join(['1' if panel[col].notna().any() else '0' for col in panel.columns])

def compute_exact_quotas(dic_panel, beta_estimates):
    quotas = {}
    for dataset in dic_panel.keys():
        quotas_dataset = {}
        cur_data = dic_panel[dataset].copy()
        st = compute_st(cur_data)
        columns_to_keep = [col for col, keep in zip(cur_data.columns, st) if keep == '1']
        cur_data = cur_data[columns_to_keep]
        cur_data = cur_data.drop(columns=['DATA_ID', 'Number', 'STATUS'])
        for _,row in cur_data.iterrows():
            idx = 0
            for col in cur_data.columns:
                cur_feature = (idx, row[col])
                if row[col] != row[col]:
                    print(dataset)
                if cur_feature not in quotas_dataset.keys():
                    quotas_dataset[cur_feature] = 1
                else:
                    quotas_dataset[cur_feature] += 1
                idx += 1
        quotas[dataset] = quotas_dataset
    return quotas

# TODO: For Carmel, fill out this function that computes the best set of alternates
def compute_best_alternates(quotas, panel, pool, dropout_set, num_alternates):
    """
    quotas: dictionary that maps feature-value pairs to number of people needed for that fv pair
    panel: pandas dataframe containing all the people in the pool of that dataset
    pool: pandas dataframe containing all people in the pool of that dataset
    dropouts: list that contains len(dropouts) dropout sets. Each dropout set is represented with a list of size equal to the length of the panel and contains 0 if the person stays and 1 if they dropout
    num_alternates: integer, number of alternates allowed
    """
    return 0












# BRUTE FORCE COMPUTATION
# def compute_score(alternates, quotas, panel, dropouts, number_of_alternates):
#     def binary_strings(n = number_of_alternates):
#         for i in range(2**n):
#             st = format(i, f'0{n}b')
#             yield st
#     cur_quotas = quotas.copy()
#     cur_panel = panel.copy()

#     st = compute_st(cur_panel)
#     columns_to_keep = [col for col, keep in zip(cur_panel.columns, st) if keep == '1']
#     cur_panel = cur_panel[columns_to_keep]
#     cur_panel = cur_panel.drop(columns=['DATA_ID', 'Number', 'STATUS'])
#     for i, dropout in enumerate(dropouts):
#         if dropout == 0 or dropout == '0':
#             continue
#         row = cur_panel.iloc[i]
#         for i, col in enumerate(cur_panel.columns):
#             if isinstance(row[col], float):
#                 if math.isnan(row[col]):
#                     continue
#             cur_quotas[(i, row[col])] -= 1
#     smallest_loss = 100
#     for R in binary_strings():
#         idx1 = 0
#         for _, row in alternates.iterrows():
#             if R[idx1] == '0':
#                 continue
#             idx2 = 0
#             for col in alternates.columns:
#                 if isinstance(row[col], float):
#                     if math.isnan(row[col]):
#                         continue
#                 if (idx2, row[col]) not in cur_quotas.keys():
#                     idx2 += 1
#                     continue
#                 cur_quotas[(idx2, row[col])] += 1
#             idx1 += 1
#         loss = 0
#         for feature in quotas.keys():
#             if isinstance(feature[1], float):
#                 if math.isnan(feature[1]):
#                     continue
#             loss += ((np.abs(quotas[feature] - cur_quotas[feature])) / quotas[feature])
#         if loss <= smallest_loss:
#             smallest_loss = loss
#         idx1 = 0
#         for _, row in alternates.iterrows():
#             if R[idx1] == '0':
#                 continue
#             idx2 = 0
#             for col in alternates.columns:
#                 if isinstance(row[col], float):
#                     if math.isnan(row[col]):
#                         continue
#                 if (idx2, row[col]) not in cur_quotas.keys():
#                     idx2 += 1
#                     continue
#                 cur_quotas[(idx2, row[col])] -= 1
#                 idx2 += 1
#             idx1 += 1
#     return smallest_loss

# def get_best_alternates(quotas, panel, pool, samples, num_alternates):
#     def binary_strings_with_k_ones(n = len(pool), k = num_alternates):
#         if k > n:
#             return
#         for ones_positions in combinations(range(n), k):
#             binary = [False] * n
#             for pos in ones_positions:
#                 binary[pos] = True
#             yield binary
#     best_alternates_set = 0
#     best_alternates_loss = 1000
#     for alt_st in binary_strings_with_k_ones():
#         alternates_df = pool[alt_st]
#         loss = 0
#         for dropouts in samples:
#             loss += compute_score(alternates_df, quotas, panel, dropouts, num_alternates)
#         loss = loss / len(samples)
#         if loss <= best_alternates_loss:
#             best_alternates_loss = loss
#             best_alternates_set = alt_st
#     return (best_alternates_set, loss)


# def compute_best_alternates(quotas, dic_pool, dic_panel, samples, number_of_alternates):
#     losses = {}
#     for dataset in samples.keys():
#         st = compute_st(dic_pool[dataset])
#         alternates_df = dic_pool[dataset].copy()
#         columns_to_keep = [col for col, keep in zip(alternates_df.columns, st) if keep == '1']
#         alternates_df = alternates_df[columns_to_keep]
#         alternates_df = alternates_df.drop(columns=['DATA_ID', 'Number', 'STATUS'])
#         for st in samples[dataset]:
#             print(dataset, st)
#             if len(samples[dataset][st]) == 0:
#                 continue
#             best_alternate_set, best_loss = get_best_alternates(quotas[dataset], dic_panel[dataset], alternates_df, samples[dataset][st], number_of_alternates)
#             losses[(dataset, st)] = (best_alternate_set, best_loss)
#     return losses
