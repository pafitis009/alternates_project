import numpy as np
import pandas as pd
import parameters
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

from mip import Model, xsum, BINARY, MAXIMIZE, MINIMIZE
from itertools import combinations
from scipy.optimize import minimize
from scipy.optimize import Bounds


#  READS data, returns two dictionaries and a list:
#  the two dictionaries are the panels and pools of the different datasets and the list is the
#  different datasets
def read_and_seperate_data():
    file_path = "old_data/cleaned_anonymized_data.csv"  # Replace with your CSV file path
    data = pd.read_csv(file_path)
    
    data_per_group_panel = data[data['STATUS'].isin(["Selected", "Selected, dropped out"])]
    data_per_group_pool = data[data['STATUS'].isin(["Not selected"])]

    unique_categories = data['DATA_ID'].unique()

    unique_categories = unique_categories.tolist()

    dic_panel = {}
    dic_pool = {}

    for category in unique_categories:
        subset = data_per_group_panel[data_per_group_panel['DATA_ID'] == category]
        dic_panel[category] = subset
    
    for category in unique_categories:
        subset = data_per_group_pool[data_per_group_pool['DATA_ID'] == category]
        dic_pool[category] = subset
    
    return dic_panel, dic_pool, unique_categories

# Computes all possible subsets of features with the given parameters 
# returns a dictionary that has a key the binary string i.e. '011000' if we want to include 2nd and 3rd
# feature and maps to a list of datasets that have this subsets of features
def compute_possible_subsets():
    def binary_strings(n = parameters.number_of_features):
        for i in range(2**n):
            st = format(i, f'0{n}b')
            if st.count('1') >= parameters.number_of_minimum_features and st.count('1') <= parameters.number_of_maximum_features:
                yield st
    def check_subset(a, b):
        for i in range(len(a)):
            if a[i] == '1' and b[i] == '0':
                return False
        return True
    
    dic = {}
    for st in binary_strings():
        st = '11' + st + '1'
        supersets = []
        for dataset in parameters.dataset_features.keys():
            if check_subset(st, parameters.dataset_features[dataset]):
                supersets.append(dataset)
        if len(supersets) > parameters.number_of_minimum_datasets:
            dic[st] = supersets
    return dic

def prepare_data(data, subset):
    data['Dropped'] = (data['STATUS'] != 'Selected').astype(int)
    arr = np.array([])
    for i in range(len(subset) + 1):
        if i == 0 or i == 1 or i >= len(subset) - 2 or subset[i] == '0':
            arr = np.append(arr, [0])
        else:
            arr = np.append(arr, [1])
    X = data.loc[:, arr.astype(bool)]
    y = data[['Dropped']]
    return X, y

def get_beta_index(i, v_i, columns):
    j = 0
    idx = 1
    while j <= i:
        if j < i:
            idx += parameters.feature_values[columns[j]]
        else:
            idx += parameters.offsets[(columns[i], v_i)]
        j += 1
    return idx

def compute_dropout_prob(betas, beta_columns, dataset_columns, features):
    p = betas[0]
    offset = 1
    idx = 0
    for i, column in enumerate(dataset_columns):
        if column not in beta_columns:
            continue
        p *= betas[offset + parameters.offsets[(beta_columns[idx], features[i])]]
        offset += parameters.feature_values[beta_columns[idx]]
        idx += 1
    return p

def compute_betas(X, y, columns):
    num_parameters = 1
    
    for column in columns:
        num_parameters += parameters.feature_values[column]
    
    def log_likelihood(beta):
        log_likelihood_value = 0
        for v, y_v in zip(X, y):
            beta_sum = beta[0] 
            for i, v_i in enumerate(v):
                if v_i == -1:
                    continue
                beta_sum += np.log(beta[get_beta_index(i, v_i, columns)])

            if y_v == 1:
                log_likelihood_value += (beta_sum)
            else:
                log_likelihood_value += np.log(1 + np.exp(beta_sum))
        
        return -log_likelihood_value
    beta_initial = np.random.rand(num_parameters)
    bounds = Bounds([0.00001] * num_parameters, [0.9999999] * num_parameters)

    result = minimize(
        log_likelihood,
        beta_initial,
        args=(),
        method='L-BFGS-B',
        bounds = bounds,
        options={'disp': True}
    )

    optimal_beta = result.x
    print("Optimal beta:", optimal_beta)
    return optimal_beta

def get_loss_betas(betas, X, y, columns):
    y_pred = []
    for person in X:
        p = betas[0]
        for i, v_i in enumerate(person):
            p *= betas[get_beta_index(i, v_i, columns)]
        y_pred.append(p)
    y_pred = np.array(y_pred)
    y_true = np.array(y)

    return np.linalg.norm(y_pred - y_true)

def k_fold_validation(X, y):
    k = len(X)
    columns = X[0].columns.tolist()
    losses = []
    X_all = pd.concat([df for df in X])
    y_all = pd.concat([df for df in y])

    for i in range(k):
        X_test = X[i]
        y_test = y[i]
        X_train = pd.concat([df for idx, df in enumerate(X) if idx != i])
        y_train = pd.concat([df for idx, df in enumerate(y) if idx != i])

        print(X_train)
        print(y_train)
        # Compute optimal betas for the training set
        betas = compute_betas(np.array(X_train), np.array(y_train), columns)

        # Compute the loss for the test set
        loss = get_loss_betas(betas, np.array(X_test), np.array(y_test), columns)
        losses.append(loss)

    average_loss = np.mean(losses)

    return compute_betas(np.array(X_all), np.array(y_all), columns), average_loss, columns

def compute_best_betas(cur_dataset, possible_subsets, dic_panel):
    best_loss = 1000000.0
    best_betas = {}
    best_columns = []
    idx = 0
    for subset in possible_subsets.keys():
        if cur_dataset not in possible_subsets[subset]:
            continue
        dataframes = []
        labels = []
        for dataset in possible_subsets[subset]:
            if dataset == cur_dataset:
                continue
            X, y = prepare_data(dic_panel[dataset], subset)
            dataframes.append(X)
            labels.append(y)
        betas, loss, columns = k_fold_validation(dataframes, labels)
        if loss < best_loss:
            best_loss = loss
            best_betas = betas
            best_columns = columns
        idx += 1
    return (best_loss, best_betas, best_columns)


def generate_dropout_samples(panel, betas, dataset, beta_columns, dataset_columns):
    # Could be made faster if needed
    dropout_samples = []
    for _ in range(parameters.num_samples):
        dropout_sample = []
        for i, features in enumerate(panel): 
            dropout = stats.bernoulli.rvs(p=compute_dropout_prob(betas, beta_columns, dataset_columns, features))
            if dropout:
                dropout_sample.append(i)
        dropout_samples.append(dropout_sample)
    return dropout_samples

def compute_quotas(panel, columns):
    quotas = {}
    for fv in parameters.offsets.keys():
        if fv[0] in columns:
            quotas[fv] = panel[panel[fv[0]] == fv[1]].shape[0]
    return quotas


def opt_l1(quotas, panel, pool, dropout_samples, alt_budget):
    prob = Model(sense=MINIMIZE)
    pool = pool[:30]
    # Variables
    x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in range(pool.shape[0])}
    y = {(i, j): prob.add_var(name=f"y_{i}_{j}", var_type=BINARY) for i in range(pool.shape[0]) for j in range(len(dropout_samples))}
    z = {(feature, value, j): prob.add_var(name=f"z_{feature}_{value}_{j}", var_type='I', lb=0) for (feature, value) in quotas.keys() for j in range(len(dropout_samples))}
    t = prob.add_var(name="obj", var_type='CONTINUOUS', lb=0)
    # Objective
    prob.objective = t

    # Constraints
    prob.add_constr(xsum(x[i] for i in range(pool.shape[0])) <= alt_budget)
    prob.add_constr(t >= (xsum( z[(feature, value, j)] / (quotas[(feature, value)] + 1) for (feature, value) in quotas.keys() for j in range(len(dropout_samples)))))
    prob.add_constr(t <= (xsum( z[(feature, value, j)] / (quotas[(feature, value)] + 1) for (feature, value) in quotas.keys() for j in range(len(dropout_samples)))))

    for j in range(len(dropout_samples)):
        for i in range(pool.shape[0]):
            prob.add_constr(y[(i, j)] <= x[i])
        dropouts = panel.iloc[dropout_samples[j]]
        for (feature, value) in quotas.keys():
            num_agents_dropped_out_with_value = len(dropouts[dropouts[feature] == value])
            prob.add_constr((xsum(y[(i, j)] for i in range(pool.shape[0]) if pool.iloc[i][feature] == value)) <= num_agents_dropped_out_with_value + z[(feature, value, j)])
            prob.add_constr((xsum(y[(i, j)] for i in range(pool.shape[0]) if pool.iloc[i][feature] == value)) >= num_agents_dropped_out_with_value - z[(feature, value, j)])

    # Print the model (objective and constraints)
    # Model.write(prob, "model.lp")
    
    # Solve the problem
    prob.optimize()
    # Print variable values after optimization
    for v in prob.vars:
        print(f"{v.name} : {v.x}")

    # get the solution
    alt_set = [i for i in range(pool.shape[0]) if x[i].x >= 0.99]
    est_l1_score = prob.objective_value

    return alt_set, est_l1_score / len(dropout_samples)

def opt_01(quotas, panel, pool, dropout_samples, alt_budget):
    prob = Model(sense=MINIMIZE)
    pool = pool[:30]
    # Variables
    x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in range(pool.shape[0])}
    y = {(i, j): prob.add_var(name=f"y_{i}_{j}", var_type=BINARY) for i in range(pool.shape[0]) for j in range(len(dropout_samples))}
    t = {j: prob.add_var(name=f"t_{j}", var_type='BINARY') for j in range(len(dropout_samples))}
    # Objective
    prob.objective = xsum(t[j] for j in range(len(dropout_samples)))

    # Constraints
    prob.add_constr(xsum(x[i] for i in range(pool.shape[0])) <= alt_budget)

    for j in range(len(dropout_samples)):
        for i in range(pool.shape[0]):
            prob.add_constr(y[(i, j)] <= x[i])
        dropouts = panel.iloc[dropout_samples[j]]
        for (feature, value) in quotas.keys():
            num_agents_dropped_out_with_value = len(dropouts[dropouts[feature] == value])
            prob.add_constr((xsum(y[(i, j)] for i in range(pool.shape[0]) if pool.iloc[i][feature] == value)) - num_agents_dropped_out_with_value <= t[j]*1000)
            prob.add_constr(num_agents_dropped_out_with_value - (xsum(y[(i, j)] for i in range(pool.shape[0]) if pool.iloc[i][feature] == value)) <= t[j]*1000)

    # Print the model (objective and constraints)
    # Model.write(prob, "model.lp")
    
    # Solve the problem
    prob.optimize()
    # Print variable values after optimization
    for v in prob.vars:
        print(f"{v.name} : {v.x}")
    
    # get the solution
    alt_set = [i for i in range(pool.shape[0]) if x[i].x is not None and x[i].x >= 0.99]

    return alt_set, -1


def get_L1_alternates_set(dataset, panel, pool, betas, beta_columns):
    dataset_columns = [col for col, bit in zip(panel.columns, parameters.dataset_features[dataset]) if bit == '1']
    dataset_columns = dataset_columns[2:-1]
    og_panel = panel
    panel = panel[dataset_columns]
    pool = pool[dataset_columns]
    pool = pool.dropna()
    panel_df = panel
    pool_df = pool
    panel = np.array(panel)
    pool = np.array(pool)

    dropout_samples = generate_dropout_samples(panel, betas, dataset, beta_columns, dataset_columns)
    quotas = compute_quotas(panel_df, dataset_columns)

    # alternates_total = 0
    # for d in dropout_samples:
    #     alternates_total += len(d)
    alt_med = panel_df[og_panel['STATUS'] != 'Selected'].shape[0]

    return opt_l1(quotas, og_panel, pool_df, dropout_samples, alt_med)

def get_01_alternates_set(dataset, panel, pool, betas, beta_columns):
    dataset_columns = [col for col, bit in zip(panel.columns, parameters.dataset_features[dataset]) if bit == '1']
    dataset_columns = dataset_columns[2:-1]
    og_panel = panel
    panel = panel[dataset_columns]
    pool = pool[dataset_columns]
    pool = pool.dropna()
    panel_df = panel
    pool_df = pool
    panel = np.array(panel)
    pool = np.array(pool)

    dropout_samples = generate_dropout_samples(panel, betas, dataset, beta_columns, dataset_columns)
    quotas = compute_quotas(panel_df, dataset_columns)

    alt_med = panel_df[og_panel['STATUS'] != 'Selected'].shape[0]

    return opt_01(quotas, og_panel, pool_df, dropout_samples, alt_med)

def get_random_alternate_set(dataset, dic_pool, num_alternates):
    n = dic_pool[dataset].shape[0]
    samples = np.random.choice(range(0, n-1), size=num_alternates, replace=False)
    return samples

def alternates_real_loss(alternates, panel, pool, dataset):
    def generate_subsets(n):
        for i in range(0, 2**n):
            yield [int(bit) for bit in f"{i:0{n}b}"]
    dataset_columns = [col for col, bit in zip(panel.columns, parameters.dataset_features[dataset]) if bit == '1']
    dataset_columns = dataset_columns[2:-1]
    pool = pool[dataset_columns]
    alternates = pool.iloc[alternates]
    dropped = panel[panel['STATUS'] != 'Selected']
    panel = panel[dataset_columns]
    
    if dropped.empty:
        return 0
    loss = 10000
    for subset in generate_subsets(len(alternates)):
        lst = []
        for i, elt in enumerate(subset):
            if elt == 1:
                lst.append(i)
        cur_alt = alternates.iloc[lst]
        cur_loss = 0.0
        for fv in parameters.offsets.keys():
            if fv[0] not in dataset_columns:
                continue
            kfv = panel[panel[fv[0]] == fv[1]].shape[0]
            if kfv == 0:
                continue
            dif = abs(cur_alt[cur_alt[fv[0]] == fv[1]].shape[0] - dropped[dropped[fv[0]] == fv[1]].shape[0])
            cur_loss += (dif / kfv)
        loss = min(loss, cur_loss)
    return loss

# def read_and_seperate_data_new():
#     file_path1 = "Deschutes_2024_cleaned.csv"  # Replace with your CSV file path
#     file_path2 = "Eugene_2020_cleaned.csv"  # Replace with your CSV file path
#     file_path3 = "Petalume_2022_cleaned.csv"  # Replace with your CSV file path
    
#     data1 = pd.read_csv(file_path1)
#     data2 = pd.read_csv(file_path2)
#     data3 = pd.read_csv(file_path3)
    
#     data_panel_1 = data1[data['Dropped'].isin(["Selected", "Selected, dropped out"])]
#     data_panel_2  = data2[data['STATUS'].isin(["Not selected"])]
#     data_panel_3  = data2[data['STATUS'].isin(["Not selected"])]

#     unique_categories = data['DATA_ID'].unique()

#     unique_categories = unique_categories.tolist()

#     dic_panel = {}
#     dic_pool = {}

#     for category in unique_categories:
#         subset = data_per_group_panel[data_per_group_panel['DATA_ID'] == category]
#         dic_panel[category] = subset
    
#     for category in unique_categories:
#         subset = data_per_group_pool[data_per_group_pool['DATA_ID'] == category]
#         dic_pool[category] = subset
    
#     return dic_panel, dic_pool, unique_categories