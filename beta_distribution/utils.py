import numpy as np
import pandas as pd
import parameters
import random
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

from mip import Model, xsum, BINARY, MAXIMIZE, MINIMIZE
from itertools import combinations
from scipy.optimize import minimize
from scipy.optimize import Bounds

def read_and_seperate_data():
    file_path1 = "new_data/Deschutes_2024_cleaned.csv"  
    file_path2 = "new_data/Eugene_2020_cleaned.csv"  # no dropped_inclusive
    file_path3 = "new_data/Petaluma_2022_cleaned.csv"  
    
    data1 = pd.read_csv(file_path1)
    data2 = pd.read_csv(file_path2)
    data3 = pd.read_csv(file_path3)

    data1 = data1.sort_index(axis=1)
    data2 = data2.sort_index(axis=1)
    data3 = data3.sort_index(axis=1)

    data_panel_1  = data1[data1['Dropped'].isin(["YES", "NO"])]
    data_panel_2  = data2[data2['Dropped'].isin(["YES", "NO"])]
    data_panel_3  = data3[data3['Dropped'].isin(["YES", "NO"])]

    condition1 = (
        (~data1["Dropped"].isin(["YES", "NO"])) |
        (~data1["Dropped_Inclusive"].isin(["YES", "NO"])) |
        (~data1["Initially Selected"].isin(["YES", "NO"]))
    )

    condition2 = (
        (~data2["Dropped"].isin(["YES", "NO"])) |
        (~data2["Initially Selected"].isin(["YES", "NO"]))
    )

    condition3 = (
        (~data3["Dropped"].isin(["YES", "NO"])) |
        (~data3["Dropped_Inclusive"].isin(["YES", "NO"])) |
        (~data3["Initially Selected"].isin(["YES", "NO"]))
    )
    
    data_pool_1 = data1[condition1]
    data_pool_2 = data2[condition2]
    data_pool_3 = data3[condition3]

    alternates1 = data1[data1['Alternate'] == 'YES']
    alternates2 = data2[data2['Alternate'] == 'YES']
    alternates3 = data3[data3['Alternate'] == 'YES']

    alternates_array1 = data_pool_1["Alternate"].to_numpy()
    alternates_array2 = data_pool_2["Alternate"].to_numpy()
    alternates_array3 = data_pool_3["Alternate"].to_numpy()
    
    alternates_idx1 = np.where(alternates_array1 == "YES")[0]
    alternates_idx2 = np.where(alternates_array2 == "YES")[0]
    alternates_idx3 = np.where(alternates_array3 == "YES")[0]


    return [data_panel_1, data_panel_2, data_panel_3] , [data_pool_1, data_pool_2, data_pool_3], [(alternates1, alternates_idx1), (alternates2, alternates_idx2),(alternates3, alternates_idx3)]

def generate_subsets():
    for i in range(0, 2**parameters.features):
        if i == 0:
            continue
        yield [int(bit) for bit in f"{i:0{parameters.features}b}"]

def get_beta_index(column, v_i, columns, idx):
    columns.sort()
    j = 0
    idx = 1
    while j < len(columns):
        if columns[j] != column:
            idx += parameters.feature_values[columns[j]]
        else:
            if pd.isna(v_i):
                v_i = 'nan'
            idx += parameters.offsets[(columns[j], v_i)]
        j += 1
    return idx

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
                beta_sum *= beta[get_beta_index(columns[i], v_i, columns, 0)]

            if y_v == 0:
                log_likelihood_value += np.log(beta_sum)
            else:
                log_likelihood_value += np.log(1 - beta_sum)
        
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
    # print("Optimal beta:", optimal_beta)
    return result.fun,optimal_beta

def compute_loss_and_betas(subset, panels):
    y = []
    X = []
    for i in range(len(panels)):
        y_cur = pd.DataFrame({
            "Dropped": panels[i]["Dropped"].apply(lambda x: 1 if x == "YES" else 0)
        })
        
        panel_cur = panels[i].drop(columns=["Initially Selected", "Alternate", "Code", "Dropped"])

        if "Dropped_Inclusive" in panel_cur.columns:
            panel_cur = panel_cur.drop(columns=["Dropped_Inclusive"])

        if "Disability" in panel_cur.columns:
            panel_cur = panel_cur.drop(columns=["Disability"])
        
        if "Party" in panel_cur.columns:
            panel_cur = panel_cur.drop(columns=["Party"])
        
        X_cur = panel_cur.loc[:, [bool(val) for val in subset]]
        y.append(y_cur)
        X.append(X_cur)
    
    X_combined = pd.concat(X, ignore_index=True)
    y_combined = pd.concat(y, ignore_index=True)
    cols = list(X_combined.columns)
    loss, betas = compute_betas(np.array(X_combined), np.array(y_combined), cols)
    return loss, betas, cols

def generate_dropout_samples(panel, betas, columns):
    # Could be made faster if needed
    dropout_samples = []
    for _ in range(parameters.num_samples):
        dropout_sample = []
        for i in range(panel.shape[0]): 
            p = betas[0]
            for j in range(panel.shape[1]):
                p *= betas[get_beta_index(columns[j], panel[i][j], columns, 1)]
            if random.random() < p:
                dropout_sample.append(i)
        dropout_samples.append(dropout_sample)
    return dropout_samples

def compute_quotas(panel, columns):
    quotas = {}
    for fv in parameters.offsets.keys():
        if fv[0] in columns:
            quotas[fv] = panel[panel[fv[0]] == fv[1]].shape[0]
    return quotas

def opt_l1(quotas, panel, pool, dropout_samples, alternates, columns):
    prob = Model(sense=MINIMIZE)
    # Variables
    x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in range(pool.shape[0])}
    y = {(i, j): prob.add_var(name=f"y_{i}_{j}", var_type=BINARY) for i in range(pool.shape[0]) for j in range(len(dropout_samples))}
    z = {(feature, value, j): prob.add_var(name=f"z_{feature}_{value}_{j}", var_type='I', lb=0) for (feature, value) in quotas.keys() for j in range(len(dropout_samples))}
    t = prob.add_var(name="obj", var_type='CONTINUOUS', lb=0)
    # Objective
    prob.objective = t

    # Constraints
    prob.add_constr(xsum(x[i] for i in range(pool.shape[0])) <= alternates)
    prob.add_constr(t >= (xsum( z[(feature, value, j)] / (quotas[(feature, value)] + 1) for (feature, value) in quotas.keys() for j in range(len(dropout_samples)))))
    prob.add_constr(t <= (xsum( z[(feature, value, j)] / (quotas[(feature, value)] + 1) for (feature, value) in quotas.keys() for j in range(len(dropout_samples)))))

    for j in range(len(dropout_samples)):
        for i in range(pool.shape[0]):
            prob.add_constr(y[(i, j)] <= x[i])
        dropouts = panel.iloc[dropout_samples[j]]
        prob.add_constr(xsum(y[(i,j)] for i in range(pool.shape[0])) == min(len(dropout_samples[j]), alternates))
        for (feature, value) in quotas.keys():
            num_agents_dropped_out_with_value = len(dropouts[dropouts[feature] == value])
            prob.add_constr((xsum(y[(i, j)] for i in range(pool.shape[0]) if pool.iloc[i][feature] == value)) <= num_agents_dropped_out_with_value + z[(feature, value, j)])
            prob.add_constr((xsum(y[(i, j)] for i in range(pool.shape[0]) if pool.iloc[i][feature] == value)) >= num_agents_dropped_out_with_value - z[(feature, value, j)])

    prob.optimize()
    # Print variable values after optimization
    # for v in prob.vars:
    #     print(f"{v.name} : {v.x}")

    # get the solution
    alt_set = [i for i in range(pool.shape[0]) if x[i].x >= 0.99]

    return alt_set

def opt_01(quotas, panel, pool, dropout_samples, alternates, columns):
    prob = Model(sense=MINIMIZE)
    # Variables
    x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in range(pool.shape[0])}
    y = {(i, j): prob.add_var(name=f"y_{i}_{j}", var_type=BINARY) for i in range(pool.shape[0]) for j in range(len(dropout_samples))}
    t = {j: prob.add_var(name=f"t_{j}", var_type='BINARY') for j in range(len(dropout_samples))}
    # Objective
    prob.objective = xsum(t[j] for j in range(len(dropout_samples)))

    # Constraints
    prob.add_constr(xsum(x[i] for i in range(pool.shape[0])) <= alternates)

    for j in range(len(dropout_samples)):
        for i in range(pool.shape[0]):
            prob.add_constr(y[(i, j)] <= x[i])
        dropouts = panel.iloc[dropout_samples[j]]
        prob.add_constr(xsum(y[(i,j)] for i in range(pool.shape[0])) == min(len(dropout_samples[j]), alternates))
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

    return alt_set

def get_random_alternate_set(pool, num_alternates):
    n = pool.shape[0]
    samples = np.random.choice(range(0, n-1), size=num_alternates, replace=False)
    return samples

def get_L1_alternates_set(panel, quotas, pool, betas, columns, alternates):
    dropout_samples = generate_dropout_samples(np.array(panel[columns]), betas, columns)

    return opt_l1(quotas, panel, pool, dropout_samples, alternates, columns)

def get_01_alternates_set(panel, quotas, pool, betas, columns, alternates):
    dropout_samples = generate_dropout_samples(np.array(panel[columns]), betas, columns)

    return opt_01(quotas, panel, pool, dropout_samples, alternates, columns)


def calculate_alternate_set(panel, pool, quotas, alternates, benchmark, betas, columns):
    if benchmark == 0:
        return get_L1_alternates_set(panel, quotas, pool, betas, columns, alternates)
    elif benchmark == 1:
        return get_01_alternates_set(panel, quotas, pool, betas, columns, alternates)
    elif benchmark == 2:
        return get_random_alternate_set(pool, alternates)
    else:
        return []

def compute_dropouts(panel, betas, columns):
    dropout_sample = []
    for i in range(panel.shape[0]): 
        p = betas[0]
        for j in range(panel.shape[1]):
            p *= betas[get_beta_index(columns[j], panel[i][j], columns, 1)]
        if random.random() < p:
            dropout_sample.append(i)
    return dropout_sample

def compute_alt_set_loss(panel, pool, quotas, alternates_list, dropouts):
    prob = Model(sense=MINIMIZE)
    dropped = panel.iloc[dropouts]
    alternates = pool.iloc[alternates_list]
    
    x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in range(alternates.shape[0])}
    z = {(feature, value): prob.add_var(name=f"z_{feature}_{value}", var_type='I', lb=0) for (feature, value) in quotas.keys()}
    t = prob.add_var(name="obj", var_type='CONTINUOUS', lb=0)
    # Objective
    prob.objective = t

    # Constraints
    prob.add_constr(xsum(x[i] for i in range(alternates.shape[0])) == min(dropped.shape[0], alternates.shape[0]))
    prob.add_constr(t >= (xsum(z[(feature, value)] / (quotas[(feature, value)] + 1) for (feature, value) in quotas.keys())))
    prob.add_constr(t <= (xsum(z[(feature, value)] / (quotas[(feature, value)] + 1) for (feature, value) in quotas.keys())))
    for (feature, value) in quotas.keys():
        num_agents_dropped_out_with_value = dropped[dropped[feature] == value].shape[0]
        prob.add_constr((xsum(x[i] for i in range(alternates.shape[0]) if alternates.iloc[i][feature] == value)) <= num_agents_dropped_out_with_value + z[(feature, value)])
        prob.add_constr((xsum(x[i] for i in range(alternates.shape[0]) if alternates.iloc[i][feature] == value)) >= num_agents_dropped_out_with_value - z[(feature, value)])
    prob.optimize()
    # Print variable values after optimization
    # for v in prob.vars:
    #     print(f"{v.name} : {v.x}")

    # get the solution
    score = prob.objective_value

    return score

def calculate_loss_sets(alt_sets, panel, pool, quotas, betas, columns):
    dropouts = compute_dropouts(np.array(panel[columns]), betas, columns)
    losses = []
    for alt_set in alt_sets:
        losses.append(compute_alt_set_loss(panel, pool, quotas, alt_set, dropouts))
    return losses

def plot_losses(losses, dataset, dataset_i):
    x = np.arange(len(parameters.alternates_numbers[dataset_i]))  # Create positions for the 10 bars
    width = 0.10  # Bar width (adjustable)

    _, ax = plt.subplots()
    coef = (1 - len(parameters.benchmarks))/2.0
    
    for i in range(len(parameters.benchmarks)):
        ax.bar(x - coef*width, [row[i] for row in losses], width, label=parameters.benchmarks[i], color=parameters.color[i])
        coef -= 1.0

    ax.set_xlabel('Number of Alternates')
    ax.set_ylabel('Loss')
    ax.set_xticks(x)
    ax.set_xticklabels([elt for elt in parameters.alternates_numbers[dataset_i]])

    # Add a legend
    ax.legend()
    plt.savefig("plots/stats_" + str(dataset) + ".png", dpi=300, bbox_inches="tight")
    # Show the plot
    plt.show()

