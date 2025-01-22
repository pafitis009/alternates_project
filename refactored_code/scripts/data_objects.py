import numpy as np
import pandas as pd
from mip import Model, xsum, MINIMIZE, BINARY, CONTINUOUS, INTEGER, MAXIMIZE
from scipy.optimize import minimize, Bounds
import os

class Instance:
    def __init__(self, name, num_train_samples=100, logging=True, file_stub="../"):
        self.name = name
        self.file_stub = file_stub
        self.logging = logging
        self.num_train_samples = num_train_samples
        self.people_df, self.people, self.panel, self.pool, self.dropouts, self.beta_train_stayin, self.quotas, self.alternates = self.read_in_data()
        self.features = list(filter(lambda x: x not in ["Code", "Dropped", "Dropped_Inclusive", "Initially Selected", "Alternate"], list(self.people_df.columns)))
        self.dropout_probs = None
        
    def read_in_data(self):
        data_folder = f"{self.file_stub}/data/{self.name}"
        if not os.path.exists(data_folder):
            print(f"No data folder found for {self.name}.")
            return None
        logging_folder = f"{self.file_stub}/logging/{self.name}"
        if not os.path.exists(logging_folder):
            os.makedirs(logging_folder)
            
        people_df = pd.read_csv(f"{data_folder}/{self.name}_cleaned.csv")
        
        categories = list(filter(lambda x: x not in ["Code", "Dropped", "Dropped_Inclusive", "Initially Selected", "Alternate"], list(people_df.columns)))
        people = {}
        panel_df = people_df[people_df["Initially Selected"] == "YES"]
        panel = list(panel_df["Code"])
        pool_df = people_df[people_df["Initially Selected"] != "YES"]
        pool = list(pool_df["Code"])
        dropout_df = people_df[people_df["Dropped"] == "YES"]
        dropouts = list(dropout_df["Code"])
        beta_train_stayin_df = people_df[people_df["Dropped"] == "NO"]
        beta_train_stayin = list(beta_train_stayin_df["Code"])
        alternates_df = people_df[people_df["Alternate"] == "YES"]
        alternates = list(alternates_df["Code"])
        
        # fill people
        for _, row in people_df.iterrows():
            agent_id = row['Code']
            people[agent_id] = {}
            for category in categories:
                people[agent_id][category] = row[category]
        
        quotas_file_path = f"{data_folder}/{self.name}_quotas.csv"
        quotas = {}
        if os.path.exists(quotas_file_path):
            quotas_df = pd.read_csv(quotas_file_path)
            # fill categories
            for feature in list(quotas_df['category'].unique()):
                quotas[feature] = {}
                for value in list(quotas_df[quotas_df['category']==feature]['name'].unique()):
                    lower_quota = int(quotas_df[(quotas_df['category']==feature) & (quotas_df['name']==value)]['min'].values[0])
                    upper_quota = int(quotas_df[(quotas_df['category']==feature) & (quotas_df['name']==value)]['max'].values[0])
                    quotas[feature][value] = {"min":lower_quota, "max":upper_quota}
                    num_people_on_panel = panel_df[panel_df[feature] == value].shape[0]
        else:
            print(f"No quotas file found for {self.name}. Constructing tight quotas based on initially selected panel.")
            inferred_quotas_file = f"{logging_folder}/{self.name}_inferred_quotas.csv"
            with open(inferred_quotas_file, 'w') as f:
                f.write("category,name,min,max\n")
                for feature in categories:
                    quotas[feature] = {}
                    for value in people_df[feature].unique():
                        num_people_on_panel = panel_df[panel_df[feature] == value].shape[0]
                        quotas[feature][value] = {"min":num_people_on_panel, "max":num_people_on_panel}
                        f.write(f"{feature},{value},{quotas[feature][value]['min']},{quotas[feature][value]['max']}\n")
                        
        return people_df, people, panel, pool, dropouts, beta_train_stayin, quotas, alternates
    
    def compute_dropout_probabilities(self, betas):
        # given a dictionary of betas, calculate the dropout probability for every person and write it to a file
        dropout_probs = {}
        for agent_id in self.panel:
            p_stay = betas["init"]["0"]
            for category in betas:
                if category in self.features:
                    p_stay *= betas[category][self.people[agent_id][category]]
            dropout_probs[agent_id] = 1-p_stay # probability of dropping out
        dropout_file = f"{self.file_stub}/logging/{self.name}/{self.name}_dropout_probs.csv"
        with open(dropout_file, 'w') as f:
            f.write("code,dropout_prob\n")
            for agent_id in dropout_probs:
                f.write(f"{agent_id},{dropout_probs[agent_id]}\n")
        self.dropout_probs = dropout_probs
        return dropout_probs
    
    def generate_dropout_samples(self, num_samples):
        # definitely can be written more smartly (e.g. using binomial for each agent)
        if self.dropout_probs is None:
            print("Need to calculate dropout probabilities first.")
            return None
        
        dropout_samples = []
        for _ in range(num_samples):
            dropout_sample = []
            for agent_id in self.panel: 
                dropout = np.random.binomial(1, self.dropout_probs[agent_id])
                if dropout:
                    dropout_sample.append(agent_id)
            dropout_samples.append(dropout_sample)
        return dropout_samples
    
    def opt_binary(self, alt_budget):
        num_samples = self.num_train_samples
        dropout_samples = self.generate_dropout_samples(num_samples)
        # print(f'dropout_samples: {dropout_samples}')
        prob = Model(sense=MINIMIZE)
        prob.verbose = 0
        
        # Variables
        x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in self.pool}
        y = {(i, j): prob.add_var(name=f"y_{i}_{j}", var_type=BINARY) for i in self.pool for j in range(num_samples)}
        z = {j: prob.add_var(name=f"z_{j}", var_type=BINARY) for j in range(num_samples)}

        prob.objective = xsum( z[j] for j in range(num_samples))

        # Constraints
        prob.add_constr(xsum(x[i] for i in self.pool) <= alt_budget)
        
        for j, dropout_ids in enumerate(dropout_samples):
            for i in self.pool:
                prob.add_constr(y[(i, j)] <= x[i])
            
            prob.add_constr(xsum(y[(i,j)] for i in self.pool) <= len(dropout_ids))

            for feature,value_quotas in self.quotas.items():
                for value in value_quotas:
                    lower_quota = value_quotas[value]["min"]
                    upper_quota = value_quotas[value]["max"]
                    num_agents_dropped_out_with_value = sum([1 for agent_id in dropout_ids if self.people[agent_id][feature] == value])
                    num_agents_on_panel_with_value = sum([1 for agent_id in self.panel if self.people[agent_id][feature] == value])
                    # lower and upper quotas for our replacement set (considering panl & drop outs)
                    adjusted_lower_quota = lower_quota - num_agents_on_panel_with_value + num_agents_dropped_out_with_value
                    adjusted_upper_quota = upper_quota - num_agents_on_panel_with_value + num_agents_dropped_out_with_value
                    prob.add_constr(adjusted_lower_quota - xsum(y[(i, j)] for i in self.pool if self.people[i][feature] == value) <= z[j] * adjusted_lower_quota)
                    prob.add_constr(xsum(y[(i, j)] for i in self.pool if self.people[i][feature] == value) - adjusted_upper_quota <= z[j] * len(dropout_ids))
                
        prob.optimize()
        alt_set = [i for i in self.pool if x[i].x >= 0.99]
        if self.logging:
            self.alt_set_logger("opt_binary", alt_set, alt_budget, num_samples, prob.objective_value)
        return alt_set
    
    def opt_l1(self, alt_budget):
        num_samples = self.num_train_samples
        dropout_samples = self.generate_dropout_samples(num_samples)
        prob = Model(sense=MINIMIZE)
        prob.verbose = 0
        # Variables
        x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in self.pool}
        y = {(i, j): prob.add_var(name=f"y_{i}_{j}", var_type=BINARY) for i in self.pool for j in range(num_samples)}
        z = {(feature, value, j): prob.add_var(name=f"z_{feature}_{value}_{j}", var_type=INTEGER, lb=0) for feature in self.quotas for value in self.quotas[feature] for j in range(num_samples)}

        prob.objective = xsum( z[(feature, value, j)] / float(max(self.quotas[feature][value]["max"], 1)) for feature in self.quotas for value in self.quotas[feature] for j in range(num_samples))

        # Constraints
        prob.add_constr(xsum(x[i] for i in self.pool) <= alt_budget)
        panel_df = self.people_df[self.people_df["Code"].isin(self.panel)]
        
        for j, dropout_ids in enumerate(dropout_samples):
            for i in self.pool:
                prob.add_constr(y[(i, j)] <= x[i])
            
            prob.add_constr(xsum(y[(i,j)] for i in self.pool) <= len(dropout_ids))
            dropout_df = self.people_df[self.people_df["Code"].isin(dropout_ids)]
            assert dropout_df.shape[0] == len(dropout_ids)
            
            for feature,value_quotas in self.quotas.items():
                for value in value_quotas:
                    lower_quota = value_quotas[value]["min"]
                    upper_quota = value_quotas[value]["max"]
                    num_agents_dropped_out_with_value = dropout_df[dropout_df[feature] == value].shape[0]
                    num_agents_on_panel_with_value = panel_df[panel_df[feature] == value].shape[0]
                    # lower and upper quotas for our replacement set (considering panl & drop outs)
                    adjusted_lower_quota = lower_quota - num_agents_on_panel_with_value + num_agents_dropped_out_with_value
                    adjusted_upper_quota = upper_quota - num_agents_on_panel_with_value + num_agents_dropped_out_with_value
                    
                    prob.add_constr(adjusted_lower_quota - (xsum(y[(i, j)] for i in self.pool if self.people[i][feature] == value)) <= z[(feature, value, j)])
                    prob.add_constr((xsum(y[(i, j)] for i in self.pool if self.people[i][feature] == value)) - adjusted_upper_quota <= z[(feature, value, j)])

        prob.optimize()
        alt_set = [i for i in self.pool if x[i].x >= 0.99]
        
        if self.logging:
            self.alt_set_logger("opt_l1", alt_set, alt_budget, num_samples, prob.objective_value)
        
        return alt_set
    
    def l1_loss(self, alt_set, dropout_sets):
        losses = []
        for dropout_set in dropout_sets:
            prob = Model(sense=MINIMIZE)
            prob.verbose = 0
            
            x = {i: prob.add_var(name=f"x_{i}", var_type=BINARY) for i in alt_set}
            z = {(feature, value): prob.add_var(name=f"z_{feature}_{value}", var_type=INTEGER, lb=0) for feature in self.quotas for value in self.quotas[feature]}

            prob.objective = xsum(z[(feature, value)] / float(max(self.quotas[feature][value]['max'], 1)) for feature in self.quotas for value in self.quotas[feature])

            # Constraints
            prob.add_constr(xsum(x[i] for i in alt_set) <= len(dropout_set))
            dropout_df = self.people_df[self.people_df["Code"].isin(dropout_set)]
            panel_df = self.people_df[self.people_df["Code"].isin(self.panel)]
            for feature,value_quotas in self.quotas.items():
                for value in value_quotas:
                    lower_quota = value_quotas[value]["min"]
                    upper_quota = value_quotas[value]["max"]
                    
                    num_agents_dropped_out_with_value = dropout_df[dropout_df[feature] == value].shape[0]
                    num_agents_on_panel_with_value = panel_df[panel_df[feature] == value].shape[0]
                    # lower and upper quotas for our replacement set (considering panl & drop outs)
                    adjusted_lower_quota = lower_quota - num_agents_on_panel_with_value + num_agents_dropped_out_with_value
                    adjusted_upper_quota = upper_quota - num_agents_on_panel_with_value + num_agents_dropped_out_with_value
                    
                    # how much are we off on the adjusted lower and upper quotas
                    prob.add_constr(adjusted_lower_quota - xsum(x[i] for i in alt_set if self.people[i][feature] == value) <= z[(feature, value)])
                    prob.add_constr(xsum(x[i] for i in alt_set if self.people[i][feature] == value) - adjusted_upper_quota <= z[(feature, value)])

            prob.optimize()
            score = prob.objective_value
            selected_alternates = [i for i in alt_set if x[i].x >= 0.99]
            # print(f"Selected alternates for dropout set {dropout_set}: {selected_alternates}")
            # violated_quotas = [(feature, value, z[(feature, value)].x) for feature in self.quotas for value in self.quotas[feature] if z[(feature, value)].x > 0]
            # print(f"Violated quotas for dropout set {dropout_set}: {violated_quotas}")
            losses.append(score)

        return losses
    
    def alt_set_logger(self, selection_function, alt_set, alt_budget, num_samples, obj_func_value):
        logger_file = f"{self.file_stub}/logging/{self.name}/{self.name}_alt_set_logger.csv"
        if not os.path.exists(logger_file):
            with open(logger_file, 'w') as f:
                f.write("time,selection_function,num_alternates,num_training_samples,alt_set,obj_func_value\n")
        with open(logger_file, 'a') as f:
            f.write(f"{pd.Timestamp.now()},{selection_function},{alt_budget},{num_samples},{alt_set},{obj_func_value}\n")

    def greedy_alt_set(self, l1_dist = True):
        # orders panel members from highest to lowest dropout probability and selects the ``closest'' remaining pool member to replace them
        if self.dropout_probs is None:
            print("Need to calculate dropout probabilities first.")
            return None
        sorted_dropout_probs = sorted(self.dropout_probs.keys(), key=lambda k: self.dropout_probs[k], reverse=True)
        alt_set = []
        remaining_pool = self.pool.copy()
        for agent_id in sorted_dropout_probs:
            closest_pool_member = min(remaining_pool, key=lambda pool_id: self.person_distance(agent_id, pool_id))
            alt_set.append(closest_pool_member)
            remaining_pool.remove(closest_pool_member)
        
        if self.logging:
            self.alt_set_logger("greedy", alt_set, len(alt_set), -1, -1)
        return alt_set

    def person_distance(self, person1, person2, use_l1_dist=True):
        l1_dist = 0.0
        ham_dist = 0.0
        for feature in self.features:
            value1 = self.people[person1][feature]
            value2 = self.people[person2][feature]
            if value1 != value2:
                ham_dist += 1
                l1_dist += ((1.0/self.quotas[feature][value1]["max"]) + (1.0/self.quotas[feature][value2]["max"]))
        if use_l1_dist:
            return l1_dist
        else:
            return ham_dist
        
class BetaLearner:
    def __init__(self, instances, file_stub="../logging/betas"):
        self.instances = instances # list of Instance objects from which to learn the betas
        self.file_stub = file_stub
        if not os.path.exists(file_stub):
            os.makedirs(file_stub)
    
    def learn_betas(self, feature_list):
        # learn betas based on every instance that has all fatures in feature_list (and only use those features)
        useful_instances = []
        feature_values = {feature: set() for feature in feature_list}
        num_params = 1
        
        for instance in self.instances:
            if all(feature in instance.features for feature in feature_list):
                useful_instances.append(instance)
                for feature in feature_list:
                    feature_values[feature].update(set(instance.people_df[feature].unique()))
        num_params += sum([len(feature_values[feature]) for feature in feature_list])
        print(f"Learning betas based on instances: {[instance.name for instance in useful_instances]}.")
        
        beta_init = np.random.rand(num_params)
        fv_indices = {feature: {} for feature in feature_list}
        index = 1
        for feature in feature_list:
            for value in feature_values[feature]:
                fv_indices[feature][value] = index
                index += 1
        
        def log_likelihood(beta):
            ll = 0
            for instance in useful_instances:
                for agent_id in instance.dropouts:
                    p_stay = beta[0]
                    for feature in feature_list:
                        p_stay *= beta[fv_indices[feature][instance.people[agent_id][feature]]]
                    p_drop = 1-p_stay
                    ll += np.log(p_drop)
                for agent_id in instance.beta_train_stayin:
                    p_stay = beta[0]
                    for feature in feature_list:
                        p_stay *= beta[fv_indices[feature][instance.people[agent_id][feature]]]
                    ll += np.log(p_stay)
            return -ll

        bounds = Bounds([0.0001] * num_params, [0.9999] * num_params)
        result = minimize(
            log_likelihood,
            beta_init,
            args=(),
            method='L-BFGS-B',
            bounds = bounds,
        )
        learned_betas = result.x
        betas = {feature: {} for feature in feature_list}
        betas["init"] = {"0" : learned_betas[0]}
        for feature in feature_list:
            for value in feature_values[feature]:
                betas[feature][value] = learned_betas[fv_indices[feature][value]]
        
        # Save the betas to a file
        betas_file = f"{self.file_stub}/betas_{'_'.join([feature[:3] for feature in feature_list])}_{'_'.join([instance.name for instance in useful_instances])}.csv"
        with open(betas_file, 'w') as f:
            f.write("feature,value,beta\n")
            f.write(f"init,0,{betas['init']['0']}\n")
            for feature in feature_list:
                for value in betas[feature]:
                    f.write(f"{feature},{value},{betas[feature][value]}\n")
        print(f"Betas saved to {betas_file}")
        return betas
        
    
    