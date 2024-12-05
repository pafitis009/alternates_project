import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds

# Load and preprocess the data
file_path = "cleaned_anonymized_data.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

if 'Number' in data.columns:
    data = data.drop(columns=['Number'])

data = data.fillna(-1)

# Filter the relevant rows and preprocess
filtered_data = data[data['STATUS'].isin(["Selected", "Selected, dropped out"])]
filtered_data = filtered_data.drop(columns=['DATA_ID'])  # Drop unnecessary columns

# Map the STATUS column to binary values
filtered_data['y'] = filtered_data['STATUS'].apply(lambda x: 1 if x == "Selected" else 0)
filtered_data = filtered_data.drop(columns=['STATUS'])  # Drop the STATUS column after mapping

# Prepare the feature matrix and target vector
feature_matrix = filtered_data.drop(columns=['y']).values
target_vector = filtered_data['y'].values

# Calculate the total number of unique values per feature
unique_values_per_feature = [set({value for value in feature_matrix[:, i] if value != -1}) for i in range(feature_matrix.shape[1])]
num_parameters = 1 + sum([len(x) for x in unique_values_per_feature])  # Beta dimension
# CHECKPOINT

feature_value_map = {}
value_feature_map = {}
offset = 1  # Start after beta_0
for i, unique_values in enumerate(unique_values_per_feature):
    for value in unique_values:  # Adjust this loop if features aren't integer-encoded
        feature_value_map[(i, value)] = offset
        value_feature_map[offset] = (i, value)
        offset += 1

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

# Initial beta values
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
print(unique_values_per_feature)
for i, val in enumerate(optimal_beta):
    if i == 0:
        print("Beta 0: " + str(val))
    else:
        if val == 0.99999:
            print(value_feature_map[i], 1, cnt_1[i], cnt_0[i])
        else:
            print(value_feature_map[i], val, cnt_1[i], cnt_0[i])