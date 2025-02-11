import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('data/cleaned_anonymized_data.csv')

# Get the unique values in the 'Dataset' column
unique_datasets = df['DATA_ID'].unique()

# # Iterate over each unique value in the 'Dataset' column
# for dataset in unique_datasets:
#     # Filter the DataFrame for the current dataset
#     dataset_df = df[df['DATA_ID'] == dataset]
    
#     # Get the columns with non-NaN values
#     non_nan_columns = dataset_df.dropna(axis=1, how='all').columns
    
#     # Print the dataset and its non-NaN columns
#     print(f"Dataset: {dataset}")
#     print("Columns with non-NaN values:", list(non_nan_columns))
#     print()
# Create an empty DataFrame to store the results
result_df = pd.DataFrame(index=df.columns, columns=unique_datasets)

# Iterate over each unique value in the 'DATA_ID' column
for dataset in unique_datasets:
    # Filter the DataFrame for the current dataset
    dataset_df = df[df['DATA_ID'] == dataset]
    
    # Check for non-NaN values for each column in the filtered DataFrame
    non_nan_counts = dataset_df.notna().sum().apply(lambda x: 1 if x > 0 else 0)
    
    # Store the counts in the result DataFrame
    result_df[dataset] = non_nan_counts

# Reorder the rows of result_df based on the number of 1s in that row (highest first)
# Drop the rows 'Unnamed: 0', 'STATUS', 'DATA_ID'
result_df = result_df.drop(['Unnamed: 0', 'STATUS', 'DATA_ID'], errors='ignore')

# Add a sum column to count the number of non-NaN values per row
result_df['sum'] = result_df.sum(axis=1)

# Sort the DataFrame based on the sum column and drop the sum column
result_df = result_df.sort_values(by='sum', ascending=False).drop(columns='sum')

# Add a sum row to count the number of 1s per column
result_df.loc['sum'] = result_df.sum()

# Sort the DataFrame columns based on the sum row and drop the sum row
result_df = result_df.loc[:, result_df.loc['sum'].sort_values(ascending=False).index].drop(index='sum')

# Print the whole resulting DataFrame without truncation
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result_df)