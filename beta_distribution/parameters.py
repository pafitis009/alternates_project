datasets = 3
features = 5
num_samples = 10
num_alternates = 5
plot_samples = 10
alternates_numbers = [[1, 4, 5, 10, 21], [2, 5, 6, 10, 15, 25], [2, 5, 6, 10, 14, 24]]
feature_values = {
    'Age Range': 7,
    'Disability': 2,
    'Housing Status': 3,
    'Gender': 3,
    'Party': 3,
    'Race/Ethnicity': 6,
    'Educational Attainment': 4,
}
offsets = {
    ('Age Range', '16-24'): 0, ('Age Range', '25-34'): 1, ('Age Range', '35-44'): 2, ('Age Range', '45-54'): 3, ('Age Range', '55-64'): 4, ('Age Range', '65-74'): 5, ('Age Range', '75_and_up'): 6,
    ('Disability', 'Yes'): 0, ('Disability', 'No'): 1,
    ('Housing Status', 'Own'): 0, ('Housing Status', 'Rent'): 1, ('Housing Status', 'Unhoused'): 2,
    ('Party', 'Dem'): 0, ('Party', 'Rep'): 1, ('Party', 'Another'): 2,
    ('Gender', 'Male'): 0, ('Gender', 'Female'): 1, ('Gender', 'Another gender identity'):2,
    ('Race/Ethnicity', 'White'): 0, ('Race/Ethnicity', 'Native_American'): 1, ('Race/Ethnicity', 'AAPI'): 2, ('Race/Ethnicity', 'Multiracial'): 3, ('Race/Ethnicity', 'Latinx'): 4, ('Race/Ethnicity', 'Black'): 5,
    ('Educational Attainment', 'Bachelors'): 0, ('Educational Attainment', 'High_school'): 1,  ('Educational Attainment', 'Some_schooling'): 2, ('Educational Attainment', 'Some_college'): 3
}

benchmarks = ["L1", "01", "Random", "Practitioners", "Empty", "Best"] # Add: Greedy
color = ["blue", "green", "red", "black", "orange"]