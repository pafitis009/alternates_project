from data_objects import Instance, BetaLearner
import seaborn as sns
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_losses_with_shaded_bands(alt_indices, upper_bound, lower_bound, losses, loss_labels, instance_name, k, filename):
    plt.figure(figsize=(10, 6))
    
    for label in loss_labels:
        means, stds = losses[label]
        plt.plot(alt_indices, means, label=label)
        plt.fill_between(
            alt_indices,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2
        )

    plt.axhline(y=upper_bound, color='r', linestyle='--', label='Empty Alternate Set')
    plt.axhline(y=lower_bound, color='b', linestyle='--', label='Pool Alternate Set')

    plt.title(f"{instance_name} Losses, k={k}")
    plt.xlabel("Alt Budget")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Modify x ticks to show fraction of k in LaTeX format
    xticks = [f"$\\frac{{k}}{{{k//x}}}$" if x != 0 else "0" for x in alt_indices]
    plt.xticks(alt_indices, xticks)
    
    plt.savefig(filename)
    plt.close()


def make_violin_plot(losses, labels, num_test_samples, title, plot_filename):
    type_labels = []
    for label in labels:
        type_labels += [label] * num_test_samples
    data = pd.DataFrame({'Losses': losses, 'Type': type_labels})
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Type', y='Losses', data=data)
    plt.ylim(0, data['Losses'].max() * 1.1)  # Start y-axis at 0 and add some padding at the top
    plt.title(title)
    plt.savefig(plot_filename)
    plt.close()
    
def greedy_test():
    Deschutes_2024 = Instance("Deschutes_2024")
    Eugene_2020 = Instance("Eugene_2020")
    Petaluma_2022 = Instance("Petaluma_2022")

    instance_list = [Deschutes_2024, Eugene_2020, Petaluma_2022]
    beta_learner = BetaLearner(instance_list)
    feature_list = ["Age Range", "Housing Status", "Gender", "Race/Ethnicity", "Educational Attainment"]
    betas = beta_learner.learn_betas(feature_list)
    for instance in instance_list:
        instance.compute_dropout_probabilities(betas)
        greedy = instance.greedy_alt_set()
        print(f'{instance.name} greedy alt set: {greedy}')


def simulation3(num_train_samples=100, num_test_samples = 100):
    Deschutes_2024 = Instance("Deschutes_2024", num_train_samples=num_train_samples)
    Eugene_2020 = Instance("Eugene_2020", num_train_samples=num_train_samples)
    Petaluma_2022 = Instance("Petaluma_2022", num_train_samples=num_train_samples)

    instance_list = [Deschutes_2024, Eugene_2020, Petaluma_2022]
    beta_learner = BetaLearner(instance_list)
    feature_list = ["Age Range", "Housing Status", "Gender", "Race/Ethnicity", "Educational Attainment"]
    betas = beta_learner.learn_betas(feature_list)
    for instance in instance_list:
        instance.compute_dropout_probabilities(betas)
        test_dropout_samples = instance.generate_dropout_samples(num_test_samples)
        k = len(instance.panel)
        practitioner_num_alts = len(instance.alternates)
        
        alt_indices = [i for i in range(1, k + 1) if k % i == 0]
        all_losses = {label: ([], []) for label in ['L1 Opt', 'Binary Opt', 'Practitioner', 'Greedy']}
        empty_alt_losses = instance.l1_loss([], test_dropout_samples)
        pool_alt_losses = instance.l1_loss(instance.pool, test_dropout_samples)
        full_practitioner_set_loss = instance.l1_loss(instance.alternates, test_dropout_samples)
        greedy_alt_set = instance.greedy_alt_set()
        
        for alt_budget in alt_indices:
            l1_alt_set = instance.opt_l1(alt_budget)
            binary_alt_set = instance.opt_binary(alt_budget)
            l1_opt_losses = instance.l1_loss(l1_alt_set, test_dropout_samples)
            binary_opt_losses = instance.l1_loss(binary_alt_set, test_dropout_samples)
            greedy_loss = instance.l1_loss(greedy_alt_set[:alt_budget], test_dropout_samples)
            if alt_budget < practitioner_num_alts:
                practitioner_losses = []
                for _ in range(100):
                    random_alt_set = random.sample(instance.alternates, alt_budget)
                    practitioner_loss = instance.l1_loss(random_alt_set, test_dropout_samples)
                    practitioner_losses.append(practitioner_loss)
                practitioner_loss = list(np.mean(np.array(practitioner_losses), axis=0))
            else:
                practitioner_loss = full_practitioner_set_loss

            losses = l1_opt_losses + binary_opt_losses + practitioner_loss + empty_alt_losses + pool_alt_losses
            make_violin_plot(losses, ['L1 Opt', 'Binary Opt', 'Practitioner', 'Empty Alt', 'Best Case Alt (A=N)'], num_test_samples, f'{instance.name} Losses, {alt_budget} Alternates', f'../plots/simulation3/{instance.name}_{alt_budget}alts_violin.png')
            # Append means and stds for each loss type
            all_losses['L1 Opt'][0].append(np.mean(l1_opt_losses))
            all_losses['L1 Opt'][1].append(np.std(l1_opt_losses))
            
            all_losses['Binary Opt'][0].append(np.mean(binary_opt_losses))
            all_losses['Binary Opt'][1].append(np.std(binary_opt_losses))

            # TODO
            all_losses['Practitioner'][0].append(np.mean(practitioner_loss))
            all_losses['Practitioner'][1].append(np.std(practitioner_loss))
            
            all_losses['Greedy'][0].append(np.mean(greedy_loss))
            all_losses['Greedy'][1].append(np.std(greedy_loss))
        
        plot_losses_with_shaded_bands(
            alt_indices,
            np.mean(empty_alt_losses),
            np.mean(pool_alt_losses),
            all_losses,
            ['L1 Opt', 'Binary Opt', 'Practitioner', 'Greedy'],
            instance.name,
            len(instance.panel),
            f"../plots/simulation3/{instance.name}_losses_line_plot.png"
        )
            
def test_binary_loss():
    num_train_samples = 1
    Deschutes_2024 = Instance("Deschutes_2024", num_train_samples=num_train_samples, logging=False)
    Eugene_2020 = Instance("Eugene_2020", num_train_samples=num_train_samples, logging=False)
    Petaluma_2022 = Instance("Petaluma_2022", num_train_samples=num_train_samples, logging=False)

    instance_list = [Deschutes_2024, Eugene_2020, Petaluma_2022]
    beta_learner = BetaLearner(instance_list)
    feature_list = ["Age Range", "Housing Status", "Gender", "Race/Ethnicity", "Educational Attainment"]
    betas = beta_learner.learn_betas(feature_list)
    
    Deschutes_2024.compute_dropout_probabilities(betas)
    alt_budget = 10
    binary_alt_set = Deschutes_2024.opt_binary(alt_budget)
    print(f'Deschutes_2024 binary_alt_set: {binary_alt_set}')
    

def plot_betas_from_csv(csv_filepath):
    labels = []
    betas = []
    with open(csv_filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            feature = row['feature']
            value = row['value']
            labels.append(f"{feature}-{value}")
            betas.append(float(row['beta']))

    plt.figure(figsize=(12, 8))
    plt.bar(labels, betas)
    plt.xlabel('Feature-Value')
    plt.ylabel('Beta')
    plt.title('Betas for Deschutes_2024, Eugene_2020, and Petaluma_2022')
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.xticks(rotation=45, ha='right')  # Rotate x-tick labels and align them to the right
    plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels
    plt.savefig('../plots/betas_bar_chart.png')
    plt.close()
        
if __name__== "__main__":
    simulation3()
   