from data_objects import Instance, BetaLearner
import plotter
import seaborn as sns
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle

DATASETS = {"HD": (["Eugene_2020", "Petaluma_2022", "Deschutes_2024"], ["Age Range", "Housing Status", "Gender", "Race/Ethnicity", "Educational Attainment"]),
            "MASS_ALDFC": (["ABG", "ABI", "ABP", "ABQ", "ABS", "ABU"], ["AGE", "GENDER", "INDIGENOUS", "RACIAL_MINORITY", "HOUSING"]),
            "MASS_ALDFN": (["ABY", "ABZ", "ACA", "ACB"], ["AGE", "GENDER", "INDIGENOUS", "RACIAL_MINORITY", "INCOME"])}

PLOT_INSTANCE_NAME = {'Eugene_2020': 'US-1', 'Petaluma_2022': 'US-2', 'Deschutes_2024': 'US-3', 'ABY': 'Can-1', 'ABZ': 'Can-2', 'ACA': 'Can-3', 'ACB': 'Can-4'}
PLOT_DATASET_NAME = {'HD': 'US', 'MASS': 'Can'}

NUM_L1_SAMPLES = 300
NUM_BINARY_SAMPLES = 300
# def get_dropout_rates():
#     # Deschutes_2024 = Instance("Deschutes_2024")
#     # Eugene_2020 = Instance("Eugene_2020")
#     # Petaluma_2022 = Instance("Petaluma_2022")
#     # instance_list = [Deschutes_2024, Eugene_2020, Petaluma_2022]
#     # beta_learner = BetaLearner(instance_list)
#     # feature_list = ["Age Range", "Housing Status", "Gender", "Race/Ethnicity", "Educational Attainment"]
#     # data_files = ["../data/Deschutes_2024/Deschutes_2024_cleaned.csv", "../data/Eugene_2020/Eugene_2020_cleaned.csv", "../data/Petaluma_2022/Petaluma_2022_cleaned.csv"]
#     # data_frames = [pd.read_csv(file) for file in data_files]
#     feature_list = ["A", "L", "D", "F", "J"]
#     datasets = ["ABS", "ABR", "ABL"]
#     df = pd.read_csv('../data/cleaned_anonymized_data.csv')
#     df = df[df['DATA_ID'].isin(datasets)]
#     combined_df = df[feature_list + ["STATUS"]]

#     dropout_rates = {}
#     for feature in feature_list:
#         feature_values = combined_df[feature].unique()
#         dropout_rates[feature] = {}
#         for value in feature_values:
#             subset = combined_df[(combined_df[feature] == value) & (combined_df["STATUS"] != "Not selected")]
#             dropout_rate = subset["STATUS"].value_counts(normalize=True).get("Selected, dropped out", 0)
#             dropout_rates[feature][value] = dropout_rate
#     print(dropout_rates)
#     # Prepare data for plotting
#     plot_data = []
#     for feature, values in dropout_rates.items():
#         for value, rate in values.items():
#             plot_data.append([feature, value, rate])
    
#     df = pd.DataFrame(plot_data, columns=["Feature", "Value", "Dropout Rate"])
    
#     # Plot clustered bar chart
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x="Feature", y="Dropout Rate", hue="Value", data=df)
#     plt.title("Dropout Rates by Feature")
#     plt.xlabel("Feature")
#     plt.ylabel("Dropout Rate")
#     plt.legend(title="Value", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig("../plots/dropout_rates_by_feature.png")
#     plt.show()
#     return dropout_rates

def check_beta_calibration(dataset):
    instance_names = DATASETS[dataset][0]
    feature_list = DATASETS[dataset][1]
    instance_list = [Instance(name, file_stub=f'{dataset}/') for name in instance_names]
    
    fig, axes = plt.subplots(1, len(instance_list), figsize=(6 * len(instance_list), 6), sharex=True, sharey=True)
    if len(instance_list) == 1:
        axes = [axes]
    
    for i, (ax, instance) in enumerate(zip(axes, instance_list)):
        scatter_data = []
        other_instances = instance_list[:i] + instance_list[i+1:]
        beta_learner = BetaLearner(other_instances, file_stub=f'{dataset}/')
        betas, file = beta_learner.learn_betas(feature_list)
        dropout_probs = instance.compute_dropout_probabilities(betas, write_to_file=False)
        dropout_probs = {person: p for person, p in dropout_probs.items() if person in instance.panel}
        expected_num_dropouts = {feature: {value: 0 for value in instance.quotas[feature]} for feature in instance.features}
        for person, p in dropout_probs.items():
            for feature in instance.features:
                expected_num_dropouts[feature][instance.people[person][feature]] += p
        true_num_dropouts = {feature: {value: 0 for value in instance.quotas[feature]} for feature in instance.features}
        for person in instance.dropouts:
            for feature in instance.features:
                true_num_dropouts[feature][instance.people[person][feature]] += 1
                
        # Prepare data for scatter plot
        for feature in instance.features:
            for value in instance.quotas[feature]:
                expected = expected_num_dropouts[feature][value]
                true = true_num_dropouts[feature][value]
                scatter_data.append([feature, expected, true])
        
        err = 0
        with open(f"../logging/{dataset}/{instance.name}/beta_calibration_check.csv", "w", newline="") as csvfile:
            fieldnames = ["Feature", "Value", "Expected Num Dropouts", "True Num Dropouts", "Difference"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for feature in instance.features:
                for value in instance.quotas[feature]:
                    expected = expected_num_dropouts[feature][value]
                    true = true_num_dropouts[feature][value]
                    difference = expected - true
                    err += abs(difference)
                    writer.writerow({"Feature": feature, "Value": value, "Expected Num Dropouts": expected, "True Num Dropouts": true, "Difference": difference})
        print(f"Error for {instance.name}: {err}")
        
        scatter_df = pd.DataFrame(scatter_data, columns=["Feature", "Expected Num Dropouts", "True Num Dropouts"])
        
        # Plot scatter plot for each instance
        sns.scatterplot(x="True Num Dropouts", y="Expected Num Dropouts", hue="Feature", data=scatter_df, palette="tab10", ax=ax)
        ax.plot([0, 6], 
                [0, 6], 
                'r--')
        ax.set_title(PLOT_INSTANCE_NAME[instance.name], fontsize=17)
        if i == 0:
            ax.set_ylabel("Expected Number of Dropouts", fontsize = 17)
            ax.legend(fontsize=14)
        else:
            ax.set_ylabel('')
            ax.legend().remove()
        ax.set_xlabel('')
    
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, title="Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.supxlabel("True Number of Dropouts", fontsize = 17)
    # fig.supylabel("Expected Number of Dropouts")
    
    plt.tight_layout()
    plt.savefig(f"../plots/beta_calibration/{dataset}_true_vs_expected_dropouts_side_by_side.png")
    plt.close()

def simulation1(dataset, loss_type='l1'):
    instance_names = DATASETS[dataset][0]
    feature_list = DATASETS[dataset][1]
    instance_list = [Instance(name, file_stub=f'{dataset}/', logging=False) for name in instance_names]

    # alt_indices = [1, 4, 8, 12, 20, 30]
    # losses_by_instance = {instance.name: {} for instance in instance_list}
    all_alt_indices = {}
    ubs = {}
    lbs = {}
    losses = {}
    loss_labels = ['Practitioner', 'Greedy', 'L1 Eq Probs', 'Binary Opt', 'L1 Opt']
    ks = {}
    true_prac_loss_tuples = {}
    for i,instance in enumerate(instance_list):
        other_instances = instance_list[:i] + instance_list[i+1:]
        beta_learner = BetaLearner(other_instances, file_stub=f'{dataset}/')
        betas, file = beta_learner.learn_betas(feature_list)
        plotter.plot_betas_from_csv(file, f"Betas Learned from {[other_inst.name for other_inst in other_instances]}", f"../plots/simulation1/{dataset}/{instance.name}_betas.png")
        instance.compute_dropout_probabilities(betas)
        k = len(instance.panel)
        alt_indices = [i for i in range(1, k + 1) if k % i == 0]
        if len(alt_indices) < 6:
            alt_indices = np.linspace(1, k, 6, dtype=int)
        # can't use existing opt sets bc need to not depend on this instance (betas)
        alg_losses, empty_alt_losses, pool_alt_losses = get_all_losses(instance, dataset, ([instance.dropouts], [[]]), ([instance.dropouts], [[]]), alt_indices, loss=loss_type, use_existing_opt=False)
        true_prac_loss_tuple = None
        print(f'len alternates: {len(instance.alternates)}')
        if len(instance.alternates) > 0:
            true_prac_loss_val = instance.loss(instance.alternates, ([instance.dropouts], [[]]), loss_type=loss_type)
            num_alts = len(instance.alternates)
            true_prac_loss_tuple = (num_alts, np.mean(true_prac_loss_val))
            print(f"True practitioner loss for {instance.name}: {true_prac_loss_tuple}")    
            
        plotter.plot_losses_with_shaded_bands(
                alt_indices,
                np.mean(empty_alt_losses),
                np.mean(pool_alt_losses),
                alg_losses,
                ['L1 Opt', 'Binary Opt', 'Practitioner', 'Greedy', 'L1 Eq Probs'],
                instance.name,
                len(instance.panel),
                f"../plots/simulation1/{dataset}/{instance.name}_losses_{loss_type}_line_plot_with_stddev.pdf",
                true_prac_loss=true_prac_loss_tuple
            )
        plotter.plot_losses_with_shaded_bands(
                alt_indices,
                np.mean(empty_alt_losses),
                np.mean(pool_alt_losses),
                alg_losses,
                ['L1 Opt', 'Binary Opt', 'Practitioner', 'Greedy', 'L1 Eq Probs'],
                instance.name,
                len(instance.panel),
                f"../plots/simulation1/{dataset}/{instance.name}_losses_{loss_type}_line_plot_without_stddev.pdf",
                true_prac_loss=true_prac_loss_tuple
            )
        all_alt_indices[instance.name] = alt_indices
        ubs[instance.name] = np.mean(empty_alt_losses)
        lbs[instance.name] = np.mean(pool_alt_losses)
        losses[instance.name] = alg_losses
        ks[instance.name] = k
        true_prac_loss_tuples[instance.name] = true_prac_loss_tuple
        
    output_data = {
    "all_alt_indices": all_alt_indices,
    "ubs": ubs,
    "lbs": lbs,
    "losses": losses,
    "ks": ks,
    "true_prac_loss_tuples": true_prac_loss_tuples
    }

    with open(f"../plots/simulation1/{dataset}/pkls/simulation1_output_loss_{loss_type}_{NUM_BINARY_SAMPLES}_{NUM_L1_SAMPLES}.pkl", "wb") as f:
        pickle.dump(output_data, f)
    
    # plotter.plot_losses_for_dataset(instance_names, all_alt_indices, ubs, lbs, losses, loss_labels, ks, f"../plots/simulation1/{dataset}/{dataset}_losses_{loss_type}_line_plot_with_stddev.pdf", true_prac_loss_tuples, loss_type, with_stddev=True)
    # plotter.plot_losses_for_dataset(instance_names, all_alt_indices, ubs, lbs, losses, loss_labels, ks, f"../plots/simulation1/{dataset}/{dataset}_losses_{loss_type}_line_plot_no_stddev.pdf", true_prac_loss_tuples, loss_type, with_stddev=False)

        
        

def get_all_losses(instance, dataset, test_dropout_samples, binary_test_dropout_samples, alt_indices, loss = 'l1', pool_dropouts=False, use_existing_opt=True):
    # practitioner_num_alts = len(instance.alternates)
    
    alg_losses = {label: ([], []) for label in ['L1 Opt', 'Binary Opt', 'Practitioner', 'Greedy', 'L1 Eq Probs']}
    empty_alt_losses = instance.loss([], test_dropout_samples, loss_type=loss)
    pool_alt_losses = instance.loss(instance.pool, test_dropout_samples, loss_type=loss)
    # full_practitioner_set_loss = instance.l1_loss(instance.alternates, test_dropout_samples)
    greedy_alt_set = instance.greedy_alt_set()
    
    for alt_budget in alt_indices:
        if use_existing_opt:
            with open(f"../logging/{dataset}/{instance.name}/opt_alt_sets/opt_alt_sets_{alt_budget}.pkl", "rb") as f:
                opt_sets = pickle.load(f)
            l1_alt_set = opt_sets["l1_opt_set"]
            binary_alt_set = opt_sets["binary_opt_set"]
            l1_eq_probs_alt_set = opt_sets["l1_eq_probs_alt_set"]
            # greedy_alt_set = opt_sets["greedy_alt_set"]
            practitioner_alt_set = opt_sets["practitioner_alt_set"]
        else:
            l1_alt_set = instance.opt_l1(alt_budget, pool_dropouts=pool_dropouts)
            binary_alt_set = instance.opt_binary(alt_budget, pool_dropouts=pool_dropouts)
            l1_eq_probs_alt_set = instance.opt_l1_eq_probs(alt_budget, pool_dropouts=pool_dropouts)
            practitioner_alt_set = instance.scaled_practitioner_alt_set(alt_budget)
            
        l1_opt_losses = instance.loss(l1_alt_set, test_dropout_samples, loss_type=loss)
        l1_eq_losses = instance.loss(l1_eq_probs_alt_set, test_dropout_samples, loss_type=loss)
        binary_opt_losses = instance.loss(binary_alt_set, test_dropout_samples, loss_type=loss)
        greedy_loss = instance.loss(greedy_alt_set[:alt_budget], test_dropout_samples, loss_type=loss)
        practitioner_loss = instance.loss(practitioner_alt_set, test_dropout_samples, loss_type=loss)

        # losses = l1_opt_losses + binary_opt_losses + practitioner_loss + greedy_loss + l1_eq_losses + empty_alt_losses + pool_alt_losses
        # plotter.make_violin_plot(losses, ['L1 Opt', 'Binary Opt', 'Practitioner', 'Greedy', 'L1 Eq Probs', 'Empty Alt', 'Best Case Alt (A=N)'], len(test_dropout_samples[0]), f'{instance.name} Losses, Loss Type: {loss}, {alt_budget} Alternates', f'../plots/simulation3/{dataset}/{instance.name}_loss_{loss}_{alt_budget}alts_violin.png')
        # Append means and stds for each loss type
        alg_losses['L1 Opt'][0].append(np.mean(l1_opt_losses))
        alg_losses['L1 Opt'][1].append(np.std(l1_opt_losses))
        
        alg_losses['Binary Opt'][0].append(np.mean(binary_opt_losses))
        alg_losses['Binary Opt'][1].append(np.std(binary_opt_losses))

        alg_losses['Practitioner'][0].append(np.mean(practitioner_loss))
        alg_losses['Practitioner'][1].append(np.std(practitioner_loss))
        
        alg_losses['Greedy'][0].append(np.mean(greedy_loss))
        alg_losses['Greedy'][1].append(np.std(greedy_loss))
        
        alg_losses['L1 Eq Probs'][0].append(np.mean(l1_eq_losses))
        alg_losses['L1 Eq Probs'][1].append(np.std(l1_eq_losses))
        
    return alg_losses, empty_alt_losses, pool_alt_losses
        
def simulation3(dataset, loss_metric = 'l1', pool_dropouts=False):
    instance_names = DATASETS[dataset][0]
    feature_list = DATASETS[dataset][1]
    instance_list = [Instance(name, file_stub=f'{dataset}/') for name in instance_names]

    beta_learner = BetaLearner(instance_list, file_stub=f'{dataset}/')
    betas, file = beta_learner.learn_betas(feature_list)
    plotter.plot_betas_from_csv(file, f"Betas for Simulation 3 Dataset {dataset}", f"../plots/simulation3/{dataset}/betas.png")
    
    all_alt_indices = {}
    ubs = {}
    lbs = {}
    losses = {}
    loss_labels = ['Practitioner', 'Greedy', 'L1 Eq Probs', 'Binary Opt', 'L1 Opt']
    ks = {}
    true_prac_loss_tuples = {}
    
    for instance in instance_list:
        instance.compute_dropout_probabilities(betas)
        test_dropout_samples = instance.generate_dropout_samples(NUM_L1_SAMPLES, pool_dropouts=pool_dropouts)
        binary_test_dropout_samples = instance.generate_dropout_samples(NUM_BINARY_SAMPLES, pool_dropouts=pool_dropouts)

        k = len(instance.panel)
        alt_indices = [i for i in range(1, k + 1) if k % i == 0]
        if len(alt_indices) < 6:
            alt_indices = np.linspace(1, k, 6, dtype=int)
        alg_losses, empty_alt_losses, pool_alt_losses = get_all_losses(instance, dataset, test_dropout_samples, binary_test_dropout_samples, alt_indices, loss=loss_metric, pool_dropouts=pool_dropouts, use_existing_opt=True)
        true_prac_loss_tuple = None
        if len(instance.alternates) > 0:
            true_prac_loss_val = instance.loss(instance.alternates, test_dropout_samples, loss_type=loss_metric)
            num_alts = len(instance.alternates)
            true_prac_loss_tuple = (num_alts, np.mean(true_prac_loss_val))
        
        pool_drop_string = '_with_alt_drops' if pool_dropouts else ''
        plotter.plot_losses_with_shaded_bands(
            alt_indices,
            np.mean(empty_alt_losses),
            np.mean(pool_alt_losses),
            alg_losses,
            loss_labels,
            instance.name,
            k,
            f"../plots/simulation3/{dataset}/{instance.name}_losses_{loss_metric}_line_plot{pool_drop_string}_with_stddev.png",
            true_prac_loss=true_prac_loss_tuple,
            loss_type=loss_metric,
            with_stddev=True
        )
        
        all_alt_indices[instance.name] = alt_indices
        ubs[instance.name] = np.mean(empty_alt_losses)
        lbs[instance.name] = np.mean(pool_alt_losses)
        losses[instance.name] = alg_losses
        ks[instance.name] = k
        true_prac_loss_tuples[instance.name] = true_prac_loss_tuple
    
    output_data = {
        "all_alt_indices": all_alt_indices,
        "ubs": ubs,
        "lbs": lbs,
        "losses": losses,
        "ks": ks,
        "true_prac_loss_tuples": true_prac_loss_tuples
    }

    with open(f"../plots/simulation3/{dataset}/pkls/simulation3_output_loss_{loss_metric}{pool_drop_string}_{NUM_BINARY_SAMPLES}_{NUM_L1_SAMPLES}.pkl", "wb") as f:
        pickle.dump(output_data, f)
    
   
def opt_convergence_test(dataset, train_samples, loss_for_conv='l1'):
    if loss_for_conv not in ['l1', 'binary', 'l1_eq']:
        print("Invalid loss type")
        return None

    instance_names = DATASETS[dataset][0]
    feature_list = DATASETS[dataset][1]
    instance_list = [Instance(name, file_stub=f'{dataset}/') for name in instance_names]
    num_test_samples = max(train_samples)

    beta_learner = BetaLearner(instance_list, file_stub=f'{dataset}/')
    betas, file = beta_learner.learn_betas(feature_list)
    
    for instance in [inst for inst in instance_list if inst.name == 'Deschutes_2024']:
        instance.compute_dropout_probabilities(betas)
        test_dropout_samples = instance.generate_dropout_samples(num_test_samples)
        losses = {num_train_samples: ([], []) for num_train_samples in train_samples}
        k = len(instance.panel)
        alt_indices = [i for i in range(1, k + 1) if k % i == 0]
        if len(alt_indices) < 6:
            alt_indices = np.linspace(1, k, 6, dtype=int)
            
        for train_sample in train_samples:
            for alt_budget in alt_indices:
                opt_set = []
                if train_sample == 300: # using pre-computed opt sets
                    with open(f"../logging/{dataset}/{instance.name}/opt_alt_sets/opt_alt_sets_{alt_budget}.pkl", "rb") as f:
                        opt_sets = pickle.load(f)
                    if loss_for_conv == 'l1':
                        opt_set = opt_sets["l1_opt_set"]
                    elif loss_for_conv == 'l1_eq':
                        opt_set = opt_sets["l1_eq_probs_alt_set"]
                    else:
                        opt_set = opt_sets["binary_opt_set"]
                else:
                    if loss_for_conv == 'l1':
                        opt_set = instance.opt_l1(alt_budget, num_train_samples=train_sample)
                    elif loss_for_conv == 'l1_eq':
                        opt_set = instance.opt_l1_eq_probs(alt_budget, num_train_samples=train_sample)
                    else:
                        opt_set = instance.opt_binary(alt_budget, num_train_samples=train_sample)
                
                l1_losses = instance.loss(opt_set, test_dropout_samples)
                losses[train_sample][0].append(np.mean(l1_losses))
                losses[train_sample][1].append(np.std(l1_losses))    
        
            output_data = {
                "alt_indices": alt_indices,
                "losses": losses,
                "train_samples": train_samples,
                "instance": instance
            }

            with open(f"../plots/l1_num_train_samples_convergence/{dataset}/pkls/{instance.name}_opt_convergence_{train_sample}_loss_{loss_for_conv}.pkl", "wb") as f:
                pickle.dump(output_data, f)
                
        plotter.plot_losses_with_shaded_bands(
            alt_indices,
            None,
            None,
            losses,
            train_samples,
            instance.name,
            len(instance.panel),
            f"../plots/l1_num_train_samples_convergence/{dataset}/{instance.name}_losses_{loss_for_conv}_line_plot.pdf",
            with_stddev=False
        )
    
def calculate_opt_sets(dataset):
    NUM_BINARY_SAMPLES = 300
    NUM_L1_SAMPLES = 300
    instance_names = DATASETS[dataset][0]
    feature_list = DATASETS[dataset][1]
    instance_list = [Instance(name, file_stub=f'{dataset}/') for name in instance_names]

    beta_learner = BetaLearner(instance_list, file_stub=f'{dataset}/')
    betas, file = beta_learner.learn_betas(feature_list)
    for instance in instance_list:
        instance.compute_dropout_probabilities(betas)
        k = len(instance.panel)
        alt_indices = [i for i in range(1, k + 1) if k % i == 0]
        if len(alt_indices) < 6:
            alt_indices = np.linspace(1, k, 6, dtype=int)
        results = {
            "num_alts": [],
            "l1_opt_set": [],
            "binary_opt_set": [],
            "l1_eq_probs_alt_set": [],
            "greedy_alt_set": [],
            "practitioner_alt_set": []
        }
        for num_alts in alt_indices:
            l1_opt_set = instance.opt_l1(num_alts, num_train_samples=NUM_L1_SAMPLES)
            binary_opt_set = instance.opt_binary(num_alts, num_train_samples=NUM_BINARY_SAMPLES)
            l1_eq_probs_alt_set = instance.opt_l1_eq_probs(num_alts, num_train_samples=NUM_L1_SAMPLES)
            greedy_alt_set = instance.greedy_alt_set()[:num_alts]
            practitioner_alt_set = instance.scaled_practitioner_alt_set(num_alts)
            
            results["num_alts"].append(num_alts)
            results["l1_opt_set"].append(l1_opt_set)
            results["binary_opt_set"].append(binary_opt_set)
            results["l1_eq_probs_alt_set"].append(l1_eq_probs_alt_set)
            results["greedy_alt_set"].append(greedy_alt_set)
            results["practitioner_alt_set"].append(practitioner_alt_set)
            
            output_data = {
            "num_alts": num_alts,
            "num_l1_samples": NUM_L1_SAMPLES,
            "num_binary_samples": NUM_BINARY_SAMPLES,
            "l1_opt_set": l1_opt_set,
            "binary_opt_set": binary_opt_set,
            "l1_eq_probs_alt_set": l1_eq_probs_alt_set,
            "greedy_alt_set": greedy_alt_set,
            "practitioner_alt_set": practitioner_alt_set
            }

            with open(f"../logging/{dataset}/{instance.name}/opt_alt_sets/opt_alt_sets_{num_alts}.pkl", "wb") as f:
                pickle.dump(output_data, f)
        
        file_path = f"../logging/{dataset}/{instance.name}/opt_alt_sets/opt_alt_sets.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "num_alts", "l1_opt_set", "binary_opt_set", 
                "l1_eq_probs_alt_set", "greedy_alt_set", "practitioner_alt_set"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(results["num_alts"])):
                writer.writerow({
                    "num_alts": results["num_alts"][i],
                    "l1_opt_set": results["l1_opt_set"][i],
                    "binary_opt_set": results["binary_opt_set"][i],
                    "l1_eq_probs_alt_set": results["l1_eq_probs_alt_set"][i],
                    "greedy_alt_set": results["greedy_alt_set"][i],
                    "practitioner_alt_set": results["practitioner_alt_set"][i]
                })

def robustness_test(dataset, num_alternates = 6, loss_metric = 'l1', num_gam_draws = 10, pool_dropouts=False):
    instance_names = DATASETS[dataset][0]
    feature_list = DATASETS[dataset][1]
    instance_list = [Instance(name, file_stub=f'{dataset}/') for name in instance_names]

    beta_learner = BetaLearner(instance_list, file_stub=f'{dataset}/')
    betas, file = beta_learner.learn_betas(feature_list)
    gammas = [0, 0.2, 0.4, 0.6]
    all_losses = {instance.name: {gamma: {} for gamma in gammas} for instance in instance_list}
    for instance in [inst for inst in instance_list if inst.name == 'Petaluma_2022']:
        # only need to compute the algs that don't use dropout probabilities once: practitioner, l1 eq probs
        instance.compute_dropout_probabilities(betas)
        test_dropout_samples = instance.generate_dropout_samples(NUM_L1_SAMPLES, pool_dropouts=pool_dropouts)
        binary_test_dropout_samples = instance.generate_dropout_samples(NUM_BINARY_SAMPLES, pool_dropouts=pool_dropouts)
        
        pract_alt_set = instance.scaled_practitioner_alt_set(num_alternates)
        prac_loss = np.mean(instance.loss(pract_alt_set, test_dropout_samples, loss_type=loss_metric))
        l1_eq_probs_alt_set = instance.opt_l1_eq_probs(num_alternates, pool_dropouts=pool_dropouts)
        l1_eq_probs_loss = np.mean(instance.loss(l1_eq_probs_alt_set, test_dropout_samples, loss_type=loss_metric))
        print(f"Practitioner Loss for {instance.name}: {prac_loss}")
        print(f"L1 Eq Probs Loss for {instance.name}: {l1_eq_probs_loss}")
        for gamma in gammas:
            all_losses[instance.name][gamma]['Practitioner'] = prac_loss
            all_losses[instance.name][gamma]['L1 Eq Probs'] = l1_eq_probs_loss
            l1_errs = []
            binary_errs = []
            greedy_errs = []
            
            if gamma == 0:
                with open(f"../logging/{dataset}/{instance.name}/opt_alt_sets/opt_alt_sets_{num_alternates}.pkl", "rb") as f:
                    opt_sets = pickle.load(f)
                l1_errs.extend(instance.loss(opt_sets["l1_opt_set"], test_dropout_samples, loss_type=loss_metric))
                binary_errs.extend(instance.loss(opt_sets["binary_opt_set"], binary_test_dropout_samples, loss_type=loss_metric))
                greedy_errs.extend(instance.loss(opt_sets["greedy_alt_set"], test_dropout_samples, loss_type=loss_metric))
                
            else:
                for _ in range(num_gam_draws):
                    l1_opt_set = instance.opt_l1(num_alternates, pool_dropouts=pool_dropouts, est_error=gamma)
                    l1_errs.extend(instance.loss(l1_opt_set, test_dropout_samples, loss_type=loss_metric))
                    binary_opt_set = instance.opt_binary(num_alternates, pool_dropouts=pool_dropouts, est_error=gamma)
                    binary_errs.extend(instance.loss(binary_opt_set, binary_test_dropout_samples, loss_type=loss_metric))
                    greedy_alt_set = instance.greedy_alt_set(est_error=gamma)[:num_alternates]
                    greedy_errs.extend(instance.loss(greedy_alt_set, test_dropout_samples, loss_type=loss_metric))
                
            all_losses[instance.name][gamma]['L1 Opt'] = np.mean(l1_errs)
            all_losses[instance.name][gamma]['Binary Opt'] = np.mean(binary_errs)
            all_losses[instance.name][gamma]['Greedy'] = np.mean(greedy_errs)     
            print(f"Gamma: {gamma}")
            print(f"L1 Opt Loss for {instance.name}: {all_losses[instance.name][gamma]['L1 Opt']}")
            print(f"Binary Opt Loss for {instance.name}: {all_losses[instance.name][gamma]['Binary Opt']}")
            print(f"Greedy Loss for {instance.name}: {all_losses[instance.name][gamma]['Greedy']}")
            
            with open(f"../logging/{dataset}/robustness/robustness_test_{instance.name}_losses_gamma_{gamma}_num_alts_{num_alternates}.pkl", "wb") as f:
                pickle.dump({
                    "losses": all_losses[instance.name][gamma],
                    "l1_errs": l1_errs,
                    "binary_errs": binary_errs,
                    "greedy_errs": greedy_errs
                }, f)

        with open(f"../logging/{dataset}/robustness/robustness_test_{instance.name}_losses.csv", "w", newline="") as csvfile:
            fieldnames = ["Gamma", "Num Alternates", "L1 Opt", "Binary Opt", "Practitioner", "Greedy", "L1 Eq Probs"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for gamma, losses in all_losses[instance.name].items():
                row = {"Gamma": gamma, "Num Alternates": num_alternates}
                row.update(losses)
                writer.writerow(row)
    
    with open(f"../logging/{dataset}/robustness_test_all_losses_num_alts_{num_alternates}.pkl", "wb") as f:
        pickle.dump(all_losses, f)
        
    # colors = {
    #     'L1 Opt': '#A020F0', # purple
    #     'Binary Opt': '#FFA500', # Bright orange
    #     'Practitioner': '#2ca02c', # Green
    #     'Greedy': '#d62728', # Dark red 
    #     'L1 Eq Probs': '#17becf' # Teal/Cyan
    # }
    # fig, axes = plt.subplots(1, len(instance_list), figsize=(6 * len(instance_list), 6), sharey=True)
    # if len(instance_list) == 1:
    #     axes = [axes]
    
    # for ax, instance in zip(axes, instance_list):
    #     for alg in ['Practitioner', 'Greedy', 'L1 Eq Probs', 'L1 Opt', 'Binary Opt']:
    #         losses = [all_losses[instance.name][gamma][alg] for gamma in gammas]
    #         ax.plot(gammas, losses, label=alg, color=colors[alg])
        
    #     ax.set_title(f"{PLOT_INSTANCE_NAME[instance.name]}: L1 Loss vs Gamma with {num_alternates} Alternates")
    #     if ax == axes[0]:
    #         ax.set_ylabel(r"$L^1$ Loss")
    #         ax.legend()
    # fig.text(0.5, 0.04, r"Prediction Error ($\gamma$)", ha='center')
    # fig.suptitle(rf"$L^1$ Loss of {PLOT_INSTANCE_NAME[dataset]} Dataset over Increasing Estimation Error", fontsize=16)
    # plt.tight_layout()
    # plt.savefig(f"../plots/robustness/{dataset}_l1_loss_vs_gamma_{num_alternates}_alternates.png")
    # plt.show()
    
    
if __name__== "__main__":
    # robustness_test("HD", num_gam_draws=15)
    # simulation1("HD")
    # dataset = "HD"
    # instance_names = DATASETS[dataset][0]
    # feature_list = DATASETS[dataset][1]
    # instance_list = [Instance(name, num_train_samples=NUM_TRAIN_SAMPLES, file_stub=f'{dataset}/') for name in instance_names]

    # beta_learner = BetaLearner(instance_list, file_stub=f'{dataset}/')
    # betas, file = beta_learner.learn_betas(feature_list)
    
    # instance_list[0].compute_dropout_probabilities(betas)
    # print(f'pre-err dropout probs: {instance_list[0].dropout_probs}')
    # instance_list[0].generate_dropout_samples(NUM_TEST_SAMPLES, est_error=0.2)
    # instance_list[0].generate_dropout_samples(NUM_TEST_SAMPLES, est_error=0.5)

    # simulation3("MASS_ALDFN", loss_metric='l1', pool_dropouts=False)
    # simulation1("MASS_ALDFN")
    # opt_convergence_test("HD", [50, 100, 300, 500, 1000], loss_for_conv='l1')
    # opt_convergence_test("HD", [25, 50, 100, 200, 300, 500], loss_for_conv='l1')
    # opt_convergence_test("HD", [25, 50, 100, 200, 300, 500], loss_for_conv='l1_eq')

    # possible_loss_metrics = ['l1_dev_below', 'max_quota_dev_norm', 'max_quota_dev', 'num_unrepped']            
    # for loss in possible_loss_metrics:
    #     simulation3("HD", loss_metric=loss, pool_dropouts=False)

    
    # beta_learner = BetaLearner(instance_list, file_stub=f'{dataset}/')
    # feature_list = ["Age Range", "Housing Status", "Gender", "Race/Ethnicity", "Educational Attainment"]
    # betas, file = beta_learner.learn_betas(feature_list)
    # Petaluma_2022.compute_dropout_probabilities(betas)
    # l1_opt_set_no_drops = Petaluma_2022.opt_l1(5, pool_dropouts=False)
    # print('l1 opt set no drops', l1_opt_set_no_drops)
    # l1_opt_set_with_drops = Petaluma_2022.opt_l1(5, pool_dropouts=True)
    # print('l1 opt set yes drops', l1_opt_set_with_drops)
    # test_dropout_samples = Petaluma_2022.generate_dropout_samples(3, pool_dropouts=True)
    # print('test dropout samples', test_dropout_samples)
    # loss_no_drops = Petaluma_2022.loss(l1_opt_set_no_drops, test_dropout_samples, loss_type='l1', verbose=True)
    # print('Loss for l1 opt set without drops:', loss_no_drops)

    # loss_with_drops = Petaluma_2022.loss(l1_opt_set_with_drops, test_dropout_samples, loss_type='l1', verbose=True)
    # print('Loss for l1 opt set with drops:', loss_with_drops)
    # dataset = "HD"
    # with open(f"../logging/{dataset}/simulation3_output.pkl", "rb") as f:
    #     output_data = pickle.load(f)

    # print(output_data)
    check_beta_calibration("HD")