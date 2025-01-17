import utils
import parameters
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

panels, pools, alternates = utils.read_and_seperate_data()
best_betas = {0: (-1, [], []), 1: (-1, [], []), 2: (-1, [], [])}
labels = ['Deschutes', 'Eugene', 'Petaluma']

for i in range(parameters.datasets):
    panels_except_i = panels[:i] + panels[i+1:]
    for subset in utils.generate_subsets():
        loss, betas, cols = utils.compute_loss_and_betas(subset, panels_except_i)
        if loss > best_betas[i][0]:
            best_betas[i] = (loss, betas, cols)
     
print(panels, pools, best_betas)

for dataset in range(parameters.datasets):
    quotas = utils.compute_quotas(panels[dataset], best_betas[dataset][2])
    for _, num_alternates in enumerate(parameters.alternates_numbers[dataset]):
        alternate_sets = []
        for benchmark in range(len(parameters.benchmarks)):
            alt_set = utils.calculate_alternate_set(panels[dataset], pools[dataset], quotas, alternates[dataset][1], num_alternates, benchmark, best_betas[dataset][1], best_betas[dataset][2])
            alternate_sets.append(alt_set)
        losses = [[] for _ in range(len(parameters.benchmarks))] 
        for _ in range(parameters.plot_samples):
            cur_losses = utils.calculate_loss_sets(alternate_sets[:len(alternate_sets)-1], panels[dataset], pools[dataset], quotas, best_betas[dataset][1], best_betas[dataset][2], num_alternates)
            for i, loss in enumerate(cur_losses):
                losses[i].append(loss)
        print(losses)
        utils.plot_violin(losses, labels[dataset], dataset, num_alternates)

