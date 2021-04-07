import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import scipy.stats as stats
import math


def plot_params_distribution(model):
    l = [module for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)]
    params_list = [module.weight.clone().detach().flatten().numpy() for module in l[1:]]
    params_array = np.zeros(0)  # parameters from all layers
    for layer in l[1:]:
        params_array = np.append(params_array, layer.weight.clone().detach().numpy())

    n_col = 2
    n_row = math.ceil(len(l)/n_col)
    fig, ax = plt.subplots(figsize=(7 * n_col, 5 * n_row), nrows=n_row, ncols=n_col)
    fig.patch.set_facecolor('w')
    idx = 0
    for row in ax:
        for col in row:
            if idx < len(params_list):
                col.grid(linestyle='dotted')
                col.set_facecolor('whitesmoke')
                bin_edges = np.histogram(params_list[idx], bins=100)[1]
                col.hist(params_list[idx], bins=100, edgecolor='steelblue', density=True, stacked=True)
                loc, std = stats.norm.fit(params_list[idx])
                col.plot(bin_edges, stats.norm.pdf(bin_edges, loc=loc, scale=std))
                col.legend(['pdf', 'normalized frequency'])
                col.title.set_text('Layer: {:02d}'.format(idx+1))
                idx+=1
            elif idx == len(params_list):
                col.grid(linestyle='dotted')
                col.set_facecolor('whitesmoke')
                bin_edges = np.histogram(params_array, bins=100)[1]
                col.hist(params_array, bins=100, edgecolor='steelblue', density=True, stacked=True)
                loc, std = stats.norm.fit(params_array)
                col.plot(bin_edges, stats.norm.pdf(bin_edges, loc=loc, scale=std))
                col.legend(['pdf', 'normalized frequency'])
                col.title.set_text('Parameters from all layers')
                idx += 1
            else:
                break
    return fig

