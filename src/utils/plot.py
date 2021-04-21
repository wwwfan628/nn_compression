import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import scipy.stats as stats
import math
import torch

def plot_distribution(param_dict, param_grad_dict, layer_out_dict, layer_out_grad_dict):
    #param_fig = plot_dict(param_dict)
    #param_grad_fig = plot_dict(param_grad_dict)
    layer_out_fig = plot_dict(layer_out_dict)
    layer_out_grad_fig = plot_dict(layer_out_grad_dict)
    #return param_fig, param_grad_fig, layer_out_fig, layer_out_grad_fig
    return None, None, layer_out_fig, layer_out_grad_fig

def plot_dict(dict):
    layer_names = []
    values_list = []
    for layer_name, tensor_list in dict.items():
        values = np.zeros(0)
        for tensor in tensor_list:
            values = np.append(values, tensor.clone().detach().flatten().cpu().numpy())
        layer_names.append(layer_name)
        values_list.append(values)
    n_col = 2
    n_row = math.ceil(len(dict)/n_col)
    fig, ax = plt.subplots(figsize=(7 * n_col, 5 * n_row), nrows=n_row, ncols=n_col)
    fig.patch.set_facecolor('w')
    idx = 0
    for row in ax:
        for col in row:
            if idx < len(dict):
                col.grid(linestyle='dotted')
                col.set_facecolor('whitesmoke')
                bin_edges = np.histogram(values_list[idx], bins=100)[1]
                col.hist(values_list[idx], bins=100, edgecolor='steelblue', density=True, stacked=True)
                #loc, std = stats.norm.fit(values_list[idx])
                loc = values_list[idx].mean()
                std = values_list[idx].std()
                col.plot(bin_edges, stats.norm.pdf(bin_edges, loc=loc, scale=std))
                col.legend(['pdf', 'normalized frequency'])
                col.title.set_text(layer_names[idx])
                idx+=1
            else:
                break
    return fig



def plot_params_distribution(model):
    l = [module for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)]
    params_list = [module.weight.clone().detach().flatten().cpu().numpy() for module in l]
    params_array = np.zeros(0)  # parameters from all layers
    for layer in l:
        params_array = np.append(params_array, layer.weight.clone().detach().cpu().numpy())

    n_col = 2
    n_row = math.ceil((len(l)+1)/n_col)
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



def plot_tensor_distribution(tensor, name):
    elements = tensor.clone().detach().flatten().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 5), nrows=1, ncols=1)
    fig.patch.set_facecolor('w')
    ax.grid(linestyle='dotted')
    ax.set_facecolor('whitesmoke')
    bin_edges = np.histogram(elements, bins=50)[1]
    ax.hist(elements, bins=50, edgecolor='steelblue', density=True, stacked=True)
    loc, std = stats.norm.fit(elements)
    ax.plot(bin_edges, stats.norm.pdf(bin_edges, loc=loc, scale=std))
    ax.legend(['pdf', 'normalized frequency'])
    ax.title.set_text(name)
    return fig