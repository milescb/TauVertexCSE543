import matplotlib.pyplot as plt
import numpy as np

import mplhep
plt.style.use(mplhep.style.ATLAS)

from .utils import response_curve, makeBins

figsize = (6, 6)

def plot_loss(train_loss, val_loss, save=''):
    plt.figure(figsize=figsize)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch', loc='right')
    plt.ylabel('Loss', loc='top')
    plt.legend()
    if save != '':
        plt.savefig(save, dpi=300)
    plt.close()
        
def plot_roc_curve(fpr, tpr, save=''):
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if save != '':
        plt.savefig(save, dpi=300)

def plot_histogram(variable, bins, range, 
                   xlabel, ylabel, save = '',
                   histtype = 'stepfilled'):
    
    plt.figure(figsize=figsize)
    plt.hist(variable, bins=bins, range=range, histtype=histtype)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save != '':
        plt.savefig(save, dpi=300)
        
def plot_multiple_histograms(variables, bins, range,
                              xlabel, ylabel, labels, 
                              save = '', histtype = 'step'):
    
    plt.figure(figsize=figsize)
    for i, variable in enumerate(variables):
        plt.hist(variable, bins=bins, range=range, 
                 alpha=0.5, label=labels[i], histtype=histtype)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save != '':
        plt.savefig(save, dpi=300)
    plt.close()
        
def plot_multiple_histograms_with_ratio(variables, bins, range,
                                        xlabel, ylabel, labels, 
                                        save='', histtype='step',
                                        normalize=False):

    # Create subplots: 2 rows, 1 column, shared x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, 
                                            figsize=(7, 8), 
                                            sharex=True,   
                                            gridspec_kw={'height_ratios': [3, 1]})

    # Plot histograms on the top subplot
    histograms = []
    bin_edges = None
    hist_objects = []  # Store histogram objects to get colors

    for i, variable in enumerate(variables):
        hist = ax_top.hist(variable, bins=bins, range=range, 
                          label=labels[i], histtype=histtype, density=normalize)
        histograms.append(hist[0])  # counts
        hist_objects.append(hist[-1][0])  # store histogram object
        if bin_edges is None:
            bin_edges = hist[1]  # edges

    ax_top.set_ylabel(ylabel)
    ax_top.legend()

    # Calculate the ratio w.r.t the first histogram
    reference = histograms[0]
    ratios = []

    for counts in histograms[1:]:
        ratio = np.divide(counts, reference, out=np.zeros_like(counts), 
                         where=reference!=0)
        ratios.append(ratio)

    # Plot ratios on the bottom subplot using matching colors
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i, ratio in enumerate(ratios):
        ax_bottom.step(bin_centers, ratio, where='mid', 
                      label=f"{labels[i+1]} / {labels[0]}", 
                      color=hist_objects[i+1].get_edgecolor())

    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_ylabel('Ratio', loc='center')

    plt.tight_layout()
    if save != '':
        plt.savefig(save, dpi=300)
    plt.close()
        
def plot_response_lineshape(truth, pred_classical, pred_nn, 
                            bins, range, xlabel, ylabel, 
                            save='', histtype='step'):
    """Plot response and lineshape (predicted / truth) of decay vertex."""
    
    # take the ratio of classical/truth and nn/truth
    # account for division by zero by setting the ratio to 1 if truth and prediction are both zero
    classical_over_truth = np.divide(pred_classical, truth, out=np.ones_like(pred_classical), where=truth!=0)
    nn_over_truth = np.divide(pred_nn, truth, out=np.ones_like(pred_nn), where=truth!=0)
    
    # check for nans and drop these:
    nan_indices = np.isnan(classical_over_truth)
    classical_over_truth = classical_over_truth[~nan_indices]
    nn_over_truth = nn_over_truth[~nan_indices]
    
    plt.figure(figsize=figsize)
    plt.hist(classical_over_truth, bins=bins, range=range, 
             histtype=histtype, label='Classical')
    plt.hist(nn_over_truth, bins=bins, range=range,
                histtype=histtype, label='NN')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    if save != '':
        plt.savefig(save, dpi=300)
    plt.close()
        
def plot_resolution_vs_variable(truth, pred_classical, pred_nn, variable, 
                                nbins, xlabel, ylabel, range=(0, 2),
                                save='', CL=0.68, mask=[]):
    """Plot resolution vs a variable."""
    
    classical_over_truth = np.divide(pred_classical, truth, out=np.ones_like(pred_classical), where=truth!=0)
    nn_over_truth = np.divide(pred_nn, truth, out=np.ones_like(pred_nn), where=truth!=0)
    
    # drop nans
    nan_indices = np.isnan(classical_over_truth)
    classical_over_truth = classical_over_truth[~nan_indices]
    nn_over_truth = nn_over_truth[~nan_indices]
    variable = variable[~nan_indices]
    
    bins = makeBins(range[0], range[1], nbins)
    
    (bin_centers_classical, bin_errors_classical, 
     means_classical, mean_stat_err_classical, 
     resol_classical) = response_curve(classical_over_truth, variable, bins, cl=CL)
    
    (bin_centers_nn, bin_errors_nn,
        means_nn, mean_stat_err_nn,
        resol_nn) = response_curve(nn_over_truth, variable, bins, cl=CL)
    
    plt.figure(figsize=figsize)
    plt.errorbar(bin_centers_classical, means_classical, bin_errors_classical, fmt='o', label='Classical')
    plt.errorbar(bin_centers_nn, means_nn, bin_errors_nn, fmt='o', label='NN')
    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} Prediction / Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()
    
    plt.figure(figsize=figsize)
    plt.plot(bin_centers_classical, resol_classical, label='Classical')
    plt.plot(bin_centers_nn, resol_nn, label='NN')
    plt.xlabel(xlabel)
    plt.ylabel(f'Resolution of {ylabel} at {int(CL*100)}% CL')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save.replace('.pdf', '_resolution.pdf'), dpi=300)
    plt.close()
    
    # split NN in events for mask and ~mask
    # just plot resolution curve
    if len(mask) > 0:
        mask = mask[~nan_indices]
        
        nn_over_truth_upper = nn_over_truth[mask]
        nn_over_truth_lower = nn_over_truth[~mask]
        
        (bin_centers_nn_upper, bin_errors_nn_upper,
            means_nn_upper, mean_stat_err_nn_upper,
            resol_nn_upper) = response_curve(nn_over_truth_upper, variable[mask], bins, cl=CL)
        
        (bin_centers_nn_lower, bin_errors_nn_lower,
            means_nn_lower, mean_stat_err_nn_lower,
            resol_nn_lower) = response_curve(nn_over_truth_lower, variable[~mask], bins, cl=CL)
        
        plt.figure(figsize=figsize)
        plt.plot(bin_centers_classical, resol_classical, label='Classical')
        plt.plot(bin_centers_nn, resol_nn, label='NN', color='orange')
        plt.plot(bin_centers_nn_upper, resol_nn_upper, label=r'$|\sigma/\mu|$ > 1',
                 color='orange', linestyle='--')
        plt.plot(bin_centers_nn_lower, resol_nn_lower, label=r'$|\sigma/\mu|$ < 1',
                 color='orange', linestyle='-.')
        plt.xlabel(xlabel)
        plt.ylabel(f'Resolution of {ylabel} at {int(CL*100)}% CL')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save.replace('.pdf', '_split_resolution.pdf'), dpi=300)
        plt.close()