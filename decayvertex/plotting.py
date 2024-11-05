import matplotlib.pyplot as plt
import numpy as np

import mplhep
plt.style.use(mplhep.style.ATLAS)

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