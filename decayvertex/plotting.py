import matplotlib.pyplot as plt
import numpy as np

figsize = (6, 6)

def plot_histogram(variable, bins, range, 
                   xlabel, ylabel, save = '',
                   histtype = 'stepfilled'):
    
    plt.figure(figsize=figsize)
    plt.hist(variable, bins=bins, range=range, histtype=histtype)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
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
    plt.show()
    if save != '':
        plt.savefig(save, dpi=300)
        
def plot_multiple_histograms_with_ratio(variables, bins, range,
                                        xlabel, ylabel, labels, 
                                        save='', histtype='step'):

    # Create subplots: 2 rows, 1 column, shared x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, 
                                            figsize=(7, 8), 
                                            sharex=True,   # Share x-axis
                                            gridspec_kw={'height_ratios': [3, 1]})

    # Plot histograms on the top subplot
    histograms = []
    bin_edges = None

    for i, variable in enumerate(variables):
        counts, edges, _ = ax_top.hist(variable, bins=bins, range=range, 
                                       label=labels[i], histtype=histtype)
        histograms.append(counts)
        if bin_edges is None:
            bin_edges = edges

    ax_top.set_ylabel(ylabel)
    ax_top.legend()

    # Calculate the ratio w.r.t the first histogram
    reference = histograms[0]
    ratios = []

    for counts in histograms[1:]:
        # Avoid division by zero
        ratio = np.divide(counts, reference, out=np.zeros_like(counts), where=reference!=0)
        ratios.append(ratio)

    # Plot ratios on the bottom subplot
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i, ratio in enumerate(ratios):
        ax_bottom.step(bin_centers, ratio, where='mid', 
                       label=f"{labels[i+1]} / {labels[0]}", linestyle='-')

    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_ylabel('Ratio', loc='center')

    plt.tight_layout()
    if save != '':
        plt.savefig(save, dpi=300)
    plt.show()