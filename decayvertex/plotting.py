import matplotlib.pyplot as plt

figsize = (6, 6)

def plot_histogram(variable, bins, range, 
                   xlabel, ylabel, save = ''):
    
    plt.figure(figsize=figsize)
    plt.hist(variable, bins=bins, range=range)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    if save != '':
        plt.savefig(save, dpi=300)
        
def plot_multiple_histograms(variables, bins, range, 
                              xlabel, ylabel, labels, save = ''):
    
    plt.figure(figsize=figsize)
    for i, variable in enumerate(variables):
        plt.hist(variable, bins=bins, range=range, alpha=0.5, label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    if save != '':
        plt.savefig(save, dpi=300)