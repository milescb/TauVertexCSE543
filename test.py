import numpy as np
import argparse
import uproot

import torch
from torch.utils.data import DataLoader

from decayvertex.process_data import testing_data
from decayvertex.architecture import MDNDecayNet
from decayvertex.plotting import plot_multiple_histograms_with_ratio, plot_response_lineshape

def evaluate_model(model, test_loader):
    """Evaluate MDN model and return predictions with uncertainties"""
    model.eval()
    predictions = []
    uncertainties = []
    true_values = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            
            # Get mixture parameters
            pi, mu, sigma = model(inputs)
            
            # Compute global statistics
            mean, std = model.get_mixture_statistics(pi, mu, sigma)
            
            predictions.append(mean.cpu())
            uncertainties.append(std.cpu())
            true_values.append(labels.cpu())
            
    return (torch.cat(predictions), 
            torch.cat(uncertainties), 
            torch.cat(true_values))

def main():
    # Load model and validation indices
    checkpoint = torch.load(args.model_path)
    val_indices = checkpoint['val_indices']
    
    # Load test data using validation indices
    test_dataset, est_decay_vertex_vec = testing_data(tree, val_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    input_size = 9
    model = MDNDecayNet(input_size, args.hidden_size, output_size=3, n_gaussians=args.n_gaussians)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get predictions with uncertainties
    predictions, uncertainties, true_values = evaluate_model(model, test_loader)
    
    # Convert to numpy
    predictions = predictions.numpy()
    uncertainties = uncertainties.numpy()
    true_values = true_values.numpy()
    
    # Plot comparisons with uncertainties
    components = ['x', 'y', 'z']
    ranges = [(-15, 15), (-15, 15), (-100, 100)]
    
    for i, (component, plot_range) in enumerate(zip(components, ranges)):
        plot_multiple_histograms_with_ratio(
            variables=[true_values[:, i], 
                      predictions[:, i], 
                      est_decay_vertex_vec[i]],
            bins=50,
            range=plot_range,
            xlabel=f'Decay Vertex {component} [mm]',
            ylabel='Normalized Events',
            labels=['Truth', 'Prediction', 'Classical'],
            save=f"{args.output_dir}/estimated_vertex_{component}.pdf",
            normalize=True
        )
        plot_response_lineshape(true_values[:, i], predictions[:, i], est_decay_vertex_vec[i],
                                bins=50, range=plot_range, 
                                xlabel=f'Prediction / Truth for {component} [mm]',
                                ylabel='Normalized Events',
                                save=f"{args.output_dir}/response_lineshape_{component}.pdf"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--n-gaussians", type=int, default=2)
    args = parser.parse_args()
    
    file = uproot.open("data/data_large.root")
    tree = file["NOMINAL"]   
    
    main()