import numpy as np
import argparse
import uproot

import torch
from torch.utils.data import DataLoader

from decayvertex.process_data import testing_data, LorentzVectorArray
from decayvertex.architecture import MDNDecayNet
from decayvertex.plotting import plot_multiple_histograms_with_ratio, plot_response_lineshape, plot_resolution_vs_variable

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
    checkpoint = torch.load(args.model_path, map_location='cpu')
    val_indices = checkpoint['val_indices']
    scaler_inputs = checkpoint['scaler_inputs']
    scaler_labels = checkpoint['scaler_labels']
    hidden_size = checkpoint['hidden_size']
    n_gaussians = checkpoint['n_gaussians']
    batch_size = checkpoint['batch_size']
    
    # Load test data using validation indices
    test_dataset, est_decay_vertex_vec, muon_pt, muon_eta = testing_data(tree, val_indices, scaler_inputs, scaler_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    input_size = 9
    model = MDNDecayNet(input_size, hidden_size, output_size=3, n_gaussians=n_gaussians)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get predictions with uncertainties
    predictions, uncertainties, true_values = evaluate_model(model, test_loader)
    
    # Convert to numpy
    predictions = predictions.numpy()
    uncertainties = uncertainties.numpy()
    true_values = true_values.numpy()
    
    predictions = scaler_labels.inverse_transform(predictions)
    true_values = scaler_labels.inverse_transform(true_values)
    
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
        
        # do split in events
        mask = np.abs(uncertainties[:, i] / predictions[:, i]) < 0.5
        
        plot_response_lineshape(true_values[:, i], est_decay_vertex_vec[i], predictions[:, i],
                                bins=50, range=(-2, 4), 
                                xlabel=f'Decay Vertex {component} [mm], Prediction / Truth',
                                ylabel='Normalized Events',
                                save=f"{args.output_dir}/response_lineshape_{component}.pdf")
        plot_resolution_vs_variable(true_values[:, i], est_decay_vertex_vec[i], 
                                    predictions[:, i], muon_pt,
                                      nbins=7, range=(30, 80),
                                      xlabel=r'Muon $p_T$ [GeV]',
                                      ylabel=f'Decay Vertex {component}',
                                      mask=mask,
                                      save=f"{args.output_dir}/pt_resolution_vs_vertex_{component}.pdf")
        plot_resolution_vs_variable(true_values[:, i], est_decay_vertex_vec[i],
                                    predictions[:, i], muon_eta,
                                    nbins=7, range=(-2, 2),
                                    xlabel=r'Muon $\eta$',
                                    ylabel=f'Decay Vertex {component}',
                                    mask=mask,
                                    save=f"{args.output_dir}/eta_resolution_vs_vertex_{component}.pdf")
        plot_resolution_vs_variable(true_values[:, i], est_decay_vertex_vec[i],
                                    predictions[:, i], true_values[:, i],
                                    nbins=7, range=plot_range,
                                    xlabel=f'Truth Decay Vertex {component}',
                                    ylabel=f'Decay Vertex {component}',
                                    mask=mask,
                                    save=f"{args.output_dir}/vertex_resolution_vs_vertex_{component}.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--data-file", type=str, default="data_large.root")
    args = parser.parse_args()
    
    file = uproot.open(f"data/{args.data_file}")
    tree = file["NOMINAL"]   
    
    main()