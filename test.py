import numpy as np
import argparse
import uproot

import torch
from torch.utils.data import DataLoader

from decayvertex.process_data import training_data, ThreeVectorArray
from decayvertex.architecture import DecayNet
from decayvertex.plotting import plot_multiple_histograms_with_ratio

def evaluate_model(model, test_loader):
    """Evaluate model on test data and return predictions and true values"""
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions.append(outputs)
            true_values.append(labels)
            
    return torch.cat(predictions), torch.cat(true_values)

def main():
    # Load test data
    _, test_dataset = training_data(tree, test_size=args.test_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    input_size = 10
    model = DecayNet(input_size, args.hidden_size, output_size=3)
    model.load_state_dict(torch.load(args.model_path))
    
    # Get predictions
    predictions, true_values = evaluate_model(model, test_loader)
    
    # Convert to numpy for plotting
    predictions = predictions.numpy()
    true_values = true_values.numpy()
    
    # Plot comparisons for each component
    components = ['x', 'y', 'z']
    ranges = [(-15, 15), (-15, 15), (-100, 100)]
    
    for i, (component, plot_range) in enumerate(zip(components, ranges)):
        plot_multiple_histograms_with_ratio(
            variables=[true_values[:, i], predictions[:, i], est_decay_vertex_vec[i]],
            bins=50,
            range=plot_range,
            xlabel=f'Decay Vertex {component} [mm]',
            ylabel='Events',
            labels=['Truth', 'Prediction', 'Classical'],
            save=f"{args.output_dir}/test_{component}.pdf",
            normalize=True
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    
    file = uproot.open("data/data_large.root")
    tree = file["NOMINAL"]
    
    # classical result
    est_decay_vertex = ThreeVectorArray(tree['est_lep_decayVertex_v3'].array())
    est_decay_vertex_vec = np.array([est_decay_vertex.x, est_decay_vertex.y, est_decay_vertex.z])   
    
    main()