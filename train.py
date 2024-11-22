import numpy as np
import argparse

import uproot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from decayvertex.process_data import training_data
from decayvertex.architecture import MDNDecayNet
from decayvertex.plotting import plot_loss

def main():

    train_dataset, val_dataset, val_indices, scaler_inputs, scaler_labels = training_data(tree)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = 9
    hidden_size = args.hidden_size
    output_size = 3
    n_gaussians = args.n_gaussians

    model = MDNDecayNet(input_size, hidden_size, output_size, n_gaussians)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    training_losses = []
    validation_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop = False

    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            pi, mu, sigma = model(inputs)
            loss = model.mdn_loss_fn(pi, mu, sigma, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                pi, mu, sigma = model(inputs)
                loss = model.mdn_loss_fn(pi, mu, sigma, labels)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Early stopping check
        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_indices': val_indices,
                'scaler_inputs': scaler_inputs,
                'scaler_labels': scaler_labels,
                'hidden_size': hidden_size,
                'n_gaussians': n_gaussians,
                'batch_size': batch_size,
                'learning_rate': args.learning_rate,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, f"{args.output_dir}/best_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            early_stop = True
            break
            
        print(f"Epoch [{epoch+1}/{args.num_epochs}], "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f}")
        
        training_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)

    # After training loop
    if not early_stop:
        print(f'Completed all {args.num_epochs} epochs')
    
    # plot loss and save model 
    plot_loss(training_losses, validation_losses, save=f"{args.output_dir}/loss.png")
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_indices': val_indices,
        'scaler_inputs': scaler_inputs,
        'scaler_labels': scaler_labels,
        'hidden_size': hidden_size,
        'n_gaussians': n_gaussians,
        'batch_size': batch_size,
        'learning_rate': args.learning_rate,
    }, f"{args.output_dir}/model.pth")
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-epoch", "--num-epochs", type=int, default=10)
    parser.add_argument("-bs", "--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("-out", "--output-dir", type=str, default="output")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0004)
    parser.add_argument("--n-gaussians", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5,
                    help="Number of epochs to wait for improvement before stopping")
    parser.add_argument("--min-delta", type=float, default=1e-3,
                    help="Minimum change in validation loss to qualify as an improvement")
    args = parser.parse_args()
    
    # checkout for mps or cuda device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    
    file = uproot.open("data/data_large.root")
    tree = file["NOMINAL"]
    
    main()
