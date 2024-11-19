import numpy as np
import argparse
import wandb
import uproot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from decayvertex.process_data import training_data
from decayvertex.architecture import MDNDecayNet
from decayvertex.plotting import plot_loss

def train(config=None):
    with wandb.init(config=config) as run:
        # Access hyperparameters from wandb
        config = wandb.config
        
        # Data loading
        file = uproot.open("data/data_large.root")
        tree = file["NOMINAL"]
        train_dataset, val_dataset, val_indices = training_data(tree)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Model setup
        input_size = 9
        output_size = 3
        
        model = MDNDecayNet(input_size, config.hidden_size, output_size, config.n_gaussians)
        
        # Apply Kaiming weight initialization
        def kaiming_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(kaiming_init)
        
        # Device setup
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop = False

        for epoch in range(config.num_epochs):
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
            
            # Log metrics to wandb
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch": epoch
            })
            
            # Early stopping check
            if avg_val_loss < best_val_loss - config.min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_indices': val_indices,
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, f"best_model_{run.id}.pth")
                wandb.save(f"wandb/best_model_{run.id}.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= config.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                early_stop = True
                break
                
        wandb.run.summary["best_val_loss"] = best_val_loss

if __name__ == "__main__":
    # Define sweep configuration
    sweep_configuration = {
        'method': 'bayes',  # Using Bayesian optimization
        'metric': {
            'name': 'best_val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': -9.21,  # log(0.0001)
                'max': -4.61,  # log(0.01)
            },
            'hidden_size': {
                'values': [64, 128, 256]
            },
            'n_gaussians': {
                'values': [1, 2, 3]
            },
            'batch_size': {'value': 128},
            'num_epochs': {'value': 25},
            'patience': {'value': 5},
            'min_delta': {'value': 1e-3},
        }
    }

    # Initialize wandb
    # wandb.login()
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep_configuration, project="decay-vertex-mdn")
    
    # Run the sweep
    wandb.agent(sweep_id, train, count=50)  # Run 20 experiments