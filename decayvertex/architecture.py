import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DecayNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecayNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    
class MDNDecayNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_gaussians):
        super(MDNDecayNet, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_gaussians = n_gaussians
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        # Output layers for mixture parameters
        self.z_pi = nn.Linear(hidden_size, n_gaussians)  # mixing coefficients
        self.z_mu = nn.Linear(hidden_size, n_gaussians * output_size)  # means
        self.z_sigma = nn.Linear(hidden_size, n_gaussians * output_size)  # standard deviations
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = self.dropout(x)
        x = F.gelu(self.fc5(x))

        pi = F.softmax(self.z_pi(x), dim=1)  # normalize mixing coefficients
        mu = self.z_mu(x).view(-1, self.n_gaussians, self.output_size)
        sigma = torch.exp(self.z_sigma(x)).view(-1, self.n_gaussians, self.output_size)
        
        return pi, mu, sigma

    def mdn_loss_fn(self, pi, mu, sigma, target):
        """Negative log likelihood loss for mixture density network"""
        target = target.unsqueeze(1).expand(-1, pi.size(1), -1)
        
        # Calculate gaussian probability
        normal = torch.distributions.Normal(mu, sigma)
        log_prob = normal.log_prob(target).sum(dim=2)
        
        # Calculate mixture probability
        prob = torch.exp(log_prob) * pi
        
        # Sum along gaussian dimension and take log
        nll = -torch.log(torch.sum(prob, dim=1) + 1e-6)
        
        return torch.mean(nll)
        
    
    def get_mixture_statistics(self, pi, mu, sigma):
        """
        Compute global mean and variance for Gaussian mixture.
        
        Args:
            pi (tensor): mixing coefficients [batch_size, n_gaussians]
            mu (tensor): means [batch_size, n_gaussians, output_size]
            sigma (tensor): standard deviations [batch_size, n_gaussians, output_size]
        
        Returns:
            global_mean: weighted mean [batch_size, output_size]
            global_var: weighted variance [batch_size, output_size]
        """
        # Global mean: sum_i π_i μ_i
        global_mean = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
        
        # Global variance: sum_i π_i (σ_i^2 + μ_i^2) - μ_global^2
        global_var = torch.sum(pi.unsqueeze(-1) * (sigma**2 + mu**2), dim=1) - global_mean**2
        
        return global_mean, torch.sqrt(global_var)
