import numpy as np
import awkward as ak

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

class LorentzVectorArray:
    """
    Class to process and hold TLorentzVector arrays from uproot.
    Provides easy access to kinematic variables through properties.
    
    Parameters:
    -----------
    lorentz_vector_array : awkward array
        Array of TLorentzVector from uproot with structure:
        TLorentzVector[fP: TVector3[fX, fY, fZ], fE]
    """
    
    def __init__(self, lorentz_vector_array):
        # Extract and store basic components
        self._px = np.array(lorentz_vector_array.fP.fX)
        self._py = np.array(lorentz_vector_array.fP.fY)
        self._pz = np.array(lorentz_vector_array.fP.fZ)
        self._e = np.array(lorentz_vector_array.fE)
        
        # Calculate derived quantities
        self._pt = np.sqrt(self._px**2 + self._py**2) / 1000
        self._p = np.sqrt(self._px**2 + self._py**2 + self._pz**2)
        
        # Calculate eta
        self._eta = np.zeros_like(self._pt)
        nonzero_pt = self._pt > 0
        self._eta[nonzero_pt] = np.arctanh(self._pz[nonzero_pt] / self._p[nonzero_pt])
        
        # Calculate phi
        self._phi = np.arctan2(self._py, self._px)
        
        # Calculate mass
        self._m = np.sqrt(np.maximum(self._e**2 - self._p**2, 0)) / 1000
    
    @property
    def pt(self):
        """Transverse momentum"""
        return self._pt
    
    @property
    def eta(self):
        """Pseudorapidity"""
        return self._eta
    
    @property
    def phi(self):
        """Azimuthal angle"""
        return self._phi
    
    @property
    def e(self):
        """Energy"""
        return self._e
    
    @property
    def px(self):
        """x-component of momentum"""
        return self._px
    
    @property
    def py(self):
        """y-component of momentum"""
        return self._py
    
    @property
    def pz(self):
        """z-component of momentum"""
        return self._pz
    
    @property
    def p(self):
        """Total momentum"""
        return self._p
    
    @property
    def m(self):
        """Invariant mass"""
        return self._m

    def __len__(self):
        """Return the length of the array"""
        return len(self._pt)
    
class ThreeVectorArray:
    """
    Class to process and hold TVector3 arrays from uproot.
    Provides easy access to components through properties.
    
    Parameters:
    -----------
    vector_array : awkward array
        Array of TVector3 from uproot with structure:
        TVector3[fX, fY, fZ]
    """
    
    def __init__(self, vector_array):
        # Extract and store components
        self._x = np.array(vector_array.fX)
        self._y = np.array(vector_array.fY)
        self._z = np.array(vector_array.fZ)
    
    @property
    def x(self):
        """x-component"""
        return self._x
    
    @property
    def y(self):
        """y-component"""
        return self._y
    
    @property
    def z(self):
        """z-component"""
        return self._z

    def __len__(self):
        """Return the length of the array"""
        return len(self._x)
    
class DecayDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
def training_data(tree, test_size=0.2, random_state=42):
    """Get training datasets from a TTree.

    Args:
        tree (ROOT::TTree): TTree containing input data
        test_size (float, optional): Percentage of dataset for testing. Defaults to 0.2.
        random_state (int, optional): Defaults to 42.

    Returns:
        training dataset, validation dataset
    """
    
    muon_p4 = LorentzVectorArray(tree['muon_truth_p4'].array())
    lep_impact_parameter = ThreeVectorArray(tree['lep_impactParameter'].array())
    primary_vertex = ThreeVectorArray(tree['truth_primaryVertex_v3'].array())

    inputs = ak.concatenate([
        muon_p4.pt[:, None],
        muon_p4.eta[:, None],
        muon_p4.phi[:, None],
        lep_impact_parameter.x[:, None],
        lep_impact_parameter.y[:, None],
        lep_impact_parameter.z[:, None],
        primary_vertex.x[:, None],
        primary_vertex.y[:, None],
        primary_vertex.z[:, None],
    ], axis=1)
    
    scaler_inputs = StandardScaler()
    inputs = scaler_inputs.fit_transform(ak.to_numpy(inputs))

    truth_decay_vertex = ThreeVectorArray(tree['truth_lep_decayVertex_v3'].array())

    labels = ak.concatenate([
        truth_decay_vertex.x[:, None],
        truth_decay_vertex.y[:, None],
        truth_decay_vertex.z[:, None],
    ], axis=1)
    
    scaler_labels = StandardScaler()
    labels = scaler_labels.fit_transform(ak.to_numpy(labels))

    dataset = DecayDataset(inputs, labels)
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=test_size, random_state=random_state
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset, val_indices, scaler_inputs, scaler_labels

def testing_data(tree, val_indices, scaler_inputs, scaler_labels):
    """Get testing dataset from a TTree.

    Args:
        tree (ROOT::TTree): TTree containing input data
        val_indices (list): List of indices to test on

    Returns:
        testing dataset, estimated (classical) decay vertex
    """
    
    muon_p4 = LorentzVectorArray(tree['muon_truth_p4'].array())
    lep_impact_parameter = ThreeVectorArray(tree['lep_impactParameter'].array())
    primary_vertex = ThreeVectorArray(tree['truth_primaryVertex_v3'].array())

    inputs = ak.concatenate([
        muon_p4.pt[:, None],
        muon_p4.eta[:, None],
        muon_p4.phi[:, None],
        lep_impact_parameter.x[:, None],
        lep_impact_parameter.y[:, None],
        lep_impact_parameter.z[:, None],
        primary_vertex.x[:, None],
        primary_vertex.y[:, None],
        primary_vertex.z[:, None],
    ], axis=1)
    
    inputs = scaler_inputs.transform(ak.to_numpy(inputs))

    truth_decay_vertex = ThreeVectorArray(tree['truth_lep_decayVertex_v3'].array())

    labels = ak.concatenate([
        truth_decay_vertex.x[:, None],
        truth_decay_vertex.y[:, None],
        truth_decay_vertex.z[:, None],
    ], axis=1)

    labels = scaler_labels.transform(ak.to_numpy(labels))

    dataset = DecayDataset(inputs, labels)
    test_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # classical estimate of decay vertex
    est_decay_vertex = ThreeVectorArray(tree['est_lep_decayVertex_v3'].array())
    est_decay_vertex_vec = np.array([est_decay_vertex.x, est_decay_vertex.y, est_decay_vertex.z])
    est_decay_vertex_vec = est_decay_vertex_vec[:, val_indices]
    
    # other important variables
    muon_p4 = LorentzVectorArray(tree['muon_truth_p4'].array())
    
    muon_pt = muon_p4.pt[val_indices]
    muon_eta = muon_p4.eta[val_indices]
    
    return test_dataset, est_decay_vertex_vec, muon_pt, muon_eta
