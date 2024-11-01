import numpy as np

def process_tlorentz_vector(lorentz_vector_array):
    """
    Process TLorentzVector array from uproot to extract kinematic variables.
    
    Parameters:
    -----------
    lorentz_vector_array : awkward array
        Array of TLorentzVector from uproot with structure:
        TLorentzVector[fP: TVector3[fX, fY, fZ], fE]
    
    Returns:
    --------
    dict : Dictionary containing numpy arrays of pt, eta, phi, e, px, py, pz, p
    """
    # Extract components
    px = lorentz_vector_array.fP.fX
    py = lorentz_vector_array.fP.fY
    pz = lorentz_vector_array.fP.fZ
    e = lorentz_vector_array.fE
    
    # Convert to numpy arrays for calculations
    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)
    e = np.array(e)
    
    # Calculate derived quantities
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)  # total momentum
    
    # Handle division by zero in eta calculation
    eta = np.zeros_like(pt)
    nonzero_pt = pt > 0
    eta[nonzero_pt] = np.arctanh(pz[nonzero_pt] / p[nonzero_pt])
    
    # Calculate phi, handling the different quadrants correctly
    phi = np.arctan2(py, px)
    
    # Package results
    kinematics = {
        'pt': pt,
        'eta': eta,
        'phi': phi,
        'e': e,
        'px': px,
        'py': py,
        'pz': pz,
        'p': p,  # added total momentum
        'm': np.sqrt(np.maximum(e**2 - p**2, 0))  # protect against numerical issues
    }
    
    return kinematics