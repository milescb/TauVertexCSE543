import numpy as np

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