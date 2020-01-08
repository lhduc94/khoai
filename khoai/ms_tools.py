import numpy as np
from scipy.spatial.distance import minkowski

def dist_with_miss(a, b, p=1, l=0.0):
    """ A function compute a distance betwwen 2 array with missing value.
    Parameters:
				a: array
					Input Array.	
				b: array
					Input Array.
				p: int
					The order of the norm of the difference in minkowski distance.
				l: float
					The lambda value of missing value.
    Output: 
				distance: float
					The distance between 2 array.
	"""
    if(len(a) != len(b)):
        return np.inf
    a = np.array(a)
    b = np.array(b)
    ls = l * np.ones(len(a))
    msk = ~ (np.isnan(a) | np.isnan(b))
    res = minkowski(a[msk],b[msk],p) + np.sum((ls[~msk]))**(1/p)
    return res
