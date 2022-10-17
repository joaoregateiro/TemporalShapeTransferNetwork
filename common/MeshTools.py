''' Mesh util class for deep learning common functions
Author: Joao Regateiro
'''
import numpy as np                  # all matrix manipulations & OpenGL args


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr


def getNormal(vertices, faces):
    
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)

    tris = vertices[faces]
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])

    normalize_v3(n)
    
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    
    return normalize_v3(norm)
