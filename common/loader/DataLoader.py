import os
import json
import numpy as np

# Torch imports
import torch

from pprint import pprint
from common.WavefrontOBJ import *
import igl


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)

#
# Function to save 3D mesh
#
def save_mesh(path, verts, index, faces, normals):
    mverts = verts #np.reshape(verts, (-1, 3))
    mnorms = np.reshape(normals, (-1, 3))

    print('verts.shape: ', mverts.shape)
    mesh = WavefrontOBJ()
    mesh.vertices = mverts
    mesh.normals = mnorms
    mesh.polygons = faces
    save_obj(mesh, path % (index))
    print('Mesh OBJ saved: ', path % (index))


#
# Function to load OBJ files of 3D meshes
#
def load_all_meshes_to_matrixv2(path_list, path_index_list, labels_list , centre_meshes=False, pattern_out=None):
    # Outputs
    vertices        = 0
    centres         = 0
    faces           = {}
    normals         = 0
    masks           = 0

    vertex_count    = 0
    file_count      = 0

    files_list      = path_list.split(" ")
    label_list      = labels_list.split(" ")
    indices_list    = path_index_list.split()

    frame_start     = 0
    frame_end       = 0
    sequenceLength  = 1

    meshList    = []
    colours     = []
    labels      = []
    laplace     = {}

    sequence_index = 0

    for (start, end), file in zip( grouped(indices_list, 2), files_list):
        current_label = label_list[sequence_index]

        for frame in range(int(start), int(end) +1):
            if file.endswith(".obj"):
                mesh_path = (file % frame)
                if os.path.isfile(mesh_path):
                    if current_label not in laplace:
                        v, f = igl.read_triangle_mesh(mesh_path)
                        l = igl.cotmatrix(v, f)
                        laplace[current_label] = [] 
                    if current_label not in faces:
                        mesh = load_obj(mesh_path)
                        faces[current_label] = mesh.polygons

                    meshList.append(mesh_path)
                    labels.append(current_label)
                    #print("Loaded ", mesh_path)

        sequence_index = sequence_index + 1
    return meshList, colours, labels, faces, laplace

def load_mesh_to_matrix(path_list, vertex_count,Only_vert=False):
    mesh = load_obj(path_list)
    verts = mesh.vertices
    norms = mesh.normals
    faces = mesh.polygons

    vertices = np.zeros([vertex_count])
    normals = np.zeros([vertex_count])
    mask = np.zeros([vertex_count])

    # Reshape and add to array (x,y,z,1,x,y,z,1,x,y,...)
    flaten = lambda l: [item for sublist in l for item in sublist]

    flat_verts = flaten(verts)
    flat_norms = flaten(norms)

    vertices[:len(flat_verts)] = flat_verts
    normals[:len(flat_norms)] = flat_norms
    mask[:len(flat_verts)] = np.ones([len(flat_verts)])
    if Only_vert:
        return vertices
    else :
        return vertices, normals, faces, mask
