''' Temporal batching for deep learning common functions
Author: Joao Regateiro
'''
# Os imports
import random

# Lib imports
import numpy as np
from sklearn import preprocessing
from scipy.spatial.transform import Rotation as R

# Torch imports
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

# Common imports
from common.loader.DataLoader import *


class RandomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.time_window = batch_size
        self.data_source = data_source
        self.batch_size = batch_size
        self.dataset_dict = data_source.dataset_dict


    def __iter__(self):
        num_batches = len (self.data_source) // self.batch_size
        while num_batches > 0:
        
            sampled = []
            # Random Character
            random_character = random.choice(list(self.dataset_dict.keys()))

            # This line excludes the same character from the list.
            random_character2 = random.choice(list(set(self.dataset_dict.keys()) - set([random_character])))

            random_beta = random.choice(list(self.dataset_dict[random_character].keys()))

            random_poselist = random.sample(range(0, len(self.dataset_dict[random_character][random_beta])), self.time_window)
            random_pose2list = random.sample(range(0,len(self.dataset_dict[random_character2][random_beta])), self.time_window)

            pose_index_list = [self.dataset_dict[random_character][random_beta][index] for index in random_poselist ]
            gt_index_list = [self.dataset_dict[random_character2][random_beta][index] for index in random_poselist ]
            id_index_list = [self.dataset_dict[random_character2][random_beta][index] for index in random_pose2list ]

            sampled.append(pose_index_list)
            sampled.append(id_index_list)
            sampled.append(gt_index_list)

            yield sampled
            num_batches -=1

    def __len__(self):
        return len(self.data_source)

class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size

class TrainDataLoader(data.Dataset):
    def __init__(self, obj_path_list, obj_path_index_list,
                 json_path_list, json_path_index_list,
                 labels_list, train,  npoints=6890, shuffle_point = False):
        self.time_window = 1

        self.meshList, _, _, self.faces, self.laplace = load_all_meshes_to_matrixv2(obj_path_list, obj_path_index_list, labels_list)
        
        self.dataset_dict ={}
        datapath2 = './dataset/%s'
        for l in self.laplace:
            self.dataset_dict[l] = {}
            for index, path in enumerate(self.meshList):
                data_path = datapath2%l
                if data_path in path:
                    for id in range(1,4):
                        sub_data_path = data_path + '/Subject_%d_F_'%id
                        if sub_data_path in path:
                            if id not in self.dataset_dict[l]:
                                self.dataset_dict[l][id] = []

                            self.dataset_dict[l][id].append(index)
                            #print("Character %s beta %d index %d"%(l, id, index))

        self.shuffle_point = shuffle_point 
        self.npoints = npoints

    def __getitem__(self, index):
        verts, _, faces, mask = load_mesh_to_matrix(self.meshList[index[1][0]], self.npoints*3)    

        identity_faces = faces
        index = np.sort(index)

        pose = [load_mesh_to_matrix(self.meshList[index], self.npoints*3, True) for index in index[0] ]
        verts = np.array(pose)

        verts = verts / 2.0
        pose_points = np.array(verts)
        pose_points = np.reshape(pose_points, (self.time_window, -1, 3))

        id = [load_mesh_to_matrix(self.meshList[index], self.npoints*3, True) for index in index[1] ]
        verts = np.array(id)

        verts = verts / 2.0
        identity_points = np.array(verts)
        identity_points = np.reshape(identity_points, (self.time_window, -1, 3))
        

        gt = [load_mesh_to_matrix(self.meshList[index], self.npoints*3, True) for index in index[2] ]
        verts = np.array(gt)

        verts = verts / 2.0
        gt_points = np.array(verts)
        gt_points = np.reshape(gt_points, (self.time_window,-1, 3))
       
        facelist = []
        for face in identity_faces:

            facelist.append(
                [face[0][0], face[1][0], face[2][0]])

        identity_faces = np.array(facelist)


        pose_points = torch.from_numpy(pose_points.astype(np.float32))
        identity_points = torch.from_numpy(identity_points.astype(np.float32))
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[:,random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[:,random_sample]
            
            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face
        

    def __len__(self):
        return 1000
