import numpy as np

import torch
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter1d

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def init_regul(source_vertices, source_faces):
    sommet_A_source = source_vertices[source_faces[:, 0]]
    sommet_B_source = source_vertices[source_faces[:, 1]]
    sommet_C_source = source_vertices[source_faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target

def get_target(vertice, face, size):
    target = init_regul(vertice,face)
    target = np.array(target)
    target = torch.from_numpy(target).float().cuda()
    #target = target+0.0001
    target = target.unsqueeze(1).expand(3,size,-1)
    return target

def compute_score(points, faces, target):
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    return torch.mean(score)

def total_energy(data):

    pos = data 

    dx= pos[1:] - pos[:-1] 
    T = dx / dx.norm()
    
    return T

def motion_statistics(data):
    Acc = []
    Vel = []
    #Unit_T = []
    Normal_T = []
    Pos = []
    Time = []

    T = total_energy(data)
    Unit_T = T

    return Acc, Vel, Unit_T, Normal_T, Pos, Time


def motion_analysis(source, target):

    source = np.transpose(source, (1, 0, 2))
    target = np.transpose(target, (1, 0, 2))
    
    S_Acc, S_Vel, _, _ = motion_statistics(source)
    T_Acc, T_Vel, _, _ = motion_statistics(target)
    

    Acc = (np.array(S_Acc) - np.array(T_Acc))**2
    Vel = (np.array(S_Vel) - np.array(T_Vel))**2

    Acc = np.array(Acc).mean(axis=0)
    Vel = np.array(Vel).mean(axis=0)

    acc_loss = Acc.sum(axis=0)
    vel_loss = Vel.sum(axis=0)

    return acc_loss, vel_loss

    
