#!/usr/bin/env python

import os, sys
import numpy as np
import string

#  ----------------------------- Variables  -----------------------------------  #

batch_size = 1
epocs = 100000

epochs_loss_tolerance = 10000
latent_space_dimension = 256

# options
# --train-model
# --save-model
# --visualise-latentspace
args = '--train-model --save-model'
#args = '--test-model'

# Data loader options
# full - loads entire dataset into memory
# light - Loads dataset per batch size
# binary - loads dataset in binary format
data_loader = 'full'

binary_file = '6charactersdataset.hdf5'

#  ----------------------------------------------------------------------------  #

# 0 ----------------------------------------------------------------------------  #
#  ---------------------- Checkpoints/Saving/log paths ------------------------  #
output_path = './results'
resume = './logs/'
#  ----------------------------------------------------------------------------  #
#  ----------------------------------------------------------------------------  #


#  ----------------------------------------------------------------------------  #
#  ------------------------ Labels --------------------------------------------  #
labels = '"female male"'

#  ----------------------------------------------------------------------------  #
#  ----------------------------------------------------------------------------  #


#  ----------------------------------------------------------------------------  #
#  ------------------------ Mesh Dataset Paths  -------------------------------  #
# Mesh Train Paths
train_data = '"./dataset/female/Subject_1_F_1_MoSh/female%d.obj ' \
            './dataset/male/Subject_1_F_1_MoSh/male%d.obj"'

# Training Sequence Lengths
train_data_lenght = '"1 9 1 9"'



#  ----------------------------------------------------------------------------  #
#  ----------------------------------------------------------------------------  #


#valgrind --leak-check=yes 
cmd = "python main.py " \
      "--binary-file %s --data-loader %s --latent-space-dimension %d --batch-size %d --epochs %d --epochs-loss-tolerance %d --resume %s --output-path %s " \
      "--train-data %s --train-data-lenght %s " \
      "--labels %s %s"%(binary_file, data_loader, latent_space_dimension, batch_size, epocs, epochs_loss_tolerance, resume, output_path,
                        train_data, train_data_lenght,
                        labels, args)

#  ----------------------------------------------------------------------------  #
#  ----------------------------------------------------------------------------  #

print (cmd)
os.system(cmd)


