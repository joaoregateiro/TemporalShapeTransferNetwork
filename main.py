#!/usr/bin/env python

''' Main program for Generative Network from skeletal pose to multiple character meshes
Author: Joao Regateiro
'''

# System imports
import time
import sys
import argparse
import os, datetime

os.environ["MKL_THREADING_LAYER"] = "GNU"

# Torch imports
import torch
from torch import nn, optim
from torchviz import make_dot


# Common Classes
from common.network.LossFunctions import *
from common.MeshTools import *
from temporal_batching import *
from model import StyleShapeTransferGenerator, Discriminator
import utils as utils


from scipy.spatial.distance import directed_hausdorff

#Shape analysis
import igl

# Globals Variables
logdir = ''
# Constant Variables

#gbatch_size     = 1
#gbatch_size     = 2
#gbatch_size     = 3
#gbatch_size     = 4
gbatch_size     = 5


adv_constant    = 1 
recon_constant  = 2 
edge_constant   = 0.5 
dist_constant   = 0.5 
motion_constant = 2

lrate_G = 0.001
lrate_D = 0.001

epochs = 1001

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



def train_model(optimizer_D, optimizer_G, model, Dmodel, dataloader):

    real_label = 1.
    fake_label = 0.

    model.train()

    total_loss          = 0
    total_edge_loss     = 0
    total_distance_loss = 0
    total_rec_loss      = 0

    total_errD_loss = 0
    total_errG_loss = 0

    for j, data in enumerate(dataloader, 0):
        optimizer_D.zero_grad()

        pose_points, random_sample, gt_points, identity_points, new_face = data
        
        # Separete the frames into bataches
        pose_points     = pose_points.reshape(-1,gbatch_size,6890,3).squeeze(0)
        gt_points       = gt_points.reshape(-1,gbatch_size,6890,3).squeeze(0)
        identity_points = identity_points.reshape(-1,gbatch_size,6890,3).squeeze(0)
        
        id_points = identity_points

        identity_points = identity_points.transpose(2, 1)
        identity_points = identity_points.to(device, non_blocking=True)

        #for frame_t, pose_t in enumerate(pose_points[0]):
        pose = pose_points.transpose(2, 1)
        pose = pose.to(device, non_blocking=True)

        gt = gt_points.to(device, non_blocking=True)

        GT_Acc, GT_Vel, GT_Unit_T, GT_Normal_T, _, _ = utils.motion_statistics(gt.reshape(gbatch_size,-1)) 

        requires_grad(model, False)
        requires_grad(Dmodel, True)

        # -------------------------------------------#
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Run Discriminator through real data
        prob_real = Dmodel(gt.view(gbatch_size, -1))

        label = torch.full((prob_real.size(0),prob_real.size(1)), real_label, dtype=torch.float, device=device)

        errD_real = bce_loss_function(prob_real, label)
        
        #.backpropagate()
        errD_real.backward()
        D_G_z1 = prob_real.mean().item()
        D_x = prob_real.mean().item()

        # -------------------------------------------#
        # Run Generator to generate "fake" data

        fake_sample = model(model.encoder(pose), identity_points)
        prob_fake   = Dmodel(fake_sample.reshape(gbatch_size, -1))

        label.fill_(fake_label)
        errD_fake = bce_loss_function(prob_fake, label)

        #.backpropagate()
        errD_fake.backward()

        errD = errD_real + errD_fake
        # Discriminator optimiser step()
        optimizer_D.step()

        # -------------------------------------------#
        requires_grad(model, True)
        requires_grad(Dmodel, True)
        optimizer_G.zero_grad()
        # (2) Update G network: maximize log(D(G(z)))
        # Run Discriminator through "fake" data
        pose_z = model.encoder(pose)
        pointsReconstructed = model(pose_z, identity_points)

        prob_style = Dmodel(pointsReconstructed.reshape(gbatch_size, -1))
        label.fill_(real_label)
        shape_loss = bce_loss_function(prob_style, label)
    
        recon_loss = torch.sum((pointsReconstructed - gt) ** 2)

        S_Acc, S_Vel, S_Unit_T, S_Normal_T, _, _ = utils.motion_statistics(pointsReconstructed.reshape(gbatch_size,-1)) 

        motion_loss = torch.sum((S_Unit_T - GT_Unit_T) ** 2)

        edg_loss = 0
        for i in range(gbatch_size):
            f = new_face[0].cpu().numpy()
            v = identity_points[i].transpose(0, 1).cpu().numpy()
            edg_loss = edg_loss + utils.compute_score(pointsReconstructed[i].unsqueeze(0), f, utils.get_target(v, f, 1))

        edg_loss = edg_loss / len(random_sample)

        distance_r = torch.triu(torch.cdist(pointsReconstructed, pointsReconstructed, p=2))
        distance_g = torch.triu(torch.cdist(gt, gt, p=2))

        distance_loss = torch.sum((distance_r - distance_g)**2) / 23729160

        # Losses and ablations
        # Arc 1
        #errG = recon_constant * recon_loss 

        # Arc 2
        #errG = ((adv_constant * shape_loss) + (recon_constant * recon_loss) )

        # Arc 3
        #errG = (adv_constant * shape_loss) + (recon_constant * recon_loss) + (edge_constant * edg_loss)

        # Arc 4 Final loss
        errG = (adv_constant * shape_loss) + (recon_constant * recon_loss) + (edge_constant * edg_loss) + (dist_constant * distance_loss)

        # Arc 5
        #errG =  (adv_constant * shape_loss) + (recon_constant * recon_loss) + (edge_constant * edg_loss) + (dist_constant * distance_loss) + (motion_constant * motion_loss)

        # .backpropagate()
        errG.backward()

        D_G_z2 = prob_style.mean().item()

        # Generator optimiser step()
        optimizer_G.step()
        # -------------------------------------------#

        total_loss = total_loss + errG.item() #+ errD.item()
        total_rec_loss = total_rec_loss + recon_loss.item()
        total_edge_loss = total_edge_loss + edg_loss.item()
        total_distance_loss = total_distance_loss + distance_loss.item()
        
        total_errD_loss = total_errD_loss + errD.item()
        total_errG_loss = total_errG_loss + errG.item()

        print('Train: [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f - rec_loss %.4f \tD(x): %.4f\tD(G(z)): %.4f / %.4f \tmotion: %.4f'% (epoch, epochs, j, len(dataloader), errD.item(), errG.item(), recon_loss.item(), D_x, D_G_z1, D_G_z2, motion_loss.item()))

        if epoch%10 == 0 and j == 0:
            element = pointsReconstructed.detach().cpu().numpy()[0]
            element2 = pose_points.detach().cpu().numpy()[0]
            element3 = id_points.detach().cpu().numpy()[0]
            element4 = gt.detach().cpu().numpy()[0]
            normals = getNormal(element, np.array(new_face[0]))

            save_mesh(logdir + "/train_v2%05d.obj",
                      element,
                      epoch,
                      new_face[0], normals)

            save_mesh(logdir + "/tr_pose%05d.obj",
                      element2,
                      epoch,
                      new_face[0], normals)

            save_mesh(logdir + "/tr_id%05d.obj",
                      element3,
                      epoch,
                      new_face[0], normals)

            save_mesh(logdir + "/tr_gt_v2%05d.obj",
                      element4,
                      epoch,
                      new_face[0], normals)

    total_loss = total_loss / (j + 1)
    total_rec_loss = total_rec_loss / (j + 1)
    total_edge_loss = total_edge_loss / (j + 1)
    total_distance_loss = total_distance_loss / (j + 1)

    total_errD_loss = total_errD_loss / (j + 1)
    total_errG_loss = total_errG_loss / (j + 1)

    return total_loss, total_rec_loss, total_edge_loss, total_distance_loss, total_errD_loss, total_errG_loss

def test_model(model, dataloader):

    model.eval()

    total_loss = 0
    total_edge_loss = 0
    total_distance_loss = 0
    total_rec_loss = 0
    for j, data in enumerate(dataloader, 0):
        pose_points, random_sample, gt_points, identity_points, new_face = data

        pose = pose_points
        id_points = identity_points

        pose_points = pose_points.transpose(2, 1)
        pose_points = pose_points.to(device, non_blocking=True)

        identity_points = identity_points.transpose(2, 1)
        identity_points = identity_points.to(device, non_blocking=True)

        gt_points = gt_points.to(device, non_blocking=True)

        pose_z = model.encoder(pose_points)
        pointsReconstructed = model(pose_z, identity_points)
        recon_loss = torch.sum((pointsReconstructed - gt_points) ** 2)

        edg_loss = 0
        for i in range(len(random_sample)):
            f = new_face[i].cpu().numpy()
            v = identity_points[i].transpose(0, 1).cpu().numpy()
            edg_loss = edg_loss + utils.compute_score(pointsReconstructed[i].unsqueeze(0), f,
                                                      utils.get_target(v, f, 1))

        edg_loss = edg_loss / len(random_sample)

        distance_r = torch.triu(torch.cdist(pointsReconstructed, pointsReconstructed, p=2))
        distance_g = torch.triu(torch.cdist(gt_points, gt_points, p=2))

        distance_loss = torch.sum((distance_r - distance_g)**2) / 23732605


        errG = recon_loss + edg_loss + distance_loss 


        total_loss = total_loss + errG.item()
        total_rec_loss = total_rec_loss + recon_loss.item()
        total_edge_loss = total_edge_loss + edg_loss.item()
        total_distance_loss = total_distance_loss + distance_loss.item()

        print('Test: [%d/%d][%d/%d]\tLoss_G: %.4f - rec_loss %.4f \tEdge_loss: %.4f\tDist_loss: %.4f '% (epoch, epochs, j, len(dataloader), errG.item(), recon_loss.item(), edg_loss, distance_loss))

        if epoch%10 == 0 and j == 0:
            element = pointsReconstructed.detach().cpu().numpy()[0]
            element2 = pose.detach().cpu().numpy()[0]
            element3 = id_points.detach().cpu().numpy()[0]
            element4 = gt_points.detach().cpu().numpy()[0]
            normals = getNormal(element, np.array(new_face[0]))

            save_mesh(logdir + "/test%05d.obj",
                      element,
                      epoch,
                      new_face[0], normals)

        total_loss = total_loss / (j + 1)
        total_rec_loss = total_rec_loss / (j + 1)
        total_edge_loss = total_edge_loss / (j + 1)
        total_distance_loss = total_distance_loss / (j + 1)

    return total_loss, total_rec_loss, total_edge_loss, total_distance_loss


if __name__ == "__main__":

    print("Pose2Mesh 4D Generative Network.")
    parser = argparse.ArgumentParser(description='Pose2Mesh 4D Generative Network!', add_help=True)
    parser.add_argument('--latent-space-dimension', type=int, default=128, metavar='N',
                        help='Input latent space dimensions (default: 128)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs-loss-tolerance', type=int, default=10, metavar='N',
                        help='number of epochs to stop for no improved loss (default: 100)')

    parser.add_argument('--output-path',
                        help='Path to where the data will be stored.')

    # Labels arguments -----------------------------------------------------------------#

    parser.add_argument('--labels',
                        help='List of labels for each sequence')
    # ---------------------------------------------------------------------------------#

    # Mesh arguments -----------------------------------------------------------------#
    parser.add_argument('--train-data',
                        help='List of paths to the training data')
    parser.add_argument('--train-data-lenght',
                        help='List of indices intervals for training data')

    parser.add_argument('--test-data',
                        help='List of paths to the test data')
    parser.add_argument('--test-data-lenght',
                        help='List of indices intervals for test data')
    # ---------------------------------------------------------------------------------#

    # Skeleton arguments --------------------------------------------------------------#
    parser.add_argument('--train-skel-data',
                        help='List of paths to the training data')
    parser.add_argument('--train-skel-data-lenght',
                        help='List of indices intervals for training data')

    parser.add_argument('--test-skel-data',
                        help='List of paths to the test data')
    parser.add_argument('--test-skel-data-lenght',
                        help='List of indices intervals for test data')
    # ----------------------------------------------------------------------------------#

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--train-model', action='store_true', default=False,
                        help='For training the current Model')
    parser.add_argument('--test-model', action='store_true', default=False,
                        help='For training the current Model')

    parser.add_argument('--visualize-layers', action='store_true', default=False,
                        help='enables debugging for layers activation')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None')

    parser.add_argument('--visualise-latentspace', action='store_true', default=False,
                        help='Records the latent space prograssion as a graph image of the frist dimession of the latent vectors')

    parser.add_argument('--data-loader', default='light', type=str,
                        help='Type of data loader: full, light, binary')

    parser.add_argument('--binary-file', default='6charactersdataset.hdf5', type=str,
                        help='Binary file containing dataset to be loaded')

    args = parser.parse_args()

    if not len(sys.argv) > 1:
        parser.print_help()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    workers = 16
    if args.data_loader == 'binary':
        workers = 0

    kwargs = {'num_workers': workers, 'pin_memory': True} if args.cuda else {}

    device = torch.device("cuda")

    visualise_latentspace_tolerance = 1000

    if args.train_model:

        logdir = os.path.join(args.output_path + '/logs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if not os.path.isdir(args.output_path + '/logs'):
            os.makedirs(args.output_path + '/logs')
            print('Created %s, folder: ' % args.output_path + '/logs')
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
            print('Created %s, folder: ' % logdir)
        if not os.path.isdir(logdir + '/saved_model'):
            os.makedirs(logdir + '/saved_model')
            print('Created %s, folder: ' % logdir + '/saved_model')


        ConsoleLossLog = open(logdir + "/ConsoleLog.txt", "w")
        LossLog = open(logdir + "/LossLog.txt", "w")

        LossLog.write("#This file contains the computed losses per epoch\n")
        LossLog.write("#Epoch train_loss val_loss errD_loss errG_loss rec_loss test_rec_loss edge_loss test_edge_loss distance_loss test_rec_loss\n")

        print("Prepare model for Training!")

        # Create Model and Optimizer
        model=StyleShapeTransferGenerator()
        model.to(device, non_blocking=True)

        Dmodel = Discriminator()
        Dmodel.to(device, non_blocking=True)

        optimizer_G = optim.Adam(model.parameters(), lr=lrate_G, betas=(0.9, 0.999),
                                  eps=1e-08)
        optimizer_D = optim.Adam(model.parameters(), lr=lrate_D,  betas=(0.9, 0.999),
                                  eps=1e-08)

        dataset = TrainDataLoader(args.train_data, args.train_data_lenght,
                                        args.train_skel_data, args.train_skel_data_lenght,
                                        args.labels, train=True, shuffle_point=False)

        sampler = RandomSampler (dataset, gbatch_size)
        batch_sampler = BatchSampler (sampler, gbatch_size, True)


        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers, sampler=batch_sampler)
        torch.autograd.set_detect_anomaly(True)

        LossLog.write("Start Training...\n")
        ConsoleLossLog.write("Start Training...\n")
        ConsoleLossLog.write("Parameters:\tlrate_G: %f\tlrate_D: %f\tadv_constant: %f\trecon_constant: %f\tedge_constant: %f\tdist_constant: %f  \n" % (lrate_G,lrate_D,  adv_constant, recon_constant, edge_constant, dist_constant))

        for epoch in range(epochs):

            start = time.time()

            total_loss, total_rec_loss, total_edge_loss, total_distance_loss, total_errD_loss, total_errG_loss = train_model(optimizer_D, optimizer_G, model, Dmodel, dataloader)

            test_total_loss, test_total_rec_loss, test_total_edge_loss, test_total_distance_loss = 0, 0, 0, 0 # test_model(model, dataloader)
            
            print('####################################')
            print(epoch)
            print(time.time() - start)

            print('train_total_loss: %.04f test_total_loss: %.04f'% (total_loss, test_total_loss))
            print('train_total_rec_loss: %.04f test_total_rec_loss: %.04f'% (total_rec_loss, test_total_rec_loss))
            print('train_total_edge_loss: %.04f test_total_edge_loss: %.04f'% (total_edge_loss, test_total_edge_loss))
            print('train_total_distance_loss: %.04f test_total_distance_loss: %.04f'% (total_distance_loss, test_total_distance_loss))


            ConsoleLossLog.write("Epoch [%d/%d] train_loss: %.04f  val_loss: %.04f  \n" % (epoch, epochs, total_loss, test_total_loss))
            ConsoleLossLog.write("\t\t total_errD_loss: %.04f  total_errG_loss: %.04f  \n" % (total_errD_loss, total_errG_loss))
            ConsoleLossLog.write("\t\t total_rec_loss: %.04f  test_total_rec_loss: %.04f  \n" % (total_rec_loss, test_total_rec_loss))
            ConsoleLossLog.write("\t\t total_edge_loss: %.04f  test_total_edge_loss: %.04f  \n" % (total_edge_loss, test_total_edge_loss))
            ConsoleLossLog.write("\t\t total_distance_loss: %.04f  test_total_rec_loss: %.04f  \n" % (total_distance_loss, test_total_distance_loss))

            LossLog.write("%d  %.04f  %.04f %.04f  %.04f %.04f  %.04f %.04f  %.04f %.04f  %.04f\n" % (epoch, total_loss, test_total_loss, total_errD_loss, total_errG_loss, total_rec_loss, test_total_rec_loss, total_edge_loss, test_total_edge_loss, total_distance_loss, test_total_distance_loss))

            ConsoleLossLog.flush()
            LossLog.flush()
            print('####################################')

            if (epoch) % 100 == 0:
                save_path = logdir + '/saved_model/' + str(epoch) + '.model'
                torch.save(model.state_dict(), save_path)


    if args.test_model:
        # Create Model and Optimizer
        model = StyleShapeTransferGenerator()
        model.cuda()


        dataset = TrainDataLoader(args.train_data, args.train_data_lenght,
                            args.train_skel_data, args.train_skel_data_lenght,
                            args.labels, train=False, shuffle_point=False)

        sampler = RandomSampler (dataset, gbatch_size)
        batch_sampler = BatchSampler (sampler, gbatch_size, True)


        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers, sampler=batch_sampler)

        if args.resume:
            checkpoint_file_resume = os.path.join(args.resume, "1000.model")
            if os.path.isfile(checkpoint_file_resume):
                print('=> loading checkpoint %s' % checkpoint_file_resume)
                checkpoint = torch.load(checkpoint_file_resume)
                model.load_state_dict(checkpoint)
                print('=> loaded checkpoint %s' % checkpoint_file_resume)
            else:
                print('=> no checkpoint found at %s' % checkpoint_file_resume)

        model.eval()

        hausdorff = 0.0
        recon_loss = 0.0
        laplacian = 0.0
        edg_loss = 0.0
        Acc_loss = 0.0
        Vel_loss = 0.0

        for j, data in enumerate(dataloader, 0):
            print("index [%dx%d]" % (len(dataloader), j))
            with torch.no_grad():
                pose_points, random_sample, gt_points, identity_points, new_face = data

                # Separete the frames into bataches
                pose_points     = pose_points.reshape(-1,gbatch_size,6890,3).squeeze(0)
                gt_points       = gt_points.reshape(-1,gbatch_size,6890,3).squeeze(0)
                identity_points = identity_points.reshape(-1,gbatch_size,6890,3).squeeze(0)
        
                id_points = identity_points

                identity_points = identity_points.transpose(2, 1)
                identity_points = identity_points.to(device, non_blocking=True)

                pose = pose_points.transpose(2, 1)
                pose = pose.to(device, non_blocking=True)

                gt = gt_points.to(device, non_blocking=True)

                pose_z = model.encoder(pose)
                noise = torch.randn(pose_z.shape, device=device)

                pointsReconstructed = model(pose_z.detach(), identity_points.detach())

                #acc, vel = utils.motion_analysis(pointsReconstructed.detach().cpu().numpy(), gt_points.detach().cpu().numpy())
                #print(Vel_loss)
                #print(Acc_loss)
                #Acc_loss += 0 if math.isnan(acc) else acc
                #Vel_loss += vel

                for i in range(0,1):
                    element = pointsReconstructed.detach().cpu().numpy()[i]
                    element2 = pose_points.detach().cpu().numpy()[i]
                    element3 = id_points.detach().cpu().numpy()[i]
                    element4 = gt_points.detach().cpu().numpy()[i]
                    normals = getNormal(element, np.array(new_face[0]))

                    hausdorff = hausdorff + max(directed_hausdorff(element, element4)[0], directed_hausdorff(element4, element)[0])

                    l1 = igl.cotmatrix(element, new_face[0].cpu().numpy())
                    laplaceR = convert(l1)

                    l2 = igl.cotmatrix(element3, new_face[0].cpu().numpy())
                    laplaceID = convert(l2)
                    laplacian += np.sqrt(np.abs(np.sum((l1 - l2) ** 2)))

                    print('laplacian Loss: %.4f ' % laplacian)
                    print('hausdorff Loss: %.4f ' % hausdorff)
                    recon = np.sqrt(np.mean((element - element4) ** 2))
                    recon_loss += recon
                    print('Loss: %.4f ' % recon)


                    for i in range(len(random_sample)):
                        f = new_face[0].cpu().numpy()
                        v = identity_points[i].transpose(0, 1).cpu().numpy()

                        edg_loss = edg_loss + utils.compute_score(pointsReconstructed[i].unsqueeze(0).cuda(), f, utils.get_target(v, f, 1))

        print(hausdorff/(len(dataloader)))
        print(recon_loss/(len(dataloader)))
        print(laplacian/(len(dataloader)))
        print(edg_loss.cpu()/(len(dataloader)))
        print(Acc_loss)
        #print(Acc_loss/(len(dataloader)))
        #print(Vel_loss/(len(dataloader)))
        print(len(dataloader))

    print("Program Finished!")
