##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Training code for LUCIR """
import torch
import tqdm
import numpy as np
import torch.nn as nn
import wandb
from torch.optim import lr_scheduler
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

cur_features = []
ref_features = []
old_scores = []
new_scores = []

class MyDataset(Dataset):
    def __init__(self, data, target, orig, transform):
        self.data = data
        self.target = target
        self.orig = orig
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        o = self.orig[index]

        if self.transform:
            x = self.transform(x)
            
        return x, y, o
    
    def __len__(self):
        return len(self.data)

def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y


def incremental_train_and_eval(the_args, epochs, fusion_vars, ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, \
    tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, iteration, \
    start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list, the_lambda, dist, \
    K, lw_mr, balancedloader, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set the 1st branch reference model to the evaluation mode
    ref_model.eval()

    # Get the number of old classes
    num_old_classes = ref_model.fc.out_features
    kmeans = KMeans(init="k-means++",n_clusters=num_old_classes+10,n_init=10,max_iter=500,random_state=42)

    images = None
    orig_labels = None
    test_images = None
    test_orig_labels = None

    for _, (inputs, orig_targets) in enumerate(trainloader):

        if images is None:
            images = inputs
            orig_labels = orig_targets
        else:
            images = torch.vstack((images, inputs))
            orig_labels = torch.hstack((orig_labels, orig_targets))

    all_images = images
    all_orig_labels = orig_labels

    for _, (inputs, orig_targets) in enumerate(testloader):

        if test_images is None:
            test_images = inputs
            test_orig_labels = orig_targets
        else:
            test_images = torch.vstack((test_images, inputs))
            test_orig_labels = torch.hstack((test_orig_labels, orig_targets))

        all_images = torch.vstack((all_images, inputs))
        all_orig_labels = torch.hstack((all_orig_labels, orig_targets))
    
    print(type(orig_labels))
    print(orig_labels) 
    print(images.shape)
    
    b1_model.eval()

    transform = transforms.ToTensor()

    with torch.no_grad():
        inputs = images.to(device)
        _, _,features = b1_model(inputs, True)

    kmeans.fit(features.cpu().numpy())
    labels = kmeans.labels_

    print(labels)
    print('Generating')


    print('**********************************************')

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MyDataset(np.moveaxis(images.numpy(),1,3), torch.from_numpy(labels[0:images.shape[0]]), orig_labels, transform)

    loader = DataLoader(
    dataset,
    batch_size=96,
    num_workers=8,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

    test_dataset = MyDataset(np.moveaxis(test_images.numpy(),1,3), torch.from_numpy(labels[images.shape[0]:]), test_orig_labels, transform)

    test_loader = DataLoader(
    test_dataset,
    batch_size=96,
    num_workers=8,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
    )

    # unfreeze only pseudo
    for param in b1_model.parameters():
        param.requires_grad = False

    for param in b1_model.pseudo.parameters():
        param.requires_grad = True

    scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=[20], gamma=0.1)

    for epoch in range(40):
        print('Epoch ' + str(epoch))
        total_loss = 0
        correct = 0
        total = 0
        b1_model.train()
        scheduler.step()
        for batch_idx, (data, target, orig) in enumerate(loader):
            data = data.to(device)
            target = target.type(torch.LongTensor)
            target = target.to(device)    
            orig = orig.to(device)       

            tg_optimizer.zero_grad()
        
            outputs_fc,outputs_pseudo,_ = b1_model(data, True)

            loss = nn.CrossEntropyLoss()(outputs_pseudo, target) + nn.CrossEntropyLoss()(outputs_fc, orig) #should be for old samples only
            loss.backward()

            _, predicted = outputs_pseudo.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            total_loss+=loss.item()
            tg_optimizer.step()

        print('Train set: {}, train loss: {:.4f}, accuracy: {:.4f}'.format(len(trainloader), total_loss/(batch_idx+1), 100.*correct/total))
        
        # test_loss = 0
        # correct = 0
        # total = 0

        # b1_model.eval()
        # with torch.no_grad():
        #     for batch_idx, (inputs, targets, _) in enumerate(test_loader):
        #         targets = targets.type(torch.LongTensor)
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         _,outputs,_ = b1_model(inputs, True)
        #         loss = nn.CrossEntropyLoss()(outputs, targets)
        #         test_loss += loss.item()
        #         _, predicted = outputs.max(1)
        #         total += targets.size(0)
        #         correct += predicted.eq(targets).sum().item()
        # print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(test_loader), test_loss/(batch_idx+1), 100.*correct/total))
    
    # # unfreeze all params
    for param in b1_model.parameters():
        param.requires_grad = True


    # Get the features from the current and the reference model
    # handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
    # handle_cur_features = b1_model.fc.register_forward_hook(get_cur_features)
    handle_old_scores_bs = b1_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
    handle_new_scores_bs = b1_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    # If the 2nd branch reference is not None, set it to the evaluation mode
    if iteration > start_iteration+1:
        ref_b2_model.eval()

    for epoch in range(epochs):
        # Start training for the current phase, set the two branch models to the training mode
        b1_model.train()
        b2_model.train()

        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        # Set the counters to zeros
        correct = 0
        total = 0
    
        # Learning rate decay
        tg_lr_scheduler.step()
        fusion_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, pseudo_targets, orig_targets) in enumerate(loader):
     
            # Get a batch of training samples, transfer them to the device 

            pseudo_targets = pseudo_targets.type(torch.LongTensor)
            inputs, targets, pseudo = inputs.to(device), orig_targets.to(device), pseudo_targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()   

            # Forward the samples in the deep networks
            #outputs_fc, outputs_pseudo, cur_features=cur_model(inputs, True)
            outputs_fc, outputs_pseudo, cur_features = b1_model(inputs, True)

            with torch.no_grad():
                _, _, ref_features=ref_model(inputs, True)
                
            # embedding loss
            loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(device)) * (the_lambda)
            
            # Loss 2: classification loss
            loss2 = nn.CrossEntropyLoss()(outputs_fc, targets)

            # Clustering loss 
            loss3 = nn.CrossEntropyLoss()(outputs_pseudo, pseudo)

            # Loss 3: margin ranking loss
            outputs_bs = torch.cat((old_scores, new_scores), dim=1)
            assert(outputs_bs.size()==outputs_fc.size())
            gt_index = torch.zeros(outputs_bs.size()).to(device)
            gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
            gt_scores = outputs_bs.masked_select(gt_index)
            max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
            hard_index = targets.lt(num_old_classes)
            hard_num = torch.nonzero(hard_index).size(0)
            if hard_num > 0:
                gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                max_novel_scores = max_novel_scores[hard_index]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                loss4 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1), torch.ones(hard_num*K).to(device)) * lw_mr
            else:
                loss4 = torch.zeros(1).to(device)

            # Sum up all looses
            loss = loss1 + loss2 + loss3 + loss4

            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += loss4.item()
            _, predicted = outputs_fc.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, train loss1: {:.4f}, train loss2: {:.4f}, train loss3: {:.4f}, train loss4: {:.4f}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1), train_loss4/(batch_idx+1), train_loss/(batch_idx+1), 100.*correct/total))
        
        # Update the aggregation weights
        b1_model.eval()
        b2_model.eval()
     
        for batch_idx, (inputs, targets) in enumerate(balancedloader):
            if batch_idx <= 500:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss.backward()
                fusion_optimizer.step()

        # Running the test for this epoch
        b1_model.eval()
        b2_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        wandb.log({"Test accuracy (FC)  ":  100.*correct/total})
        wandb.log({"Distil ":  train_loss1/(batch_idx+1)})
        wandb.log({"CE":  train_loss2/(batch_idx+1)})
        wandb.log({"Clustering":  train_loss3/(batch_idx+1)})

    # print("Removing register forward hook")
    # handle_ref_features.remove()
    # handle_cur_features.remove()
    handle_old_scores_bs.remove()
    handle_new_scores_bs.remove()
    return b1_model, b2_model
