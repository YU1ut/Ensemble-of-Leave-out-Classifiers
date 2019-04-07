import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import pickle

def cal_in_cls(fold, nclasses, nnName):
    nsplit = 5
    In_classes = []
    np.random.seed(3)
    p1 = np.random.permutation(nclasses).tolist()
    nclass_split = int(nclasses/nsplit)
    Out_classes = p1[(fold - 1) * nclass_split : nclass_split * fold]
    for item in p1:
        if item not in Out_classes:
            In_classes.append(item)
    return In_classes


def testData(net1, criterion, testloaderIn, testloaderOut, in_dataset, out_dataset, noiseMagnitude1, temper, fold):
    t0 = time.time()
    print("Processing in-distribution images")

    norm = [63.0/255.0, 62.1/255.0, 66.7/255.0]
    nclasses = int(in_dataset[5:])
    nsplit = int(nclasses*0.8)

    N = len(testloaderIn)
    inclass = cal_in_cls(fold, nclasses, in_dataset)
    in_sfx = np.array([])
    in_pro = np.array([])
    out_sfx = np.array([])
    out_pro = np.array([])

########################################In-distribution###########################################
    for j, data in enumerate(testloaderIn):
        images, _ = data
        
        inputs = images.cuda().requires_grad_()
        outputs = net1(inputs)
        # print (inputs, outputs)
        o_output = np.zeros((images.size()[0], nclasses))

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.detach().cpu()
        nnOutputs = nnOutputs.numpy()
        #print (nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        for idx in range(nsplit):
            o_output[:, inclass[idx]] = nnOutputs[:, idx]
        in_sfx = np.vstack((in_sfx, o_output)) if in_sfx.size else o_output

        o_output = np.zeros((images.size()[0], nclasses))

        # Using temperature scaling
        outputs = outputs / temper
	
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs, axis=1)
        labels = torch.LongTensor(maxIndexTemp).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[:,0] = (gradient[:,0] )/(norm[0])
        gradient[:,1] = (gradient[:,1] )/(norm[1])
        gradient[:,2] = (gradient[:,2])/(norm[2])
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(tempInputs)
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.detach().cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        for idx in range(nsplit):
            o_output[:, inclass[idx]] = nnOutputs[:, idx]
        in_pro = np.vstack((in_pro, o_output)) if in_pro.size else o_output
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1, N, time.time()-t0))
            t0 = time.time()
        
    N = len(testloaderOut)
    t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloaderOut):
        images, _ = data
    
    
        inputs = images.cuda().requires_grad_()
        outputs = net1(inputs)
        

        o_output = np.zeros((images.size()[0], nclasses))
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.detach().cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        for idx in range(nsplit):
            o_output[:, inclass[idx]] = nnOutputs[:, idx]
        out_sfx = np.vstack((out_sfx, o_output)) if out_sfx.size else o_output
        
        # Using temperature scaling
        outputs = outputs / temper
  
        o_output = np.zeros((images.size()[0], nclasses))
  
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs, axis=1)
        labels = torch.LongTensor(maxIndexTemp).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[:,0] = (gradient[:,0] )/(norm[0])
        gradient[:,1] = (gradient[:,1] )/(norm[1])
        gradient[:,2] = (gradient[:,2])/(norm[2])
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(tempInputs)
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.detach().cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        for idx in range(nsplit):
            o_output[:, inclass[idx]] = nnOutputs[:, idx]
        out_pro = np.vstack((out_pro, o_output)) if out_pro.size else o_output
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1, N, time.time()-t0))
            t0 = time.time()
    # print (in_sfx.shape)
    # print (in_pro.shape)
    # print (out_sfx.shape)
    # print (out_pro.shape)
    data = {'in_sfx':in_sfx, 'in_pro':in_pro, 'out_sfx':out_sfx, 'out_pro':out_pro}
    pickle.dump(data, open(f"./results/{in_dataset}_{out_dataset}_{fold}.p", "wb"))




