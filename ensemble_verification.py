from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

from ensemble_network.resnet_18 import avg_ensemble_3_resnet18

bs = 1

## Step 1: Define computational graph by implementing forward()
model = avg_ensemble_3_resnet18(100)

# Load the pretrained weights
model_path = 'ensemble_network/models/best_model.pth'
model.load_state_dict(torch.load(model_path)['net'])

print(torch.load(model_path)['acc'])

model.cuda()
model.eval()

## Step 2: Prepare dataset as usual
DATAROOT = '/data/datasets/pytorch_datasets/CIFAR100'
test_data = torchvision.datasets.CIFAR100(root=DATAROOT, train=False, download=True, transform=torchvision.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=bs,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
# For illustration we only use 2 image from dataset
n_classes = 100

for batch_idx, (inputs, targets) in enumerate(testloader):
    if True:
        inputs, targets = inputs.cuda(), targets.cuda()

    if batch_idx < 2:

        ## Step 3: wrap model with auto_LiRPA
        # The second parameter is for constructing the trace of the computational graph, and its content is not important.
        
        model = BoundedModule(model, inputs, device="cuda")

        ## Step 4: Compute bounds using LiRPA given a perturbation
        eps = 0.3
        norm = np.inf
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        image = BoundedTensor(inputs, ptb)
        # Get model prediction as usual
        pred = model(image)
        label = torch.argmax(pred, dim=1).cpu().numpy()
        # Compute bounds
        lb, ub = model.compute_bounds()

        ## Step 5: Final output
        pred = pred.detach().cpu().numpy()
        lb = lb.detach().cpu().numpy()
        ub = ub.detach().cpu().numpy()
        for i in range(N):
            print("Image {} top-1 prediction {}".format(i, label[i]))
            for j in range(n_classes):
                print("f_{j}(x_0) = {fx0:8.3f},   {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(j=j, fx0=pred[i][j], l=lb[i][j], u=ub[i][j]))
            print()