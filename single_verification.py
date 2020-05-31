import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from single_network.resnet18 import resnet18

model = resnet18().cuda()

pretrained_model = 'single_network/model/best_model.pth'
model.load_state_dict((torch.load(pretrained_model))['net'])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
DATAROOT = '/data/CIFAR100'
test_data = torchvision.datasets.CIFAR100(root=DATAROOT, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

n_classes = 100
for batch_idx, (image, targets) in enumerate(testloader):
    image, targets = image.cuda(), targets.cuda()
    if batch_idx < 2:
        print(batch_idx)
        model = BoundedModule(model, torch.empty_like(image), device="cuda")
        eps = 0.3
        norm = np.inf
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        image = BoundedTensor(image, ptb)
        # Get model prediction as usual
        pred = model(image)
        label = torch.argmax(pred, dim=1).cpu().numpy()
        # Compute bounds
        lb, ub = model.compute_bounds()

        pred = pred.detach().cpu().numpy()
        lb = lb.detach().cpu().numpy()
        ub = ub.detach().cpu().numpy()
        for i in range(batch_idx):
            print("Image {} top-1 prediction {}".format(i, label[i]))
            for j in range(n_classes):
                print("f_{j}(x_0) = {fx0:8.3f},   {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(j=j, fx0=pred[i][j], l=lb[i][j], u=ub[i][j]))
            print()

