import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
#GENERAL PARAMETERS
input_size = 224

#NUMBER OF CLASSES
num_classes = 9

#DATA_DIR
data_dir = '/content/drive/My Drive/stylegan2-colab/stylegan2/results/00003-generate-images/pics/'

#RESULT_DIR
res_dir = '/content/drive/My Drive/FMGAN-V2/'

#CHECK FOR GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TRANSFORMATIONS
transform = transforms.Compose(
    [transforms.Resize(input_size),
    transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#DEFINE DATA
#testset = torchvision.datasets.ImageFolder(data_dir, transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=1,
#                                         shuffle=False, num_workers=4)


classes = ['African:Caribbean', 'Asian (Central)', 'East Asian', 'Mediterranean:Hispanic', 'Mixed Race (Black:White)', 'North African:Middle Eastern', 'Northern European', 'Pacific Islander', 'South East Asian']

# LOAD MODEL
PATH = './FMGAN_net.pth'
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, num_classes)
model.load_state_dict(torch.load(PATH))
model.eval()

#TRANSFORM TO CUDA
model.to(device)

#CREATE SUBFOLDERS
[os.makedirs(res_dir+sub, exist_ok=True) for sub in classes]

#RUNNING
with torch.no_grad():
    for j, entry in enumerate(os.scandir(data_dir)):
        print(entry.name)
        img = Image.open(data_dir+entry.name)
        img = transform(img)
        print(img.shape)
        inputs = img.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted.cpu().numpy())
        break
#TODO: sort dataset

