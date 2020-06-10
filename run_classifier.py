import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
from tqdm import tqdm
import argparse
#GENERAL PARAMETERS
input_size = 224

#NUMBER OF CLASSES
num_classes = 12

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--inp', type=str)
parser.add_argument('--out', type=str)

args = parser.parse_args()

#DATA_DIR
data_dir = args.inp

#RESULT_DIR
res_dir = args.out

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


classes = ["Arabian", "African", "East Asian", "South East Asian", "Mediterranean", "Scandinavian", "East European", "Central Asian", "Central European", "UK", "South American", "Carribean"]

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
    for entry in tqdm(os.scandir(data_dir)):
        img_original = Image.open(data_dir+entry.name)
        img = transform(img_original)
        img = img.unsqueeze(0)
        inputs = img.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        sub_idx = predicted.cpu().numpy()[0]
        sub = classes[sub_idx]
        new_img = img_original.resize((250,250))
        new_img.save(res_dir+sub+'/'+entry.name)
        #shutil.copyfile(entry.path, res_dir+sub+'/'+entry.name)

