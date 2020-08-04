from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import yaml
from torch.optim import lr_scheduler
from torchvision import transforms
from model import PCB
from training_function import train_model, save_network
from shutil import copyfile
from torchvision.datasets import ImageFolder

version =  torch.__version__


######################################################################
#Arg setting

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--name',default='pcb', type=str,help='save model name')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  1')
parser.add_argument('--data_path',default='../dataset/Market1501',type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
opt = parser.parse_args()

data_path = opt.data_path
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = int(opt.gpu_ids)

torch.cuda.set_device(gpu_ids)
cudnn.benchmark = True

#transform of data augmentation
transform_train = [
    transforms.Resize((384,192), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_val = [
    transforms.Resize(size=(384,192),interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

transform_train = transforms.Compose( transform_train )
transform_val = transforms.Compose( transform_val )

#load dataset
datasets = {}
datasets['train'] = ImageFolder(os.path.join(data_path, 'train'),
                                               transform_train)
datasets['val'] = ImageFolder(os.path.join(data_path, 'val'),
                                             transform_val)

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True)
              for x in ['train', 'val']}



#number of dataset's samples
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
#number of class
number_of_class = len(datasets['train'].classes)
opt.nclasses = number_of_class

use_gpu = torch.cuda.is_available()

if not use_gpu:
    print('Training process will run in CPU')




######################################################################
# Training model

model = PCB(number_of_class)

ignored_params = list(map(id, model.model.fc.parameters() ))
ignored_params += (list(map(id, model.classifier0.parameters() ))
                 +list(map(id, model.classifier1.parameters() ))
                 +list(map(id, model.classifier2.parameters() ))
                 +list(map(id, model.classifier3.parameters() ))
                 +list(map(id, model.classifier4.parameters() ))
                 +list(map(id, model.classifier5.parameters() ))
                  )
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

#learning rates of the backbone and classifier are different
optimizer_ft = optim.SGD([
         {'params': base_params, 'lr': 0.1*opt.lr},
         {'params': model.model.fc.parameters(), 'lr': opt.lr},
         {'params': model.classifier0.parameters(), 'lr': opt.lr},
         {'params': model.classifier1.parameters(), 'lr': opt.lr},
         {'params': model.classifier2.parameters(), 'lr': opt.lr},
         {'params': model.classifier3.parameters(), 'lr': opt.lr},
         {'params': model.classifier4.parameters(), 'lr': opt.lr},
         {'params': model.classifier5.parameters(), 'lr': opt.lr},
     ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# decay lr by 0.1 every 40 epochs
our_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
dir_name = os.path.join('./save_models',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save settings by yaml
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

model = model.cuda()

criterion = nn.CrossEntropyLoss()

model = train_model(model, criterion, optimizer_ft, our_lr_scheduler, dataloaders, batchsize = opt.batchsize,
                    dataset_sizes = dataset_sizes, use_gpu = use_gpu, num_epochs=60, name = name, gpu_ids=gpu_ids)