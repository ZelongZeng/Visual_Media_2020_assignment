from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
import scipy.io
import yaml
import math
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from model import PCB, PCB_test

######################################################################
#Arg setting

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_path', default='../Market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='pcb', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')

opt = parser.parse_args()

# load the training setting
config_path = os.path.join('./save_models', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
if 'nclasses' in config:
    opt.nclasses = config['nclasses']
gpu_ids = int(opt.gpu_ids)
name = opt.name
test_path = opt.test_path


torch.cuda.set_device(gpu_ids)
cudnn.benchmark = True

######################################################################
# Load Data

data_transforms = transforms.Compose([
    transforms.Resize((384, 192), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_path = test_path

image_datasets = {x: ImageFolder(os.path.join(data_path, x), data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in ['gallery', 'query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Load model pth

def load_network(network):
    save_path = os.path.join('./save_models', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()  # we have six parts

        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)


######################################################################
# Load Collected data Trained model
print('-------test-----------')
model_structure = PCB(opt.nclasses)

# if opt.fp16:
#    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model = PCB_test(model)
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])


# Save to Matlab for check
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('pytorch_result.mat', result)

print(opt.name)
result = './save_models/%s/result.txt' % opt.name
os.system('python evaluate.py | tee -a %s' % result)