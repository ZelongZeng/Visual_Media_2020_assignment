from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import os


version =  torch.__version__


######################################################################
# Save model

def save_network(network, epoch_label, name, gpu_ids):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./save_models',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids)

##########################################################################
#Training part

def train_model(model, criterion, optimizer, scheduler, dataloaders, batchsize, dataset_sizes, use_gpu = True, num_epochs=25
                , name = 'pcb', gpu_ids = 0):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            calculate_loss = 0.0 # loss
            calculate_corrects = 0.0 # accuracy

            if phase == 'train':
                scheduler.step()
                model.train(True)  #training mode
            else:
                model.train(False)  #evaluate mode

            for data in dataloaders[phase]:
                optimizer.zero_grad()
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < batchsize:  # skip the last batch
                    continue

                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                part = {}
                sm = nn.Softmax(dim=1)
                num_part = 6
                for i in range(num_part):
                    part[i] = outputs[i]

                score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
                _, preds = torch.max(score.data, 1)

                loss = criterion(part[0], labels)
                for i in range(num_part - 1):
                    loss += criterion(part[i + 1], labels)


                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                calculate_loss += loss.item() * now_batch_size
                calculate_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = calculate_loss / dataset_sizes[phase]
            epoch_acc = calculate_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Save model each 10 epoch
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch, name, gpu_ids)

        #running time of epoch
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    #running time of training process
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last', name, gpu_ids)
    return model