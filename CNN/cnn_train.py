import random
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# DataLoader
class Loader(data.Dataset):
    def __init__(self, partition='train'):
        super(Loader, self).__init__()
        #  load data
        #    train_data  (1024,2387)
        #    train_label (1,2387)
        #    test_data   (1024,1023)
        #    test_label  (1,1023)
        self.partition = partition
        mat_contents = sio.loadmat('../facedata.mat')
        train_data = mat_contents['train_data']
        test_data = mat_contents['test_data']
        train_label = mat_contents['train_label']
        test_label = mat_contents['test_label']
        self.train_data = torch.unsqueeze(torch.from_numpy(np.transpose(train_data.T.reshape(-1,32,32),(0,2,1))),dim=1).float()
        self.test_data = torch.unsqueeze(torch.from_numpy(np.transpose(test_data.T.reshape(-1,32,32),(0,2,1))),dim=1).float()
        self.train_label = torch.squeeze(torch.from_numpy(train_label.T)).long()-1
        self.test_label = torch.squeeze(torch.from_numpy(test_label.T)).long()-1

    def getbatch(self,batch_size,seed):
        random.seed(seed)
        if self.partition == 'train':
            data = self.train_data
            label = self.train_label
        else:
            data = self.test_data
            label = self.test_label
        len_ls = list(range(data.size(0)))
        idxs = random.sample(len_ls,batch_size)
        batch_data = data[idxs,:,:,:]
        batch_label = label[idxs]

        return batch_data,batch_label

# Model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=5)
        self.fc1 = nn.Linear(in_features=1250,out_features=500)
        self.out = nn.Linear(in_features=500,out_features=21)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.reshape(-1,50*5*5)
        x = F.relu(self.fc1(x))
        x = self.out(x)

        return x

# Trainer
class ModelTrainer(object):
    def __init__(self,
                 model,
                 data_loader,
                 lr=1e-3,
                 weight_decay=1e-6,
                 iteration=10000):
        # learning parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.iteration = iteration
        # set cnn model
        self.model = model

        # get data loader
        self.data_loader = data_loader

        # set optimizer
        self.module_params = self.model.parameters()
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)

        # set loss
        self.loss = nn.CrossEntropyLoss()
    
    def train(self):
        running_loss = 0.0
        best_loss = 100.0
        for i in range(self.iteration):
            # get inputs
            inputs,labels = self.data_loader.getbatch(batch_size=10,seed=i)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss(outputs,labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  #print every 200 mini-batches
                print('step:%d  loss:%f' %
                        (i+1,running_loss / 200))
                if best_loss > running_loss:
                    best_loss = running_loss
                    torch.save(self.model.state_dict(), './trained_cnn_model.pth')
                running_loss = 0.0

        print('Finished Training')

    def test(self):

        inputs,labels = self.data_loader.getbatch(batch_size=1023,seed=0)

        outputs = self.model(inputs)
        pred = torch.argmax(outputs,dim=1)
        acc = sum((pred==labels).float())/1023
        print('acc: %.2f%%' % (acc*100))

if __name__ == '__main__':

    m_lr = 1e-3
    m_weight_decay = 1e-6
    m_train_iteration = 20000

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = Loader(partition='train')
    
    cnn_model = Net()

    # create trainer
    trainer = ModelTrainer(model=cnn_model,
                           data_loader=train_loader,
                           lr=m_lr,
                           weight_decay=m_weight_decay,
                           iteration=m_train_iteration)

    trainer.train()
