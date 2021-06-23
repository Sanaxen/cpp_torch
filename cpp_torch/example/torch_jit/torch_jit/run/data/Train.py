import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms


images_name = '.\pytorch/train/train_images_tr.csv'
labels_name = '.\pytorch/train/train_labels_tr.csv'
ary_img = pd.read_csv(images_name, delimiter=',', header=None).to_numpy()
ary_lbl = pd.read_csv(labels_name, delimiter=',', header=None).to_numpy()
I_MIN = -2.051906
I_MAX = 1.713744
datas = []
targets = []
for i in range(len(ary_img)):
    datas.append(ary_img[i])
    targets.append(ary_lbl[i])

N_TRAIN_EPOCHS = 1000
N_HIDDEN_SIZE = 32
N_MINIBATCH = 32

datas = np.array(datas, dtype='float32')
targets = np.array(targets, dtype='float32')

train_X = torch.from_numpy(datas)
train_Y = torch.from_numpy(targets)

train = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(train, batch_size=N_MINIBATCH, shuffle=True)
train_loader_check = torch.utils.data.DataLoader(train, batch_size=N_MINIBATCH, shuffle=False)


#input 3
#1,96,1
#output 3
#1,96,1
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, ngpu):
        super(Net, self).__init__()
        self.ngpu = ngpu
        self.fc0 = nn.Linear(96, 96, bias = False)
        self.fc1 = nn.Linear(96, 96, bias = False)
        self.lstm0 = nn.LSTM(3,32,num_layers=1,dropout=0.000000, batch_first=True)
        self.fc2 = nn.Linear(32, 5, bias = True)
        self.fc3 = nn.Linear(5, 5, bias = True)
        self.fc4 = nn.Linear(5, 5, bias = True)
        self.fc5 = nn.Linear(5, 5, bias = True)
        self.fc6 = nn.Linear(5, 5, bias = True)
        self.fc7 = nn.Linear(5, 5, bias = True)
        self.fc8 = nn.Linear(5, 1, bias = True)


    def forward(self, x, batch_size):

        y = x.view(batch_size, -1)
        x = self.fc0(y)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc1(y)
        x = F.tanh(x)

        x = x.view(-1, 32, 3)
        x,(hn,cn) = self.lstm0(x)
        x = hn
        x= x.contiguous().view(batch_size, -1)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc2(y)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc3(y)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc4(y)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc5(y)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc6(y)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc7(y)
        x = F.tanh(x)

        y = x.view(batch_size, -1)
        x = self.fc8(y)
        return x




device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(device).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000100)

fc_cnt = 0
while(True):
    if(hasattr(model, 'fc'+str(fc_cnt)) == False ):
        break
    exec('nn.init.xavier_normal_(model.{}.weight)'.format('fc'+str(fc_cnt)))
    fc_cnt += 1


best_epochs = 0
best_loss = -1
log_epoch = 50
loss_graph = []
acc_graph = []
try:
    for epoch in range(N_TRAIN_EPOCHS):
        total_loss = 0
        cnt_loss = 0
        correct_num = 0
        model.train()
        for ith, data in enumerate(train_loader):
            optimizer.zero_grad()
            x, t = data
            x = x.to(device)
            t = t.to(device)

            output = model(x, x.shape[0])

            loss = criterion(output, t)
            total_loss += loss.data
            loss.backward()

            optimizer.step()
            cnt_loss += 1
        print('Train Epoch: {}/{} 	Loss: {:.6f}'.format
              (epoch, N_TRAIN_EPOCHS, total_loss/cnt_loss))
        loss_graph.append(total_loss/cnt_loss)

        if total_loss/cnt_loss < 0.000001:
            print('tolerance stop')
            break

        if (total_loss/cnt_loss < best_loss)or(best_loss < 0):
            best_loss = total_loss/cnt_loss
            best_epochs = epoch

        if (N_TRAIN_EPOCHS * 0.1) < (epoch - best_epochs):
            print('early stop')
            break
        if epoch % log_epoch == 0:
            fig = plt.figure(figsize=(8.0, 4.0))
            plt.plot(loss_graph)
            fig.savefig('loss.png')

            pred0 = []
            true0 = []
            model.eval()
            with torch.no_grad():
                for hth, data_check in enumerate(train_loader_check):
                    x_ck, t_ck = data_check
                    x_ck = x_ck.to(device)
                    t_ck = t_ck.to(device)
                    output_ck = model(x_ck, x_ck.shape[0])

                    for gth in range(len(output_ck)):
                        pred0.append(output_ck[gth][0])
                        true0.append(t_ck[gth][0])
                fig = plt.figure(figsize=(8.0, 4.0))
                plt.plot(true0, label = 'original')
                plt.plot(pred0, label = 'predict')
                plt.legend(loc = 'upper right')
                fig.savefig('fitting_0.png')


    torch.save(model.state_dict(), 'model.pt')
except KeyboardInterrupt:
    print('KeyboardInterrupt')
    torch.save(model.state_dict(), 'model.pt')


