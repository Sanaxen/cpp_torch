import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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


    def forward(self, x, batch_size=1):

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
device = 'cpu'
model = Net(device).to(device)
model.load_state_dict(torch.load('model.pt'))


images_name = '.\pytorch/test/train_images_ts.csv'
labels_name = '.\pytorch/test/train_labels_ts.csv'
ary_img = pd.read_csv(images_name, delimiter=',' , header=None).to_numpy()
ary_lbl = pd.read_csv(labels_name, delimiter=',' , header=None).to_numpy()
I_MIN = -2.051906
I_MAX = 1.713744
datas = []
targets = []

for i in range(len(ary_img)):
    datas.append(ary_img[i])
    targets.append(ary_lbl[i])

datas = np.array(datas, dtype='float32')
targets = np.array(targets, dtype='float32')

test_X = torch.from_numpy(datas)
test_Y = torch.from_numpy(targets)

test = torch.utils.data.TensorDataset(test_X, test_Y)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)


MIN_MEAN = [0.2052302509546280,]
MAX_VAR = [0.6977003216743469,]
out_sequence_length = 1
target_position = 1
sequence_length = 32
prophecy = 0
x_dim = 2
y_dim = 1
pred0 = []
t0 = []
use_observed_value = 1.000000
model.eval()


sample_x = []
xxx = 0
with torch.no_grad():
    for ith, data in enumerate(test_loader):
        if ith% out_sequence_length != 0:
            continue

        x, t = data

        x = x.to(device)
        t = t.to(device)
        if use_observed_value * len(test_loader) < ith:
            x_add= torch.zeros(out_sequence_length * (x_dim + y_dim)).to(device)
            for o_seq_cnt in range(out_sequence_length):
                for d_cnt in range(y_dim):
                    x_add[(x_dim + y_dim) * o_seq_cnt + d_cnt] = output[0][y_dim * o_seq_cnt +d_cnt]
                for d_cnt in range(x_dim):
                    x_add[(x_dim + y_dim) * o_seq_cnt + y_dim +d_cnt] = x[0][-(x_dim + y_dim) * (out_sequence_length - o_seq_cnt) + y_dim + d_cnt]

            x = torch.cat((x_tmp[0][(x_dim + y_dim) * out_sequence_length:], x_add)  ,0).reshape(1, -1)

        x_tmp = x
        if xxx==0:
            sample_x = x
            xxx=1

        output = model(x, x.shape[0])

        for jth in range(len(output[0])//y_dim):
            pred0.append(MAX_VAR[0] * output[0][jth * y_dim + 0] + MIN_MEAN[0])
            t0.append(MAX_VAR[0] * t[0][jth * y_dim + 0] + MIN_MEAN[0])
    output = model(x, x.shape[0])
    x = torch.cat(( x[0][x_dim + y_dim:], torch.zeros(x_dim + y_dim).to(device) ),0).reshape(1, -1)

    for jth in range(y_dim):
        x[0][-(y_dim - jth)] = output[0][jth]

    for ith in range(prophecy):
        output = model(x, x.shape[0])

        x = torch.cat(( x[0][x_dim + y_dim:], torch.zeros(x_dim + y_dim).to(device) ),0).reshape(1, -1)
        for jth in range(y_dim):
            x[0][-(y_dim - jth)] = output[0][jth]

        pred0.append(MAX_VAR[0] * output[0][-y_dim + 0] + MIN_MEAN[0])


print(sample_x.shape)
script_module = torch.jit.trace(model, sample_x)
script_module.save('script_module.pt')

fig = plt.figure()
plt.plot(t0,  label = 'original')
plt.plot(pred0,  label = 'predict')
plt.legend(loc = 'upper right')
plt.show()
fig.savefig('result_0.png')

loss_0 = 0
for ith in range(len(t0)):
    loss_0 += abs(pred0[ith] - t0[ith]) ** 2
print('Y0', loss_0.item()/len(pred0))


images_name_tr = '.\pytorch/train/train_images_tr.csv'
labels_name_tr = '.\pytorch/train/train_labels_tr.csv'
ary_trn_img = pd.read_csv(images_name_tr, delimiter=',', header=None).to_numpy()
ary_trn_lbl = pd.read_csv(labels_name_tr, delimiter=',', header=None).to_numpy()

train = [[],]

for kth in range(sequence_length + target_position - 1):
    train[0].append(MAX_VAR[0] * ary_trn_img[kth][0] + MIN_MEAN[0])
for kth in range(len(ary_trn_lbl)):
    train[0].append(MAX_VAR[0] * ary_trn_lbl[kth][0] + MIN_MEAN[0])


fig = plt.figure(figsize=(12.0, 4.0))
plt.plot(range(0, len(train[0]+ t0)), train[0] + t0, label = 'Y0_origin')
plt.plot(range(len(train[0]), len(train[0])+ len(pred0)), pred0,  label = 'predict')
plt.legend(loc = 'upper right')
plt.show()
fig.savefig('result0_all.png')


