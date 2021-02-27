#!_tf) bash-4.2$ vim usr/bin/python3




from dataGenSequences_sru import dataGenSequences
from compute_priors import compute_priors
from shutil import copy
import sys
import os
import torch.utils.data as data
import torch.nn as nn
import torch
import json
from lib.ops import Dense
import numpy
import time
from sru import SRU
#!!! please modify these hyperprameters manually
import warnings
import random
seed = 21
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed) # cpu 
torch.cuda.manual_seed_all(seed) # gpu 
torch.backends.cudnn.deterministic = True
warnings.filterwarnings('ignore')
# this depend on the feature you applied
mfccDim=40


if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 8:
    raise TypeError ('USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir dnn_dir init_lr')


data_cv = sys.argv[1]
ali_cv  = sys.argv[2]
data_tr = sys.argv[3]
ali_tr  = sys.argv[4]
gmm     = sys.argv[5]
exp     = sys.argv[6]
init_lr = float(sys.argv[7])

##!!! please modify these hyperprameters manually
## Learning parameters
learning = {'rate' : init_lr,
            'singFeaDim' : mfccDim, 
            'minEpoch' : 30,
            'batchSize' : 40,#40 at first
            'timeSteps' : 20,
            'dilDepth' : 1,
            'minValError' : 0,
            'left' : 0,
            'right': 4,
            'hiddenDim' : 1280,
            'modelOrder' : 1,
            'layerNum': 12,# 12 at first
            'historyNum' : 1}

## Copy final model and tree from GMM directory
os.makedirs (exp, exist_ok=True)
copy (gmm + '/final.mdl', exp)
copy (gmm + '/tree', exp)



## Compute priors
compute_priors (exp, ali_tr, ali_cv)

# The input feature of the neural network has this form:  0-1-4 features
feaDim = (learning['left'] + learning['right']+1)*mfccDim

# load data from data iterator
trDataset = dataGenSequences (data_tr, ali_tr, gmm,learning['batchSize'],learning['timeSteps'], feaDim,learning['left'],learning['right'])
cvDataset = dataGenSequences (data_cv, ali_cv, gmm,learning['batchSize'],learning['timeSteps'], feaDim,learning['left'],learning['right'])

# Recommend shuffle=False, because this iterator's shuffle can only work on the single split
trGen = data.DataLoader(trDataset,batch_size=learning['batchSize'],shuffle=False,num_workers=0)
cvGen = data.DataLoader(cvDataset,batch_size=learning['batchSize'],shuffle=False,num_workers=0)


##load the configurations from the training data
learning['targetDim'] = trDataset.outputFeatDim

with open(exp + '/learning.json', 'w') as json_file:
    json_file.write(json.dumps(learning))

class lstm(nn.Module):
    def __init__(self,batch_size=learning['batchSize'],input_size = feaDim, hidden_size = 1024 , output_size = 1095, num_layers = learning['layerNum'] ):
        super(lstm,self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Dense_layer1 = Dense(input_size,self.hidden_size)
        self.sru_list = nn.ModuleList()
        #for i in range(self.num_layers):
        #    self.sru_layer = SRU_Formula_Cell(n_in=self.hidden_size, n_out=self.hidden_size, layer_numbers=1, bias=True, dropout = 0.1)
            #self.sru_layer = SRU_Formula_Cell(n_in=self.hidden_size, n_out=self.hidden_size, layer_numbers=1, bias=True)
        #    self.sru_list.append(self.sru_layer)

        self.sru = SRU(input_size=self.hidden_size, hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=0.1,use_tanh=True)
        #self.init_cell4lstm2 = torch.zeros(self.num_layers, batch_size, hidden_size).cuda()
        #self.init_hidden4lstm2 = torch.zeros(self.num_layers, batch_size, hidden_size).cuda()
        self.Dense_layer3 = Dense(self.hidden_size, 1024)
        self.Dense_layer4 = Dense(1024, output_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self,x, hidden):
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer1(x)
        x = self.dropout(x)
        x = torch.reshape(x, (b, t, self.hidden_size))
        hidden_after = torch.zeros_like(hidden)
        x = x.permute(1,0,2)
        x, hidden_after = self.sru(x, hidden)
        x = x.permute(1, 0, 2)
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.Dense_layer4(x)
        return x, hidden_after



# If you run this code on CPU, please remove the '.cuda()'
model = lstm(input_size=feaDim, hidden_size= learning['hiddenDim'],output_size=learning['targetDim']).cuda()
#model.load_state_dict(torch.load(exp + '/dnn.nnet.pth'))




loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD( model.parameters(),lr=learning['rate'],momentum=0.5,nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)


optimizer2 = torch.optim.SGD( model.parameters(),lr=0.01*learning['rate'],momentum=0.5,nesterov=True)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,step_size=1,gamma=0.5)




def train(model, train_loader, my_loss, optimizer, epoch, hidden ):
    model.train()
    acc = 0

    for batch_idx, (x,y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        b, t, h = x.size()
        if b == learning['batchSize']:
            #model.zero_grad()
            optimizer.zero_grad()
            if batch_idx == 0:
                #hidden_before = torch.from_numpy(hidden).cuda()
                hidden_before = hidden.cuda()

            else:
                #hidden_before = torch.from_numpy(hidden_after).cuda()
                hidden_before = hidden_after.cuda()

            output,hidden_after = model(x,hidden_before)
            #time2 = time.time()

            #output= model(x)
            y_batch_size, y_time_steps = y.size()
            y = torch.reshape(y, tuple([y_batch_size * y_time_steps]))
            y = y.long()
            loss = my_loss(output, y)
            loss.backward()
            #time1 = time.time()
            #print(time1 - time2)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            _, pred = torch.max(output.data, 1)


            hidden_after = hidden_after.detach()

            acc += ((pred == y).sum()).cpu().numpy()
            if (batch_idx % 3000 == 0):#3000/600
                print("train:        epoch:%d ,step:%d, loss:%f"%(epoch+1,batch_idx,loss))

    print(acc)
    print(trDataset.numFeats)
    print("Accuracy: %f"%(acc/trDataset.numFeats))

#def val(model, train_loader, my_loss, optimizer, epoch, hidden, cell):
def val(model, train_loader, my_loss, optimizer, epoch, hidden):
#def val(model, train_loader, my_loss, optimizer, epoch):
    model.eval()
    acc = 0
    val_loss = 0
    val_loss_list = []
    # model.hidden = model.sru_layer2.init_hidden(num_layers=1, batch_size=learning['batchSize'])
    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        b, t, h = x.size()
        if b == learning['batchSize']:
            #model.zero_grad()
            optimizer.zero_grad()
            if batch_idx == 0:
                # hidden_before = torch.from_numpy(hidden).cuda()
                hidden_before = hidden.cuda()

            else:
                # hidden_before = torch.from_numpy(hidden_after).cuda()
                hidden_before = hidden_after.cuda()

            #output, hidden_after, cell_after = model(x, hidden_before, cell_before)
            #print(hidden_before.dtype)
            with torch.no_grad():
                output, hidden_after = model(x, hidden_before)
                #output = model(x)

            _, pred = torch.max(output.data, 1)
            y_batch_size, y_time_steps = y.size()
            y = torch.reshape(y, tuple([y_batch_size * y_time_steps]))
            y = y.long()
            loss = my_loss(output, y)
            val_loss += float(loss.item())
            val_loss_list.append(val_loss)
            acc += ((pred == y).sum()).cpu().numpy()
            if (batch_idx % 1000 == 0):#1000/60
                print("val:        epoch:%d ,step:%d, loss:%f" % (epoch + 1, batch_idx, loss))

    print(acc)
    print(cvDataset.numFeats)
    val_acc = acc/cvDataset.numFeats
    print("Accuracy: %f" % (val_acc))
    print("LOSS: %f" % (val_loss / len(val_loss_list)))
    return float(val_loss / len(val_loss_list)),val_acc
'''h = []
for i in range(learning['layerNum']):
    h.append(torch.zeros(1, learning['batchSize'], learning['hiddenDim']).cuda())'''
val_loss_before = 10000
count_epoch = 0

#init_cell4lstm2 = torch.zeros(1, batch_size, hidden_size).cuda()
#init_hidden4lstm2 = torch.zeros(1, batch_size, hidden_size).cuda()

for epoch in range(90):
    print("=====================================================================")

    h = torch.zeros(learning['layerNum'], learning['batchSize'], learning['hiddenDim'])


    time_start = time.time()
    #train(model, trGen, loss_function, optimizer, epoch)
    train(model, trGen, loss_function, optimizer, epoch, h )
    #train(model, trGen, loss_function, optimizer, epoch, h, c )

    print(scheduler.get_lr())

    val_loss_after,_ = val(model, cvGen, loss_function, optimizer, epoch, h)
    #val_loss_after = val(model, cvGen, loss_function, optimizer, epoch)
    #if (float(scheduler.get_lr()[0]) > 0.001):

    if(val_loss_before - val_loss_after < 0) and (count_epoch > 4):
        scheduler.step()
        val_loss_before = 10000
        count_epoch = 0
    else:
        val_loss_before = val_loss_after
        count_epoch += 1

    torch.save(model.state_dict(), exp + '/dnn.nnet.pth')
    time_end = time.time()
    time_cost = time_end - time_start
    print("Time Cost : %f"%(time_cost))
    if (float(scheduler.get_lr()[0]) < 0.001):
        break


print("The second ladder is coming!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
val_acc_before = 0.1
val_acc = 0
for epoch in range(90):
    print("=====================================================================")


    #train(model, trGen, loss_function, optimizer, epoch, h )
    h = torch.zeros(learning['layerNum'],  learning['batchSize'], learning['hiddenDim'])
    time_start = time.time()
    train(model, trGen, loss_function, optimizer2, epoch, h )
    #train(model, trGen, loss_function, optimizer2, epoch)

    print(scheduler2.get_lr())
    val_acc_before = val_acc
    val_loss_after,val_acc = val(model, cvGen, loss_function, optimizer2, epoch, h)
    #val_loss_after = val(model, cvGen, loss_function, optimizer2, epoch)
    #if (float(scheduler.get_lr()[0]) > 0.001):
    scheduler2.step()
    torch.save(model.state_dict(), exp + '/dnn.nnet.pth')
    time_end = time.time()
    time_cost = time_end - time_start
    print("Time Cost : %f"%(time_cost))
    if float(scheduler2.get_lr()[0]) < 1e-6:
        break

