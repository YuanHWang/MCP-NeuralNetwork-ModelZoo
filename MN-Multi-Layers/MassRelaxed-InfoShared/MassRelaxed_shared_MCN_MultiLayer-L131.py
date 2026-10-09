import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch import Tensor
from torch.nn.parameter import Parameter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
from typing import Tuple
import numpy as np
import pandas as pd
import os
import math
import datetime
from tqdm import tqdm
import sys
from sklearn.metrics import mean_squared_error
import csv
import glob
# import class
from MCPBRNN_lib_tools.MCNZoo_MultiLayers import MCN_unit_sharedlossoutput, MCN_SingleMultiMR_SharedLossOutput_IC
from MCPBRNN_lib_tools.headLayers import MR_Linear_Layer
from MCPBRNN_lib_tools.Eval_Metric import ANLL_out, correlation, NS, KGE
from MCPBRNN_lib_tools.Loss_Function import  KGELoss

parser = argparse.ArgumentParser()

parser.add_argument('--case_no',
                        type=int,
                        default=0,
                        help="Corresponds to different random seeds")

parser.add_argument('--epoch_no',
                        type=int,
                        default=3000,
                        help="number of epoch to train the MCP-Based network")

parser.add_argument('--gate_dim',
                        type=int,
                        default=1,
                        help="")

parser.add_argument('--seed_no',
                        type=int,
                        default=2925,
                        help="")

parser.add_argument('--hidden_size',
                        type=int,
                        default=2,
                        help="")

parser.add_argument('--c_mean',
                        type=float,
                        default=412.9139085,
                        help="")

parser.add_argument('--c_std',
                        type=float,
                        default=77.49943867,
                        help="")

parser.add_argument('--depth_size1',
                        type=int,
                        default=1,
                        help="number of hidden nodes")

parser.add_argument('--depth_size2',
                        type=int,
                        default=3,
                        help="number of hidden nodes")

parser.add_argument('--depth_size3',
                        type=int,
                        default=1,
                        help="number of hidden nodes")

cfg = vars(parser.parse_args())

# setup random seed
seed_no = cfg["seed_no"]
np.random.seed(seed_no)
torch.manual_seed(seed_no)

# Define timesteps
spinLen = 1095 
traintimeLen = 8400 
timeLen = 15705 
timeLenR = 15705 - spinLen

# Define Matrix size & Hyperparameters
input_size = 1
hidden_sizeM = 1
hidden_size = cfg["hidden_size"]
hidden_size_layer1 = cfg["depth_size1"]
hidden_size_layer2 = cfg["depth_size2"]
hidden_size_layer3 = cfg["depth_size3"]
gate_dim = cfg["gate_dim"]
num_features = 1
seq_length = 1
input_size_dyn = 1
input_size_stat = 0
num_output = 1
num_epochs = cfg["epoch_no"]
batch_size = timeLen 
learning_rate = 0.025
learning_rates = {300: 0.0125, 600: 0.0125}
hidden_sizeMo = hidden_size - 1
cmean = cfg["c_mean"]
cstd = cfg["c_std"]

# Define case & directory
CaseName = 'MR_DNN_MCN_Layer_' + str(hidden_size_layer1)  + str(hidden_size_layer2) +  str(hidden_size_layer3) + '_case_' + str(cfg["case_no"]) + '_sharedoutputloss'
directory = CaseName
parent_dir = "/home/u9/yhwang0730/PB-LSTM-Papers/20240825-MCN-Deep-Part2/"
#parent_dir = "/Users/yhwang/Desktop/HPC_DownloadTemp/2023-Spring-New/20240825-MCN-MultiLayers"
# Best Case for generic model 
Ini_dir = "20240825-MCN-Deep/Results-InfoShared-MassRelaxed/Layer-1-3-0/MassRelaxed_DNN_MCN_Layer_130_case_4_sharedoutputloss/best_model_epoch463.pt"
path = os.path.join(parent_dir, directory)
isExist = os.path.exists(path)
if not isExist:
  os.mkdir(path)
  print("The new directory is created!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Import Data
F_data = pd.read_csv('../20220527-MDUPLEX-LeafRiver/LeafRiverDaily_43YR.txt', header=None, delimiter=r"\s+")
F_data = F_data.rename(columns={0: 'P', 1: 'PET', 2: 'Q'})

# Skill Flag Training
SkillFlag = pd.read_csv('../20220527-MDUPLEX-LeafRiver/LeafRiverDaily_43YR_Flag.txt', header=None, delimiter=r"\s+")
SkillFlag = SkillFlag.rename(columns={0: 'Flag'})
SF = torch.tensor(SkillFlag['Flag'])
Mask_Test = SF.eq(1).unsqueeze(1)
Mask_Select = SF.eq(0).unsqueeze(1)
Mask_Train = SF.eq(-1).unsqueeze(1)
Mask_Spin = SF.eq(-99999).unsqueeze(1)

Xmean1=0.0
Xmean2=0.0
Xmean3=0.0
Xstd1=1.0
Xstd2=1.0
Xstd3=1.0
#print(F2_data['Qsim'])
F_data['P'][:] = (F_data['P'][:]-Xmean1)/Xstd1
F_data['Q'][:] = (F_data['Q'][:]-Xmean2)/Xstd2
F_data['PET'][:] = (F_data['PET'][:]-Xmean3)/Xstd3

# Define output matrix
par_no = 0 # Temporarily not save the parameter value
skillmetric = 7
OutMatrix = np.zeros((num_epochs,par_no+skillmetric*4))

x_data1 = np.array(F_data['P']).reshape(timeLen,-1)
x_data2 = np.array(F_data['PET']).reshape(timeLen,-1)
x_data = np.concatenate((x_data1, x_data2), axis=1)
y_data = np.array(F_data['Q']).reshape(timeLen,-1)

train_x = x_data[0:timeLen,:]
train_y = y_data[0:timeLen,:]

num_samples, num_features = train_x.shape
x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
y_new = np.zeros((num_samples - seq_length + 1, 1))

for i in range(0, x_new.shape[0]):
    x_new[i, :, :num_features] = train_x[i:i + seq_length, :]
    y_new[i, :] = train_y[i + seq_length - 1, 0]
       
x_new = torch.tensor(x_new)
y_new = torch.tensor(y_new)
x_new = x_new.float()
y_new = y_new.float()
trainx, trainy = Variable(x_new), Variable(y_new)

class Model_old(nn.Module):

    def __init__(self,
                 input_size_dyn: int,
                 hidden_size1: int,
                 hidden_size2: int,
                 hidden_size3: int,                 
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):

        super(Model_old, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.hidden_size1 = hidden_size_layer1 
        self.hidden_size2 = hidden_size_layer2
        self.hidden_size3 = hidden_size_layer3                 
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout       
        self.batch_size = batch_size 

        self.MCPBRNNNode_Layer1 = MCN_unit_sharedlossoutput(input_size=input_size_dyn,
                                                            hidden_size=hidden_size_layer1,  
                                                            initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)

        self.MCPBRNNNode_Layer12_linear = MR_Linear_Layer(input_size=input_size_dyn,
                                                          hidden_size_old=hidden_size_layer1,
                                                          hidden_size_new=hidden_size_layer2)

        self.MCPBRNNNode_Layer2 = MCN_unit_sharedlossoutput(input_size=input_size_dyn,
                                                            hidden_size=hidden_size_layer2,
                                                            initial_forget_bias=initial_forget_bias)


        self.MCPBRNNNode_Layer2out_linear = MR_Linear_Layer(input_size=input_size_dyn,
                                                          hidden_size_old=hidden_size_layer2,
                                                          hidden_size_new=1)

    def forward(self, x_d, cmean, cstd):

        h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, ol_constraint, f = self.MCPBRNNNode_Layer1(x_d, cmean, cstd)   

        h_t_layer1_out = self.MCPBRNNNode_Layer12_linear(h_t)
        x_layer2 = torch.cat((h_t_layer1_out, x_d[:,:,1]), 1)
        x_d_layer2 = x_layer2.unsqueeze(1)

        h_t_layer2, c_t_layer2, l_t_layer2, lc_t_layer2, bp_t_layer2, i_layer2, oo_layer2, ol_layer2, ol_constraint_layer2, f_layer2 = self.MCPBRNNNode_Layer2(x_d_layer2, cmean, cstd)

        h_t_layer2_out = self.MCPBRNNNode_Layer2out_linear(h_t_layer2)
        last_h = self.dropout(h_t_layer2_out)          
        out = last_h        

        return out, h_t, h_t_layer2, c_t, c_t_layer2, l_t, l_t_layer2, lc_t, lc_t_layer2, bp_t, bp_t_layer2, i, i_layer2, oo, oo_layer2, ol, ol_layer2, ol_constraint, ol_constraint_layer2, f, f_layer2, h_t_layer1_out, h_t_layer2_out

model_old = Model_old(input_size_dyn=input_size_dyn,
                      hidden_size1=hidden_size_layer1,
                      hidden_size2=hidden_size_layer2,    
                      hidden_size3=hidden_size_layer3,                                          
                      initial_forget_bias=0,
                      dropout=0).to(device)

class Model(nn.Module):

    def __init__(self,
                 input_size_dyn: int,
                 hidden_size1: int,
                 hidden_size2: int,
                 hidden_size3: int,                 
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):

        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.hidden_size1 = hidden_size_layer1 
        self.hidden_size2 = hidden_size_layer2
        self.hidden_size3 = hidden_size_layer3                 
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout       
        self.batch_size = batch_size 

        self.MCPBRNNNode_Layer1 = MCN_unit_sharedlossoutput(input_size=input_size_dyn,
                                                            hidden_size=hidden_size_layer1,  
                                                            initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)

        self.MCPBRNNNode_Layer12_linear = MR_Linear_Layer(input_size=input_size_dyn,
                                                          hidden_size_old=hidden_size_layer1,
                                                          hidden_size_new=hidden_size_layer2)

        self.MCPBRNNNode_Layer2 = MCN_unit_sharedlossoutput(input_size=input_size_dyn,
                                                            hidden_size=hidden_size_layer2,
                                                            initial_forget_bias=initial_forget_bias)


        self.MCPBRNNNode_Layer23_linear = MR_Linear_Layer(input_size=input_size_dyn,
                                                          hidden_size_old=hidden_size_layer2,
                                                          hidden_size_new=hidden_size_layer3)

        self.MCPBRNNNode_Layer3 = MCN_unit_sharedlossoutput(input_size=input_size_dyn,
                                                            hidden_size=hidden_size_layer3,
                                                            initial_forget_bias=initial_forget_bias)

        self.MCPBRNNNode_Layer3out_linear = MR_Linear_Layer(input_size=input_size_dyn,
                                                           hidden_size_old=hidden_size_layer3,
                                                           hidden_size_new=1)

    def forward(self, x_d, cmean, cstd):

        h_t, c_t, l_t, lc_t, bp_t, i, oo, ol, ol_constraint, f = self.MCPBRNNNode_Layer1(x_d, cmean, cstd)

        h_t_layer1_out = self.MCPBRNNNode_Layer12_linear(h_t)
        x_layer2 = torch.cat((h_t_layer1_out, x_d[:,:,1]), 1)
        x_d_layer2 = x_layer2.unsqueeze(1)

        h_t_layer2, c_t_layer2, l_t_layer2, lc_t_layer2, bp_t_layer2, i_layer2, oo_layer2, ol_layer2, ol_constraint_layer2, f_layer2 = self.MCPBRNNNode_Layer2(x_d_layer2, cmean, cstd)
        h_t_layer2_out = self.MCPBRNNNode_Layer23_linear(h_t_layer2)

        x_layer3 = torch.cat((h_t_layer2_out, x_d[:,:,1]), 1)
        x_d_layer3 = x_layer3.unsqueeze(1)

        h_t_layer3, c_t_layer3, l_t_layer3, lc_t_layer3, bp_t_layer3, i_layer3, oo_layer3, ol_layer3, ol_constraint_layer3, f_layer3  = self.MCPBRNNNode_Layer3(x_d_layer3, cmean, cstd)
        h_t_layer3_out = self.MCPBRNNNode_Layer3out_linear(h_t_layer3)
        last_h = self.dropout(h_t_layer3_out)
        out = last_h

        return out, h_t, h_t_layer2, h_t_layer3, c_t, c_t_layer2, c_t_layer3, l_t, l_t_layer2, l_t_layer3, lc_t, lc_t_layer2, lc_t_layer3, bp_t, bp_t_layer2, bp_t_layer3, i, i_layer2, i_layer3, oo, oo_layer2, oo_layer3, ol, ol_layer2, ol_layer3, ol_constraint, ol_constraint_layer2, ol_constraint_layer3, f, f_layer2, f_layer3, h_t_layer1_out, h_t_layer2_out, h_t_layer3_out

model = Model(input_size_dyn=input_size_dyn,
                hidden_size1=hidden_size_layer1,
                hidden_size2=hidden_size_layer2,    
                hidden_size3=hidden_size_layer3,                                          
                initial_forget_bias=0,
                dropout=0).to(device)

for name, param in model.state_dict().items():
   if model.state_dict()[name][:].dim() == 1:
      YHW=np.random.normal(0,1,size=(model.state_dict()[name][:].shape[0]))
      model.state_dict()[name][:]=torch.Tensor(YHW)
   elif model.state_dict()[name][:].dim() == 2:
      YHW=np.random.normal(0,1,size=(model.state_dict()[name][:].shape[0],model.state_dict()[name][:].shape[1]))
      model.state_dict()[name][:]=torch.Tensor(YHW)
   elif model.state_dict()[name][:].dim() == 3:
      YHW=np.random.normal(0,1,size=(model.state_dict()[name][:].shape[0],model.state_dict()[name][:].shape[1],model.state_dict()[name][:].shape[2]))
      model.state_dict()[name][:]=torch.Tensor(YHW)

weight_IC_file = Ini_dir
model_old.load_state_dict(torch.load(weight_IC_file))

#Layer1
model.state_dict()["MCPBRNNNode_Layer1.weight_r_yom"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.weight_r_yom"][:]
model.state_dict()["MCPBRNNNode_Layer1.weight_r_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.weight_r_ylm"][:]
model.state_dict()["MCPBRNNNode_Layer1.weight_r_yfm"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.weight_r_yfm"][:]
model.state_dict()["MCPBRNNNode_Layer1.bias_b0_yom"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.bias_b0_yom"][:]
model.state_dict()["MCPBRNNNode_Layer1.bias_b0_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.bias_b0_ylm"][:]
model.state_dict()["MCPBRNNNode_Layer1.weight_b1_yom"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.weight_b1_yom"][:]
model.state_dict()["MCPBRNNNode_Layer1.weight_b1_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.weight_b1_ylm"][:]
model.state_dict()["MCPBRNNNode_Layer1.weight_b2_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer1.weight_b2_ylm"][:]

# Layer-12-LinearTransformation
model.state_dict()["MCPBRNNNode_Layer12_linear.weight_w"][:] = model_old.state_dict()["MCPBRNNNode_Layer12_linear.weight_w"][:]

# Layer 2
model.state_dict()["MCPBRNNNode_Layer2.weight_r_yom"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.weight_r_yom"][:]
model.state_dict()["MCPBRNNNode_Layer2.weight_r_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.weight_r_ylm"][:]
model.state_dict()["MCPBRNNNode_Layer2.weight_r_yfm"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.weight_r_yfm"][:]
model.state_dict()["MCPBRNNNode_Layer2.bias_b0_yom"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.bias_b0_yom"][:]
model.state_dict()["MCPBRNNNode_Layer2.bias_b0_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.bias_b0_ylm"][:]
model.state_dict()["MCPBRNNNode_Layer2.weight_b1_yom"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.weight_b1_yom"][:]
model.state_dict()["MCPBRNNNode_Layer2.weight_b1_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.weight_b1_ylm"][:]
model.state_dict()["MCPBRNNNode_Layer2.weight_b2_ylm"][:] = model_old.state_dict()["MCPBRNNNode_Layer2.weight_b2_ylm"][:]

# Layer-23-LinearTransformation
model.state_dict()["MCPBRNNNode_Layer23_linear.weight_w"][:] = model_old.state_dict()["MCPBRNNNode_Layer2out_linear.weight_w"][:]

loss_func = KGELoss()

train_dataset = Data.TensorDataset(trainx, trainy)
loader = Data.DataLoader(
         dataset=train_dataset, 
         batch_size=batch_size, 
         shuffle=False, num_workers=0)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

for name, param in model.state_dict().items():
    print(name)
    print(param)

savetext = CaseName + '/' +'model_epoch0.pt'
torch.save(model.state_dict(), savetext)

model.train()
for epoch in range(1,num_epochs+1):

    if epoch in learning_rates.keys():
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rates[epoch]

    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')
    savetext = CaseName + '/' +'model_epoch' + str(epoch) + '.pt'
    pbar.set_description(f'# Epoch {epoch}')

    for data in pbar:
        optimizer.zero_grad()
        x, y,= data
        predictions = model(x, cmean, cstd)[0]
        sim = torch.masked_select(predictions, Mask_Train).unsqueeze(1)
        obs =  torch.masked_select(y, Mask_Train).unsqueeze(1)
        loss = loss_func(sim, obs)
        loss.backward()
        optimizer.step()
        sim = torch.masked_select(predictions, Mask_Train).unsqueeze(1)
        obs =  torch.masked_select(y, Mask_Train).unsqueeze(1)
        sim_select = torch.masked_select(predictions, Mask_Select).unsqueeze(1)
        obs_select =  torch.masked_select(y, Mask_Select).unsqueeze(1)
        sim_test = torch.masked_select(predictions, Mask_Test).unsqueeze(1)
        obs_test =  torch.masked_select(y, Mask_Test).unsqueeze(1)
        sim_spinup = torch.masked_select(predictions, Mask_Spin).unsqueeze(1)
        obs_spinup =  torch.masked_select(y, Mask_Spin).unsqueeze(1)

    #calculate the skill
    sim = sim.detach().cpu().numpy()
    obs = obs.detach().cpu().numpy()
    sim_spinup = sim_spinup.detach().cpu().numpy()
    obs_spinup = obs_spinup.detach().cpu().numpy()
    sim_select = sim_select.detach().cpu().numpy()
    obs_select = obs_select.detach().cpu().numpy()
    sim_test = sim_test.detach().cpu().numpy()
    obs_test = obs_test.detach().cpu().numpy()

    [KGE_train, CC_train, KGE_A_train, KGE_B_train, KGEss_train] =KGE(sim, obs)
    NS_train = NS(sim, obs)
    MSE_train = mean_squared_error(obs, sim)

    [KGE_select, CC_select, KGE_A_select, KGE_B_select, KGEss_select] =KGE(sim_select, obs_select)
    NS_select = NS(sim_select, obs_select)
    MSE_select = mean_squared_error(obs_select, sim_select)

    [KGE_test, CC_test, KGE_A_test, KGE_B_test, KGEss_test] =KGE(sim_test, obs_test)
    NS_test = NS(sim_test, obs_test)
    MSE_test = mean_squared_error(obs_test, sim_test)

    [KGE_spinup, CC_spinup, KGE_A_spinup, KGE_B_spinup, KGEss_spinup] =KGE(sim_spinup, obs_spinup)
    NS_spinup = NS(sim_spinup, obs_spinup)
    MSE_spinup = mean_squared_error(obs_spinup, sim_spinup)

    OutMatrix[epoch-1,par_no] = NS_train
    OutMatrix[epoch-1,par_no+7] = NS_select
    OutMatrix[epoch-1,par_no+14] = NS_test
    OutMatrix[epoch-1,par_no+21] = NS_spinup

    OutMatrix[epoch-1,par_no+1] = KGE_train
    OutMatrix[epoch-1,par_no+8] = KGE_select
    OutMatrix[epoch-1,par_no+15] = KGE_test
    OutMatrix[epoch-1,par_no+22] = KGE_spinup

    OutMatrix[epoch-1,par_no+2] = KGE_A_train
    OutMatrix[epoch-1,par_no+3] = KGE_B_train
    OutMatrix[epoch-1,par_no+9] = KGE_A_select
    OutMatrix[epoch-1,par_no+10] = KGE_B_select
    OutMatrix[epoch-1,par_no+16] = KGE_A_test
    OutMatrix[epoch-1,par_no+17] = KGE_B_test
    OutMatrix[epoch-1,par_no+23] = KGE_A_spinup
    OutMatrix[epoch-1,par_no+24] = KGE_B_spinup

    OutMatrix[epoch-1,par_no+4] = CC_train
    OutMatrix[epoch-1,par_no+11] = CC_select
    OutMatrix[epoch-1,par_no+18] = CC_test
    OutMatrix[epoch-1,par_no+25] = CC_spinup

    OutMatrix[epoch-1,par_no+5] = MSE_train
    OutMatrix[epoch-1,par_no+12] = MSE_select
    OutMatrix[epoch-1,par_no+19] = MSE_test
    OutMatrix[epoch-1,par_no+26] = MSE_spinup

    OutMatrix[epoch-1,par_no+6] = KGEss_train
    OutMatrix[epoch-1,par_no+13] = KGEss_select
    OutMatrix[epoch-1,par_no+20] = KGEss_test
    OutMatrix[epoch-1,par_no+27] = KGEss_spinup

    print(KGE_select)
    torch.save(model.state_dict(), savetext)

# Save Results file
outtofile = pd.DataFrame(OutMatrix, columns=['NSE','KGE','KGE-A','KGE-B','Corr','mse','KGEss','NSE_selection','KGE_selection','KGE-A_selection','KGE-B_selection','Corr_selection','mse_selection','KGEss_selection','NSE_testing','KGE_testing','KGE-A_testing','KGE-B_testing','Corr_testing','mse_testing','KGEss_testing','NSE_spinup','KGE_spinup','KGE-A_spinup','KGE-B_spinup','Corr_spinup','mse_spinup','KGEss_spinup'])
outname = CaseName + '/' + 'IC_caseno_' + str(cfg["case_no"]) + '_summary.csv'
outtofile.to_csv(outname)

# output flowseries
indx = np.argmax(OutMatrix[:,8+par_no])
mpt = CaseName + '/model_epoch' + str(indx) + '.pt'
model.load_state_dict(torch.load(mpt))

for data_full in pbar:
    x_full, y_full,= data_full
    Result = model(x_full, cmean, cstd)
    out_eval = Result[0]
    hidden_eval = Result[1]
    hidden_layer2_eval = Result[2]
    hidden_layer3_eval = Result[3]
    cell_eval = Result[4]
    cell_layer2_eval = Result[5]
    cell_layer3_eval = Result[6]
    loss_eval = Result[7]
    loss_layer2_eval = Result[8]
    loss_layer3_eval = Result[9]
    lossc_eval = Result[10]
    lossc_layer2_eval = Result[11]
    lossc_layer3_eval = Result[12]
    o_gate = Result[19]
    o_gate_layer2 = Result[20]
    o_gate_layer3 = Result[21]
    l_gate = Result[22]
    l_gate_layer2 = Result[23]
    l_gate_layer3 = Result[24]
    lc_gate = Result[25]
    lc_gate_layer2 = Result[26]
    lc_gate_layer3 = Result[27]
    f_gate = Result[28]
    f_gate_layer2 = Result[29]
    f_gate_layer3 = Result[30]
    h_t_layer1_out = Result[31]
    h_t_layer2_out = Result[32]
    h_t_layer3_out = Result[33]

out_eval = out_eval.detach().cpu().numpy()
hidden_eval = hidden_eval.detach().cpu().numpy()
hidden_layer2_eval = hidden_layer2_eval.detach().cpu().numpy()
hidden_layer3_eval = hidden_layer3_eval.detach().cpu().numpy()
cell_eval = cell_eval.detach().cpu().numpy()
cell_layer2_eval = cell_layer2_eval.detach().cpu().numpy()
cell_layer3_eval = cell_layer3_eval.detach().cpu().numpy()
loss_eval = loss_eval.detach().cpu().numpy()
loss_layer2_eval = loss_layer2_eval.detach().cpu().numpy()
loss_layer3_eval = loss_layer3_eval.detach().cpu().numpy()
lossc_eval = lossc_eval.detach().cpu().numpy()
lossc_layer2_eval = lossc_layer2_eval.detach().cpu().numpy()
lossc_layer3_eval = lossc_layer3_eval.detach().cpu().numpy()
o_gate = o_gate.detach().cpu().numpy()
o_gate_layer2 = o_gate_layer2.detach().cpu().numpy()
o_gate_layer3 = o_gate_layer3.detach().cpu().numpy()
l_gate = l_gate.detach().cpu().numpy()
l_gate_layer2 = l_gate_layer2.detach().cpu().numpy()
l_gate_layer3 = l_gate_layer3.detach().cpu().numpy()
lc_gate = lc_gate.detach().cpu().numpy()
lc_gate_layer2 = lc_gate_layer2.detach().cpu().numpy()
lc_gate_layer3 = lc_gate_layer3.detach().cpu().numpy()
f_gate = f_gate.detach().cpu().numpy()
f_gate_layer2 = f_gate_layer2.detach().cpu().numpy()
f_gate_layer3 = f_gate_layer3.detach().cpu().numpy()
h_t_layer1_out = h_t_layer1_out.detach().cpu().numpy()
h_t_layer2_out = h_t_layer2_out.detach().cpu().numpy()
h_t_layer3_out = h_t_layer3_out.detach().cpu().numpy()

# Define output index
Out_out = np.zeros((timeLen,1))
Out_hidden = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))
Out_hidden_out = np.zeros((timeLen, hidden_size_layer2 + hidden_size_layer3 + 1))
Out_cell = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))
Out_loss = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))
Out_lossc = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))
Out_o_gate = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))
Out_l_gate = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))
Out_lc_gate = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))
Out_f_gate = np.zeros((timeLen,hidden_size_layer1 + hidden_size_layer2 + hidden_size_layer3))

Out_out = out_eval
Out_hidden = np.concatenate((hidden_eval, hidden_layer2_eval, hidden_layer3_eval), axis=1)
Out_hidden_out = np.concatenate((h_t_layer1_out, h_t_layer2_out, h_t_layer3_out), axis=1)
Out_cell = np.concatenate((cell_eval, cell_layer2_eval, cell_layer3_eval), axis=1)
Out_loss = np.concatenate((loss_eval, loss_layer2_eval, loss_layer3_eval), axis=1)
Out_lossc = np.concatenate((lossc_eval, lossc_layer2_eval, lossc_layer3_eval), axis=1)
Out_o_gate = np.concatenate((o_gate, o_gate_layer2, o_gate_layer3), axis=1)
Out_l_gate = np.concatenate((l_gate, l_gate_layer2, l_gate_layer3), axis=1)
Out_lc_gate = np.concatenate((lc_gate, lc_gate_layer2, lc_gate_layer3), axis=1)
Out_f_gate  = np.concatenate((f_gate, f_gate_layer2, f_gate_layer3), axis=1)

# Save timeseries
Out_file = pd.DataFrame(Out_out)
outname = CaseName + '/' + 'Outhidden_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_hidden)
outname = CaseName + '/' + 'Outhidden_ALL_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_hidden_out)
outname = CaseName + '/' + 'Outhidden_out_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_cell)
outname = CaseName + '/' + 'Outcell_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_loss)
outname = CaseName + '/' + 'Outloss_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_lossc)
outname = CaseName + '/' + 'Outlossc_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_o_gate)
outname = CaseName + '/' + 'Out_gateo_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_l_gate)
outname = CaseName + '/' + 'Out_gatel_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_lc_gate)
outname = CaseName + '/' + 'Out_gatelc_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

Out_file = pd.DataFrame(Out_f_gate)
outname = CaseName + '/' + 'Out_gatef_' + str(cfg["case_no"]) + '_summary.csv'
Out_file.to_csv(outname)

outtofile = pd.DataFrame(np.array([indx]), columns=['Best_Epoch'])
outname = CaseName + '/' + 'Best_Epoch_Caseno' + str(cfg["case_no"]) + '_summary.csv'
outtofile.to_csv(outname)

old_file = CaseName + '/model_epoch' + str(indx) + '.pt'
new_file = CaseName + '/best_model_epoch' + str(indx) + '.pt'
os.rename(old_file, new_file)

outtext = CaseName + '/model_epoch*.pt'
files = glob.glob(outtext, recursive=True)

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
