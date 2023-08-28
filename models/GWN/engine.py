import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

class trainer_mean_variance():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet_mean_variance(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss_gauss = torch.nn.GaussianNLLLoss()
        self.loss_l2 = torch.nn.MSELoss()
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output_mean,output_var = self.model(input)
        output_mean,output_var = output_mean.transpose(1,3),output_var.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict_mean = self.scaler.inverse_transform(output_mean) # Inverse transform

        loss_gauss = self.loss_gauss(real, predict_mean, output_var)
        loss_l2 = self.loss_l2(predict_mean,real)
        loss = loss_gauss + loss_l2
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict_mean,real,0.0).item()
        rmse = util.masked_rmse(predict_mean,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        # Ouput size: (batch_size,seq_length,num_nodes,1)
        output_mean,output_var = self.model(input)
        output_mean,output_var = output_mean.transpose(1,3),output_var.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict_mean = self.scaler.inverse_transform(output_mean) # Inverse transform
        
        loss_gauss = self.loss_gauss(real, predict_mean, output_var)
        loss_l2 = self.loss_l2(predict_mean,real)
        loss = loss_gauss + loss_l2
        
        mape = util.masked_mape(predict_mean,real,0.0).item()
        rmse = util.masked_rmse(predict_mean,real,0.0).item()
        return loss.item(),mape,rmse
    

class trainer_nb():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet_nb(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.NegativeBinomialLoss()
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output_mean,output_theta = self.model(input)
        output_mean,output_theta = output_mean.transpose(1,3),output_theta.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict_mean = output_mean.clone() # Origin: inverse transform of scaler
        
        loss = self.loss(output_mean,output_theta,real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict_mean,real,0.0).item()
        rmse = util.masked_rmse(predict_mean,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        # Ouput size: (batch_size,seq_length,num_nodes,1)
        output_mean,output_theta = self.model(input)
        output_mean,output_theta = output_mean.transpose(1,3),output_theta.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict_mean = output_mean.clone() # Inverse transform
        
        loss = self.loss(output_mean,output_theta,real)
        
        mape = util.masked_mape(predict_mean,real,0.0).item()
        rmse = util.masked_rmse(predict_mean,real,0.0).item()
        return loss.item(),mape,rmse

class trainer_calib():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet_calib(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = GaussianCalibrationLoss()
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output_mean,output_var, alpha, gamma = self.model(input)
        output_mean,output_var = output_mean.transpose(1,3),output_var.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict_mean = self.scaler.inverse_transform(output_mean) # Inverse transform

        loss = self.loss(real, predict_mean, output_var, alpha, gamma)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict_mean,real,0.0).item()
        rmse = util.masked_rmse(predict_mean,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        # Ouput size: (batch_size,seq_length,num_nodes,1)
        output_mean,output_var, alpha, gamma = self.model(input)
        output_mean,output_var = output_mean.transpose(1,3),output_var.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict_mean = self.scaler.inverse_transform(output_mean) # Inverse transform
        
        loss = self.loss(real, predict_mean, output_var,alpha, gamma)
        mape = util.masked_mape(predict_mean,real,0.0).item()
        rmse = util.masked_rmse(predict_mean,real,0.0).item()
        return loss.item(),mape,rmse