import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch import nn
from torch.optim import AdamW
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from colorama import Style, Fore
import numpy as np

def extract_time(df):
    df2={}
    df2["Timestamp"] = pd.to_datetime[df["Timestamp"]]

    df2['hour'] = df["Timestamp"].dt.hour
    df2["day_of_week"] = df["Timestamp"].dt.dayofweek
    df2

    hours = 24
    df2["hours_sin"] = np.sin(2*np.pi * df["hour"]/hours)
    df2['hour_cos'] = np.cos(2* np.pi * df["hour"])

    day = 7
    df2['day_sin'] = np.sin(2*np.pi*df['day_of_week']/day)
    df2['day_cos'] = np.cos(2*np.pi*df["day_of_week"]/day)
    df2 = df2.drop(columns=["hour", "day_of_week"])

    return df2

class eletrical_dataset(Dataset):
    def __init__(self, csv_path):
        self.dataframe = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        
        
        x = self.dataframe["Electricity_Consumed"][:]
        y = self.dataframe["Temperature"][index]
        z = self.dataframe["Power Factor"][index]
        x2 = self.dataframe["Grid Supply (kW)"][index]
        y1 = self.dataframe["Humidity"][index]
        
        

        return




class eletric_neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(2, 64)
        
        self.drop = nn.Dropout(0.1)

        self.layer2 = nn.Linear(64, 32)

        self.drop1 = nn.Dropout(0.1)

        self.layer3 = nn.Linear(32, 2)
    
    def forward(self, data):

        x_embed = self.embed(data)
        x_embed= self.drop(x_embed)

        x_in = self.layer2(x_embed)
        x_in = self.drop1(x_in)

        x_in = self.layer3(x_in)
        
        return x_in


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = eletric_neuralnet()
model.to('cuda')


optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=2, eta_min=1e-6)


criterion = nn.MSELoss()
scaler = GradScaler()




if __name__ == "__main__":

    dst = eletrical_dataset(r'D:\HackForChange\smart_grid_dataset.csv')
    dst_loader = DataLoader(dst, batch_size=1, num_workers=12, prefetch_factor=24)

    model.train()
    #"""
    total_loss = 0
    for epoch in tqdm(range(1, 100 + 1), desc=Fore.LIGHTRED_EX + "Running:" + Style.RESET_ALL, colour='Red'):

        for batch in tqdm(dst_loader, desc=Fore.GREEN+"loading_data"+Style.RESET_ALL, colour="Green", mininterval=0.3):
            y_out, y_in = batch
            y_in = torch.tensor(y_in, dtype=torch.float32).to('cuda')
            y_out = torch.tensor(y_in, dtype=torch.float32).to('cuda')


            with autocast(device_type="cuda"):
                out = model(y_in)
        
                loss = criterion(out, y_out)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
    
        print(f"Epoch: {epoch:03d} | Loss: {loss:.4f} | LR: {current_lr:.6f}")
       
        if epoch % 10 == 0:
            torch.save({"model_state_dict":model.state_dict()}, rf"weather_ratio_{epoch}_{loss}.pt")
    print("Advanced training complete!")
    

    """

    total_loss = 0
    for epoch in tqdm(range(1, 100 + 1), desc=Fore.LIGHTRED_EX + "Running:" + Style.RESET_ALL, colour='Red'):

        for batch in tqdm(dst_loader, desc=Fore.GREEN+"loading_data"+Style.RESET_ALL, colour="Green", mininterval=0.3):
            x, y, z, x2, y2 = batch
            x = torch.tensor(x, dtype=torch.float32).to('cuda')
            y = torch.tensor(y, dtype=torch.float32).to('cuda')
            z = torch.tensor(z, dtype=torch.float32).to('cuda')
            y2 = torch.tensor(y2, dtype=torch.float32).to('cuda')

            with autocast(device_type="cuda"):
                out = model([z, y2])
        
                loss = criterion(out, x)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
    
        print(f"Epoch: {epoch:03d} | Loss: {loss:.4f} | LR: {current_lr:.6f}")
       
        if epoch % 10 == 0:
            torch.save({"model_state_dict":model.state_dict()}, rf"model_checkpoint_epoch_{epoch}_{loss}.pt")
    print("Advanced training complete!")
        """

