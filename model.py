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


class eletrical_dataset(Dataset):
    def __init__(self, csv_path):
        self.dataframe = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        
        
        x = self.dataframe["Electricity_Consumed"][index]
        y = self.dataframe["Electricity Price"][index]
        

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor,y_tensor




class eletric_neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        
        self.drop = nn.Dropout(0.1)

        self.layer2 = nn.Linear(32, 16)

        self.drop1 = nn.Dropout(0.1)

        self.layer3 = nn.Linear(16, 1)
    
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
    
    total_loss = 0
    for epoch in tqdm(range(1, 100 + 1), desc=Fore.LIGHTRED_EX + "Running:" + Style.RESET_ALL, colour='Red'):

        for batch in tqdm(dst_loader, desc=Fore.GREEN+"loading_data"+Style.RESET_ALL, colour="Green", mininterval=0.3):
            x, y = batch
            x = torch.tensor(x, dtype=torch.float32).to('cuda')
            y = torch.tensor(y, dtype=torch.float32).to('cuda')
            
            with autocast(device_type="cuda"):
                out = model(x)
        
                loss = criterion(out, y)
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
        

