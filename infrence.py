import torch
from model import eletric_neuralnet

model = eletric_neuralnet()
model.to("cuda")

model.eval()
model.state_dict(torch.load(r"model_checkpoint_epoch_20_0.0026973667554557323.pt")["model_state_dict"])
inPT = torch.tensor(1.1914022179376131, dtype=torch.float32)
inPT = inPT.unsqueeze(0).to("cuda")
out = model(inPT)
print(out)





