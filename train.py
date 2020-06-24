from torch import optim
from torch.autograd import Variable

import loadData
from Net import Net
from settings import epochs, device
import torch.nn.functional as F

from utils import log

trainloader, testloader = loadData.load_split_test()

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("lol")

def train(e):
    model.train()
    for data, type in trainloader:
        log("Start for loop")
        batch_id=0

        data=Variable(data).to(device)
        type=Variable(type).to(device)
        optimizer.zero_grad()
        out=model(data)
        criterion= F.nll_loss
        loss=criterion(out, type)
        loss.backward()
        optimizer.step()
        log(f"Epoch {e}/{epochs}\t{batch_id*len(data)}/{len(trainloader)}\tLoss: {loss.data[0]}")
        batch_id+=1
        break


if __name__ == '__main__':
    train(1)
    for e in range(1, epochs + 1):
        pass
