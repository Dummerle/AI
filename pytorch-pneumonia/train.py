import os

import loadData
import torch.nn.functional as F
from Net import Net
from settings import epochs, device
from torch import optim
from torch.autograd import Variable
from utils import log

trainloader = loadData.load_split_test()

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(e):
    model.train()
    batch_id = 0
    for data, type in trainloader:
        data = Variable(data).to(device)
        type = Variable(type).to(device)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, type)
        loss.backward()
        optimizer.step()
        # log(f"Epoch {e}/{epochs}\t{batch_id * len(data)}/{len(trainloader)}\tLoss: {loss.data[0]}")
        log(f"{batch_id}/{len(trainloader)} finished")
        log(f"Loss: {loss.data}")
        batch_id += 1
    log(f"Epoch {e}/{epochs} finished")


def test():
    model.eval()
    files = os.listdir("Data/test/")
    for data in trainloader:
        #data = Variable(data).to(device)
        print(data)
        break


if __name__ == '__main__':
    # train(1)
    test()
    for e in range(1, epochs + 1):
        pass
