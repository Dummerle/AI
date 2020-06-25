import os

import loadData
import torch.nn.functional as F
from Net import Net
from settings import epochs, device
from torch import optim, nn
from torch.autograd import Variable
from utils import log

trainloader, testloader = loadData.load_split_test()

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
        criterion = nn.NLLLoss()
        loss = criterion(out, type)
        loss.backward()
        optimizer.step()


        # log(f"{batch_id}/{len(trainloader)} finished")
        # log(f"Loss: {loss.data}")
        batch_id += 1
    log(f"Epoch {e}/{epochs} finished")


def test():
    model.eval()
    correct = 0
    len_test = len(testloader)
    files = os.listdir("Data/test/")
    for data, type in testloader:
        type = Variable(type).to(device)
        data = Variable(data).to(device)
        out = model(data)

        if type == out.data.max(1, keepdim=True)[1]:
            correct += 1

    print(str(correct / len_test * 100) + "%")


if __name__ == '__main__':

    for e in range(1, epochs + 1):
        train(e)
        test()
