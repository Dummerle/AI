import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from Net import Net
from settings import epochs, device, batch_size
from utils import log

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root="Data/train", transform=transform)
test_dataset = datasets.ImageFolder(root="Data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def train(e):
    model.train()
    running_loss = 0.0
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        out = model(data)

        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        # log(f"{batch_id}/{len(trainloader)} finished")
        # log(f"Loss: {loss.data}")
        running_loss += loss.item()
    log(f"Epoch {e}/{epochs} finished | loss: {round(running_loss, 3)}")


def test():
    model.eval()
    correct = 0
    total = 0
    for data, type in test_loader:
        type = type.to(device)
        data = data.to(device)
        out = model(data)
        _, predicted = torch.max(out.data, 1)
        total += type.size(0)
        correct += (predicted == type).sum().item()
    print(str(round(correct / total * 100, 2)) + "%")
    return round(correct/total, 3)


if __name__ == '__main__':
    max_val = 0.0
    for e in range(1, epochs + 1):
        train(e)
        val = test()
        if val >= max_val:
            max_val = val
            torch.save(model.state_dict(), "Models/MaxVal.pth")
torch.save(model.state_dict(), "Models/EndModel.pth")
