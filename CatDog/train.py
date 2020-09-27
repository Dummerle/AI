import torch
from matplotlib import pyplot
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from CatDog.ModelClass import ConvNet

BATCH_SIZE = 32
EPOCHS = 10

# LRS = [0.001, 0.0005, 0.0003, 0.0001]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.ImageFolder(root="Data/train", transform=transform)
test_dataset = datasets.ImageFolder(root="Data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

model = ConvNet()
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
print("TRAINING ON " + torch.cuda.get_device_name() if DEVICE == "cuda" else "CPU")
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training
def train():
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return round(running_loss / 100, 3)


# Validation
def val():
    model.eval()
    correct = 0
    total = 0

    for data in test_loader:
        img, label = data
        img, label = img.to(DEVICE), label.to(DEVICE)
        out = model(img)
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    return round(correct / total * 100)


max_val = 0
losses = []
validations = []
if __name__ == '__main__':

    for e in range(EPOCHS):
        print(f"TRAINING EPOCH {e + 1}/{EPOCHS}", end="  ")
        loss = train()
        losses.append(loss)
        print(f"loss: {loss}")
        valuation = val()
        if max_val <= valuation:
            max_val = valuation
            torch.save(model.state_dict(), "Models/maxVal.pth")
        validations.append(valuation)
        print(f"Validation: {valuation}, MaxVal = {max_val}\n")

    print("Validation: ", val())
    print(f"TRAINING FINISHED. Validation: {val()}")
    torch.save(model.state_dict(), "Models/EndModel.pth")
    pyplot.plot(losses)
    pyplot.ylabel("Loss")
    pyplot.xlabel("Epoch")
    pyplot.show()
    pyplot.plot(validations)
    pyplot.show()
