import os

import torch
from PIL import Image
from torchvision import transforms

from Net import Net

model = Net()
model.load_state_dict(torch.load("Models/MaxVal.pth"))
model.eval()
transform = transforms.Compose([
    transforms.ToTensor()])


def inference(img_path):
    img = Image.open(img_path)
    img = img.resize((200, 200))
    img = img.convert("RGB")
    tensor = transform(img)
    tensor = torch.unsqueeze(tensor, 0)
    out = model(tensor)
    _, predicted = torch.max(out.data, 1)

    return "PN" if predicted.item() == 1 else "Normal"


correct, total = 0, 0

for type in os.listdir("validation"):

    for i in os.listdir(f"validation/{type}"):
        if inference(f"validation/{type}/{i}") == "Normal" and "NORMAL" in i:
            correct += 1
        elif inference(f"validation/{type}/{i}") == "PN" and "bacteria" in i:
            correct += 1
        else:
            print(i)

        total += 1

print(f"{correct}/{total}")
