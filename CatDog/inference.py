import torch
from PIL import Image
from torchvision import transforms

from ModelClass import Net

model = Net()
model.load_state_dict(torch.load("Models/EndModel.pth"))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])


def inference(img_path):
    img = Image.open(img_path)
    img = img.resize((200, 200))
    tensor = transform(img)
    tensor = torch.unsqueeze(tensor, 0)
    out = model(tensor)
    _, predicted = torch.max(out.data, 1)
    return "Katze" if predicted.item() == 0 else "Hund"


print("katze: "+inference("Validation/katze.jpg"))
print("Hund: "+inference("Validation/Hund.jpg"))
