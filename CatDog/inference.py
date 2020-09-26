import torch
from ModelClass import ConvNet
from PIL import Image
from torchvision import transforms

model = ConvNet()
model.load_state_dict(torch.load("Models/Model.pth"))

transform = transforms.Compose([
    transforms.ToTensor()])


def inference(img_path):
    img = Image.open(img_path)
    img = img.resize((200, 200))
    tensor = transform(img)
    tensor = torch.unsqueeze(tensor, 0)
    out = model(tensor)
    print(torch.max(out.data))
    _, predicted = torch.max(out.data, 1)
    return "Katze" if predicted.item() == 0 else "Hund", round(torch.max(out.data).item(), 1)


print("katze: " + str(inference("Validation/katze.jpg")))
print("Hund: " + str(inference("Validation/Hund.jpg")))
print("katze2" + str(inference("Validation/Katze2.jpeg")))
