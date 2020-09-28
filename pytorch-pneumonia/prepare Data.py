import os

import PIL.Image as Image

path = "/home/lennard/Datasets/Pneumonia/test/NORMAL/"

for im in os.listdir(path):
    try:
        img = Image.open(path + im)
        img = img.resize((200, 200))
        img.save("Data/test/normal/"+im)
    except:
        pass