from torchvision.transforms import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 10

