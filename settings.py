from torchvision.transforms import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 3

#normalize = transforms.Normalize(mean=[0.485], std=[0.229])

transformation = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                     transforms.Resize(1217),
                                     transforms.CenterCrop(1217),
                                     transforms.ToTensor()])
