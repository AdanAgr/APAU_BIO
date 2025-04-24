import medmnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print(medmnist.__version__)
from medmnist import PneumoniaMNIST



# Información del dataset
download = True
info = medmnist.INFO['pneumoniamnist']
print(info)

# Obtener la clase de datos
DataClass = getattr(medmnist, info['python_class'])

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Cargar dataset
train_dataset = DataClass(split='train', transform=transform, download=download)

healthy_images = []
pneumonia_images = []

for img, label in train_dataset:
    if label.item() == 0:
        healthy_images.append(img.squeeze())  # quitar dimensión del canal
    if len(healthy_images) == 12:
        break

for img, label in train_dataset:
    if label.item() == 1:
        pneumonia_images.append(img.squeeze())
    if len(pneumonia_images) == 12:
        break