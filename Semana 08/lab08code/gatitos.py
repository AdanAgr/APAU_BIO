import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

def CreateMosaic(inputs, filename="cats.vs.dogs.png"):
    selected_indices = random.sample(range(inputs.size(0)), 16)
    to_pil = transforms.ToPILImage()
    images = [to_pil(inputs[i]) for i in selected_indices]
    img_width, img_height = images[0].size
    grid_image = Image.new('RGB', (img_width * 4, img_height * 4))

    # Paste images into the grid
    for i, img in enumerate(images):
        row = i // 4  # Get the row (0-3)
        col = i % 4   # Get the column (0-3)
        grid_image.paste(img, (col * img_width, row * img_height))

    # Save the resulting grid image
    grid_image.save(filename)

# Ruta del conjunto de datos
dataset_dir = "/home/estudante/Escritorio/APAU_BIO/Tema8/cats_vs_dogs_small/"

num_classes = 2
img_dim = 180
num_epochs = 50
batch_size = 64

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

# Transformación de imágenes
custom_transform = transforms.Compose([
    transforms.Resize((img_dim, img_dim)),
    transforms.ToTensor()
])

# Cargar las imágenes de entrenamiento, validación y prueba
train_dir = "/home/estudante/Escritorio/APAU_BIO/Tema8/cats_vs_dogs_small/train"
val_dir = "/home/estudante/Escritorio/APAU_BIO/Tema8/cats_vs_dogs_small/validation"
test_dir = "/home/estudante/Escritorio/APAU_BIO/Tema8/cats_vs_dogs_small/test"

train_dataset = datasets.ImageFolder(train_dir, transform=custom_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=custom_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=custom_transform)

print("Length of training dataset:", len(train_dataset), "samples")
print("Length of validation dataset:", len(val_dataset), "samples")
print("Length of test dataset:", len(test_dataset), "samples")

# Crear DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Red neuronal simple 
net = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(32, 64, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(64, 128, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(128, 256, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(256, 256, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(12544, num_classes),
    torch.nn.Softmax(dim=1)
  
).to(device)

# Imprimir el modelo
print(net)

total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

# Función de pérdida y optimizador (usar Adam)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)  # Usar una tasa de aprendizaje más baja

# Vectores para almacenar las métricas
loss_v = np.empty(0)
loss_val_v = np.empty(0)
accuracy_v = np.empty(0)
accuracy_val_v = np.empty(0)

# Entrenamiento
for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if epoch == 0 and i == 0:
            CreateMosaic(inputs)  # Crear mosaico de imágenes en el primer batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(inputs)
        batch_loss = loss_fn(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        # Estadísticas
        train_loss += batch_loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # Evaluación en el conjunto de validación al final de cada época
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct_predictions_val = 0
        total_samples_val = 0

        for inputs_val, labels_val in val_loader:
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = net(inputs_val)
            batch_loss_val = loss_fn(outputs_val, labels_val)

            val_loss += batch_loss_val.item()
            _, predicted = torch.max(outputs_val, 1)
            correct_predictions_val += (predicted == labels_val).sum().item()
            total_samples_val += labels_val.size(0)

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions_val / total_samples_val

    accuracy = correct_predictions / total_samples
    train_loss = train_loss / len(train_loader)

    print(f"Epoch {epoch+1:02d}: loss {train_loss:.4f} - accuracy {accuracy:.4f} - val. loss {val_loss:.4f} - val. acc. {val_accuracy:.4f}")

    # Guardar métricas
    loss_v = np.append(loss_v, train_loss)
    loss_val_v = np.append(loss_val_v, val_loss)
    accuracy_v = np.append(accuracy_v, accuracy)
    accuracy_val_v = np.append(accuracy_val_v, val_accuracy)

# Graficar los resultados

num_epochs_stop = len(loss_val_v)
epochs = range(1, num_epochs_stop + 1)

# Loss
plt.figure()
plt.plot(epochs, loss_v, 'b-o', label='Training ')
plt.plot(epochs, loss_val_v, 'r-o', label='Validation ') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylim((0, 2))
plt.legend()
plt.savefig("Gatos.vs.Perros.Loss.png")

# Accuracy
accuracy_v = accuracy_v[0:num_epochs_stop]
accuracy_val_v = accuracy_val_v[0:num_epochs_stop]
plt.figure()
plt.plot(epochs, accuracy_v, 'b-o', label='Training ')
plt.plot(epochs, accuracy_val_v, 'r-o', label='Validation ') 
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig("Gatos.vs.Perros.Accuracy.png")
