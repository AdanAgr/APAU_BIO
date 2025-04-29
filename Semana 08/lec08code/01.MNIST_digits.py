###  ---------- IMPORTS ---------- ###
import torch # librería deep learning
from torch.utils.data import random_split, DataLoader # dividir datasets y cargarlos en "mini-lotes" (DataLoader).
from torchvision import datasets, transforms # cargar datasets y transformar imágenes 
import sys # para salir del programa si no hay GPU disponible


###  ---------- CONFIGURACIÓN ---------- ###

### CHECK GPU ###
# Comprobar si hay GPU disponible
if torch.cuda.is_available():
    device = torch.device('cuda:0') # qué GPU usar
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    print("ERROR: no GPU available")
    sys.exit(0) # salir del programa si no hay GPU disponible



### ---------- DATASET---------- ###

### TRANSFORMACIONES ###
# se aplican a cada imagen del dataset:
custom_transform = transforms.Compose([
    transforms.ToTensor(), # convierte la imagen a un tensor
    transforms.Lambda(lambda x: x.view(-1)) # aplana imagen 28x28 (2D) a vector de 784 etos (1D)
])

### CARGAR DATASET ###
# se descarga en /tmp/data si no lo tienes + aplica las transformaciones
mnist_dataset = datasets.MNIST(root='/tmp/data', download=True, transform = custom_transform)
print("Length of dataset:", len(mnist_dataset)) # nº ejemplos
print("Length of first vector in dataset: ", mnist_dataset[0][0].shape) # shape del primer ejemplo (784,)
print("Label of first vector in dataset: ", mnist_dataset[0][1]) # label del primer ejemplo (0-9)

### DIVIDIR DATASET ###
train_data, val_data = random_split(mnist_dataset, [50000, 10000]) # 50k train, 10k validation

## Print the length of train and validation datasets
print("Length of Train Dataset: ", len(train_data))
print("Length of Test Dataset: ", len(val_data))

batch_size = 128 # tamaño cada mini-lote (de 128 en 128 imágenes)

train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=2) # divide datasets en mini-lotes aleatorios 
val_loader = DataLoader(val_data, len(val_data) , shuffle = False, num_workers=2) # carga todos los datos de validación de una vez


### ---------- RED NEURONAL ---------- ###

### DEFINICIÓN RED ###
net = torch.nn.Sequential( # modelo secuencial (los datos pasan de una capa a otra)
      torch.nn.Linear(784, 512), # reducción de 784 entradas a 512 salidas
      torch.nn.ReLU(), # función de activación ReLU (quita negativos, deja positivos)
      torch.nn.Linear(512, 10), # reducción de 512 entradas a 10 salidas (una para cada dígito)
      torch.nn.Softmax(dim=1) # función de activación Softmax (normaliza las salidas para que sumen 1 --> probabilidades)
      ).to(device)

print(net) # imprime la red neuronal


### ---------- ENTRENAMIENTO ---------- ###

# entrenar el modelo durante 50 ciclos completos (epochs) sobre todos los datos
num_epochs = 50

# optimizador RMSprop, que ajusta los pesos
optimizer = torch.optim.RMSprop(net.parameters(), lr = 0.001) 

# función de pérdida (CrossEntropyLoss) para clasificación multiclase
criterion = torch.nn.CrossEntropyLoss()


### BUCLE DE ENTRENAMIENTO ###

# Por cada epoch:
for epoch in range(num_epochs):

    ## 🔴 MODO ENTRENAMIENTO:
    net.train() # activa dropout, batchnorm, etc.

    total_loss = 0.0 # inicializa la pérdida total
    correct_predictions = 0 # inicializa el nº de predicciones correctas
    total_samples = 0 # inicializa el nº de ejemplos totales

    # para CADA MINI-BATCH de datos de entrenamiento 
    # (train_loader está dividido en mini-batches):
    for i, data in enumerate(train_loader, 0):
        
        # imágenes y etiquetas
        inputs, labels = data # inputs: imágenes, labels: etiquetas (0-9)
        inputs = inputs.to(device) # mueve los datos a la GPU
        labels = labels.to(device) # mueve las etiquetas a la GPU

        # forward + backward + optimize
        optimizer.zero_grad() # limpiar los gradientes acumulados previamente

        outputs = net(inputs) # hacer predicción con la red (forward pass)
        loss = criterion(outputs, labels) # calcular la pérdida (CrossEntropyLoss)
        loss.backward() # calcular los gradientes (backward pass)
        optimizer.step() # actualizar los pesos de la red (optimización)

        # statistics after a batch
        total_loss += loss.item() # acumula la pérdida total por batch
        _, predicted = torch.max(outputs, 1) # clase con mayor probabilidad
        correct_predictions += (predicted == labels).sum().item() # acumula nº de predicciones correctas
        total_samples += labels.size(0) # nº de ejemplos procesados


    ## 🔴 MODO VALIDACIÓN (al final de cada epoch):
    net.eval() # desactiva dropout, batchnorm, etc.

    with torch.no_grad(): # NO se calculan gradientes en la evaluación

        data_iter = iter(val_loader) # carga datos de validación

        inputs_val, labels_val = next(data_iter) # inputs: imágenes, labels: etiquetas
        inputs_val = inputs_val.to(device) # mueve las imágenes a la GPU
        labels_val = labels_val.to(device) # mueve las etiquetas a la GPU

        outputs_val = net(inputs_val) # hacer predicción con la red (forward pass)

        _, predicted = torch.max(outputs_val, 1) # clase con mayor probabilidad
        correct_predictions_val = (predicted == labels_val).sum().item() # acumula nº de predicciones correctas
        total_samples_val = labels_val.size(0) # nº de ejemplos procesados

    accuracy = correct_predictions / total_samples # exactitud del modelo
    accuracy_val = correct_predictions_val / total_samples_val # exactitud del modelo en validación
    average_loss = total_loss / len(train_loader) # pérdida media por batch
    
    # STATS
    print("Epoch {:02d}: loss {:.4f} - accuracy {:.4f} - validation accuracy {:.4f}".format(epoch+1, average_loss, accuracy, accuracy_val))