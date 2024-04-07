import torch
import torch.nn as nn

# Definir la arquitectura de la red neuronal convolucional
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 52 * 40, 64) # Calculated based on input dimensions
        self.fc2 = nn.Linear(64, 10) # 10 classes for example
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 52 * 40) # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear una instancia del modelo
# modelo = CNN()

# Imprimir el resumen del modelo
# print(modelo)
