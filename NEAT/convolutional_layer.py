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
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        return x

# Crear una instancia del modelo
#modelo = CNN()

# Ejemplo de uso
#input_data = torch.randn(1, 3, 210, 160)  # Entrada de ejemplo
#output = modelo(input_data)

# Aplanar el tensor de salida con flatten()
#flattened_output = torch.flatten(output, start_dim=1)
#print(flattened_output.shape)
