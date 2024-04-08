import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calcula el tamaño esperado después de las operaciones de convolución y agrupación
        self.fc_input_size = self._get_fc_input_size()
        
        self.fc1 = nn.Linear(self.fc_input_size, 128)  # Ajustamos el tamaño de entrada de fc1
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Aplanar el tensor antes de pasar por la capa completamente conectada
        x = x.reshape(-1, self.fc_input_size)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _get_fc_input_size(self):
        # Función auxiliar para calcular el tamaño de entrada para la capa completamente conectada
        # Utiliza una entrada de ejemplo para calcular el tamaño
        with torch.no_grad():
            x = torch.zeros(1, 3, 210, 160)  # Tamaño de entrada de ejemplo
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            return x.reshape(1, -1).size(1)

# Crear una instancia del modelo
modelo = CNN()
