""" import torch
import torch.nn as nn

# Crear una capa de convolución
conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# Crear un tensor de entrada de ejemplo (batch_size, canales, altura, ancho)
input_tensor = torch.randn(1, 3, 32, 32)

# Aplicar la capa de convolución al tensor de entrada
output_tensor = conv_layer(input_tensor)

# Ver la forma del tensor de salida
print("Forma del tensor de salida:", output_tensor.shape)
 """

import torch

# Verificar si CUDA (GPU) está disponible
if torch.cuda.is_available():
    # Seleccionar el dispositivo CUDA
    device = torch.device()
    # Imprimir el nombre de la GPU
    print("GPU disponible:", torch.cuda.get_device_name(0))
else:
    # Si CUDA no está disponible, utilizar la CPU
    device = torch.device("cpu")
    print("CUDA no está disponible, utilizando la CPU.")
