from PIL import Image
import os

def show_image(filename):
    filepath = os.path.join("./network_visualizations"+filename + '.png')
    if os.path.exists(filepath):
        img = Image.open(filepath)
        img.show()
    else:
        print(f"File {filepath} does not exist.")

# Ejemplo de uso
show_image('genome_1210')