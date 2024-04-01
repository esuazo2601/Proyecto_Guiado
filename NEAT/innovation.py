from .node_genes import Genes

class Innovation:
    innovation = 0
    exists: dict[(int, int)] = {}

    def get(genes1: int, genes2: int):
        num = Innovation.exists.get((genes1, genes2))                   # Verifica si existe una conexion con Numero de Innovacion asignado
        if num is None:                                                 # En caso de que no exista, le entregara "None"
            Innovation.innovation += 1                                  # Crea un nuevo Numero de Innovacion
            Innovation.exists[(genes1, genes2)] = Innovation.innovation # Agrega un Numero de Innovacion a la Conexion creada
            num = Innovation.innovation
        return num