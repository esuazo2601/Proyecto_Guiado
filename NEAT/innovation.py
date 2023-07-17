from .node_genes import Genes

class Innovation:
    innovation = 0
    exists: dict[(int, int)] = {}

    def get(genes1: Genes, genes2: Genes):
        num = Innovation.exists[(genes1.id, genes2.id)]         # Verifica si existe una conexion con numero de innovacion asignado
        if num == 0:                                            # En caso de que no exista, le entregara un valor igual a 0.
            Innovation.innovation += 1                          # Crea un nuevo Valor de Innovacion
            num = Innovation.innovation
        return num