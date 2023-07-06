from gene import Gene
from node_gene import NodeGene

class ConnectionGene(Gene):
    def __init__(self, in_node: NodeGene, out_node: NodeGene, weight: float, innov: int):
        self.in_node = in_node                  # Nodo de entrada
        self.out_node = out_node                # Nodo de salida
        self.weight = weight                    # Peso de la conexion
        self.enabled: bool = True               # Inicialmente habilitado
        self.innovation = innov                 # Numero de innovacion
    
    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.innovation)