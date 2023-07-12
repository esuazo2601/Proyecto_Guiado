from genes import Genes
from node_genes import NodeGenes
import random

class Connection():
    def __init__(self, Input: Genes, Output: Genes, Innovation: int):
        self.Input = Input              # Nodo de Entrada
        self.Output = Output            # Nodo de Salida
        self.Weight = random.random()   # Peso de la Conexion
        self.Enabled = True             # Inicialmente Habilitado
        self.Innovation = Innovation    # Numero de Innovacion  #TODO: Implementar

    def copy(self):
        return self.Input, self.Output, self.Weight, self.Enabled, self.Innovation

class ConnectionGenes(NodeGenes):
    def __init__(self, nodes: NodeGenes):
        self.genes = list[Connection] = {}

        num = 1
        for in_node in nodes:
            for out_node in nodes:
                if in_node.type == "INPUT" and out_node.type == "OUTPUT":
                    conn = Connection(in_node, out_node, num)
                    self.genes.append(conn)
                    num = num + 1
    
    def add_connection():
        pass

    def copy(self):
        return ConnectionGenes(self.in_node, self.out_node, self.weight, self.innovation)