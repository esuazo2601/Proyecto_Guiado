from .node_genes import NodeGenes, Genes
from .innovation import Innovation as inn_num
import random

class Connection():
    def __init__(self, Input: int, Output: int, Weight: float, Innovation: int):
        self.Input = Input              # Nodo de Entrada
        self.Output = Output            # Nodo de Salida
        self.Weight = Weight            # Peso de la Conexion
        self.Enabled = True             # Inicialmente Habilitado
        self.Innovation = Innovation    # Numero de Innovacion

    def copy(self):
        return self.Input, self.Output, self.Weight, self.Enabled, self.Innovation


class ConnectionGenes(NodeGenes):
    def __init__(self, nodes: NodeGenes):
        self.genes: dict[(int, int)] = {}
        self.node_count = len(nodes.genes)

        for in_node in nodes.genes:
            for out_node in nodes.genes:
                if in_node.type == "INPUT" and out_node.type == "OUTPUT":
                    conn = Connection(in_node.id, out_node.id, random.random(), inn_num.get(in_node.id, out_node.id))
                    self.genes[in_node.id, out_node.id] = conn

    """ 
    Crea un Nodo en una Conexion ya existente, resultando en lo siguiente:
    - El Nodo Inicial mantiene el Peso de la Conexion anterior
    - El Nodo Final crea una Conexion con Peso de valor 1
    """
    def mutate_add_node(self, conn: Connection):
        self.genes[conn.Input, conn.Output].Enabled = False             # Deshabilita la Conexion Anterior para Habilitar la Nueva

        self.node_count += 1                                            # Aumenta el contador de Nodos
        new_node = Genes(self.node_count, "HIDDEN")                     # Nuevo Nodo
        
        new_conn1 = Connection(conn.Input, new_node.id, conn.Weight, inn_num.get(conn.Input, new_node.id))
        self.genes[new_conn1.Input, new_conn1.Output] = new_conn1
        
        new_conn2 = Connection(new_node.id, conn.Output, 1, inn_num.get(new_node.id, conn.Output))
        self.genes[new_conn2.Input, new_conn2.Output] = new_conn2

        return new_node                                                 # Retorna el nuevo Nodo para que lo agregue a NodeGenes del Genome
    
    """ 
    Crea una Conexion entre dos Nodos ya existentes, resultando en lo siguiente:
    - Ambos Nodos se encuentran conectados y recibe un Numero de Innovacion
    """
    def mutate_add_connection(self, node1: int, node2: int):
        new_conn = Connection(node1, node2, random.random(), inn_num.get(node1, node2))
        self.genes[new_conn.Input, new_conn.Output] = new_conn

    """ 
    Cambia el valor del Peso de una Conexion, resultando en lo siguiente:
    - La Conexion actualiza el valor de su Peso basada en lo que recibe
    """
    def mutate_weight(self, new_weight: float, conn: Connection):
        self.genes[conn.Input, conn.Output].Weight = new_weight