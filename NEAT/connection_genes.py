from genes import Genes
from node_genes import NodeGenes
import random

class Connection():
    def __init__(self, Input: Genes, Output: Genes, Weight: float, Innovation: int):
        self.Input = Input              # Nodo de Entrada
        self.Output = Output            # Nodo de Salida
        self.Weight = Weight            # Peso de la Conexion
        self.Enabled = True             # Inicialmente Habilitado
        self.Innovation = Innovation    # Numero de Innovacion

    def copy(self):
        return self.Input, self.Output, self.Weight, self.Enabled, self.Innovation
    
    def mutate_weight(self, new_weight):    #! Mutacion aplicada al Hijo de dos Padres en una Conexion Existente
        self.Weight = new_weight


class ConnectionGenes(NodeGenes):
    def __init__(self, nodes: NodeGenes):
        self.genes = list[Connection] = {}
        self.node_count = len(nodes)

        num = 1
        for in_node in nodes:
            for out_node in nodes:
                if in_node.type == "INPUT" and out_node.type == "OUTPUT":
                    conn = Connection(in_node, out_node, random.random(), num)
                    self.genes.append(conn)
                    num = num + 1

    """ 
    Crea un Nodo en una Conexion ya existente, resultando en lo siguiente:
    - El Nodo Inicial mantiene el Peso de la Conexion anterior
    - El Nodo Final crea una Conexion con Peso de valor 1
    """
    def mutate_add_node(self, conn: Connection):
        fix = 0                                                         # TODO: implementar innovation_number
                                                                        # TODO: implementar correctamente esto ↓
        self.genes[conn.Innovation].Enabled = False                     # Deshabilita la Conexion Anterior para Habilitar la Nueva

        self.node_count += 1                                            # Aumenta el contador de Nodos
        new_node = Genes(self.node_count, "HIDDEN")                     # Nuevo Nodo
        new_conn1 = Connection(conn.Input, new_node, conn.Weight, fix)  # Conexion desde Nodo Inicial a Nuevo Nodo
        new_conn2 = Connection(new_node, conn.Output, 1, fix)           # Conexion desde Nuevo Nodo a Nodo Final

        return new_node                                                 # Retorna el nuevo Nodo para que lo agregue a NodeGenes del Genome
    
    """ 
    Crea una Conexion entre dos Nodos ya existentes, resultando en lo siguiente:
    - Ambos Nodos se encuentran conectados y recibe un Numero de Innovacion
    """
    def mutate_add_connection(self, node1: NodeGenes, node2: NodeGenes, inn_num: int):
        new_conn = Connection(node1, node2, random.random(), inn_num)   # TODO: implementar innovation_number
        self.genes.append(new_conn)

    """ 
    Cambia el valor del Peso de una Conexion, resultando en lo siguiente:
    - La Conexion actualiza el valor de su Peso basada en lo que recibe
    """
    def mutate_weight(self, new_weight: float, conn: Connection):
        self.genes[conn.Innovation].mutate_weight(new_weight)           # TODO: implementar correctamente esto ←