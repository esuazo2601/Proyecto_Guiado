from connection_gene import ConnectionGene
from node_gene import NodeGene
from genome import Genome

class NEAT:
    def __init__(self, inputSize, outputSize, clients):
        self.input_size: int
        self.output_size: int
        self.max_clients: int

        self.all_connections: list[ConnectionGene]
        self.all_nodes: list[NodeGene]

        self.C1: float
        self.C2: float
        self.C3: float

        self._reset(self.input_size, self.output_size, self.max_clients)

    def _empty_genome(self):
        g = Genome(self)
        for i in range(self.input_size + self.output_size):
            g.nodes.append(self._get_node(i + 1))
        
        return g

    def _reset(self, input_size, output_size, max_clients):
        self.input_size = input_size
        self.output_size = output_size
        self.max_clients = max_clients

        self.all_connections.clear()
        self.all_nodes.clear()

        for i in range(self.input_size):
            n: NodeGene = self._get_node()
            n.x = 0.1
            n.y = (i + 1) / (self.input_size + 1)

        for i in range(self.output_size):
            n: NodeGene = self._get_node()
            n.x = 0.9
            n.y = (i + 1) / (self.output_size + 1)

    def _get_connection(self, connection: ConnectionGene):
        c = ConnectionGene(connection.from_node, connection.to_node)
        c.innovation_number = connection.innovation_number
        c.weight = connection.weight
        c.enabled = connection.enabled
        return c

    def _get_connection(self, node1: NodeGene, node2: NodeGene):
        c = ConnectionGene(node1, node2)
        if c in self.all_connections:
            c.innovation_number = self.all_connections[self.all_connections.index(
                c)].innovation_number
        else:
            c.innovation_number = len(self.all_connections) + 1
            self.all_connections.append(c)
        return c

    def _get_node(self, id=None):
        if id and id <= len(self.all_nodes):
            return self.all_nodes[id - 1]

        n = NodeGene(len(self.all_nodes) + 1)
        self.all_nodes.append(n)
        return n
