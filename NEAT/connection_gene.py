from gene import Gene

class ConnectionGene(Gene):
    def __init__(self, from_node, to_node):
        self.in_node = from_node
        self.out_node = to_node

        self.weight: float
        self.enabled: bool