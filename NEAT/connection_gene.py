from gene import Gene

class ConnectionGene(Gene):
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node

        self.weight: float
        self.enabled: bool