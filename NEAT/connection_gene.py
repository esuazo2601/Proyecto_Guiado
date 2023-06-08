from gene import Gene

class ConnectionGene(Gene):
    def __init__(self, in_node, out_node, innov):
        self.in_node: int = in_node
        self.out_node: int = out_node
        self.innovation: int = innov
        self.weight: float
        self.enabled: bool