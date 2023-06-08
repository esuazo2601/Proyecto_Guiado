from gene import Gene

class NodeGene(Gene):
    def __init__(self, id, type):
        self.id: int = id
        self.type: str = type
