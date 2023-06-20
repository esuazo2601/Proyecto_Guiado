from gene import Gene

class NodeGene(Gene):
    def __init__(self, id: int, type: str):
        self.id = id
        self.type = type
