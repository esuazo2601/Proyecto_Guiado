from gene import Gene

class NodeGene(Gene):
    def __init__(self, id: int, type: str):
        self.id = id        # Numero de nodo
        self.type = type    # Tipo de nodo (INPUT, OUTPUT, HIDDEN)
