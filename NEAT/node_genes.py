class Genes():
    def __init__(self, id: int, type: str):
        self.id = id        # Numero de Nodo
        self.type = type    # Tipo de Nodo (INPUT, OUTPUT, HIDDEN)

class NodeGenes():
    def __init__(self, inputSize, outputSize):
        self.genes = list[Genes] = {}

        inputN : list[NodeGenes] = {}
        outputN : list[NodeGenes] = {}

        for i in range(0, inputSize):
            n = NodeGenes(i+1, "INPUT")
            inputN.append(n)
            self.genes.append(n)

        for i in range(inputSize, inputSize+outputSize):
            n = NodeGenes(i+1, "OUTPUT")
            outputN.append(n)
            self.genes.append(n)

    def add_node(self, node: Genes):
        self.genes.append(node)