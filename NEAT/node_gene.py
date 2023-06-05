from gene import Gene

class NodeGene(Gene):
    def __init__(self, x, y, innovation_number):
        super().__init__(innovation_number)
        self.x = x
        self.y = y
