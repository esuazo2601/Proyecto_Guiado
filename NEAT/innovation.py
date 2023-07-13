from genes import Genes

class Innovation:
    innovation = 0
    exists: dict[(int, int)] = {}

    def get():
        Innovation.innovation += 1
        return Innovation.innovation

    def innovation_exists(self, genes1: Genes, genes2: Genes):  # Verifica si existe una conexion con numero de innovacion asignado
        return self.exists[(genes1.id, genes2.id)]              # En caso positivo, entregara un valor distinto de 0.