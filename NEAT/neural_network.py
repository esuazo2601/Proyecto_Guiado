from .genome import Genome
import torch
import torch.nn.functional as F

# Funciones de activación usando PyTorch
def relu(x):
    return F.relu(x)

def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.01)

def tanh(x):
    return torch.tanh(x)

def identity(x):
    return x

def softmax(x):
    return F.softmax(x, dim=0)

class Neuron:
    def __init__(self, type: str, device: torch.device) -> None:
        self.input: list[Neuron] = []
        self.output: list[Neuron] = []
        self.value: torch.Tensor = torch.tensor(0.0, device=device)  # Inicializa el tensor en el dispositivo
        self.weight: dict[Neuron, torch.Tensor] = {}
        self.type = type
        self.ready = False
        self.device = device
    
    def __str__(self):
        return f"""
input = {self.input}
output = {self.output}
weights = {self.weight}
type = {self.type}
ready = {self.ready}
        """

class NeuralNetwork:
    def __init__(self, genome: Genome, device: torch.device):
        self.neuron: dict[int, Neuron] = {}
        self.outputs = []
        self.inputs = []
        self.device = device
        for n in genome.nodes.genes:
            node = Neuron(n.type, device)
            node.type = n.type
            self.neuron[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.ready = True
                self.inputs.append(node)
        
        for n, conn in genome.connections.genes.items():
            if not conn.Enabled:
                continue
            _in = conn.Input
            _out = conn.Output
            _weight = conn.Weight
            self.neuron[_out].input.append(self.neuron[_in])
            self.neuron[_in].output.append(self.neuron[_out])
            self.neuron[_in].weight[self.neuron[_out]] = torch.tensor(_weight, device=device)  # Mueve el peso al dispositivo
        
    def forward(self, _input: dict[int, float]):
        for node_id, value in _input.items():
            if node_id in self.neuron:
                self.neuron[node_id].value = torch.tensor(value, device=self.device)  # Mueve el valor al dispositivo
        
        queue = []
        for n in self.outputs:
            queue.append(n)

        while len(queue) != 0:
            n = queue.pop()
            n.ready = True
            for i in n.input:
                if i.ready:
                    n.value += i.value * i.weight[n]
                    #print(f"Neuron {n}: value {n.value}, weight {i.weight[n]}")  # Debugging line
                else:
                    n.ready = False
                    queue.insert(0, i)
            if n.ready:
                if n.type != "INPUT":
                    n.value = leaky_relu(n.value)
            else:
                queue.insert(0, n)
        
        output_values = [x.value for x in self.outputs]
        return softmax(torch.stack(output_values))
