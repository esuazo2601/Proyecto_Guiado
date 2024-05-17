from .genome import Genome
from typing import Self
import numpy as np
import math

# Funciones de activacion
def sig(x):
 return 1/(1 + np.exp(-x))

def relu(x):
    return max(0.0, x)

def leaky_relu(x):
  if x>0 :
    return x
  else :
    return 0.01*x

def tanh(x):
    return math.tanh(x)

def indentity(x):
    return x

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Clase encargada de ejecutar la red seleccionada por neat y reconstruirla
# a partir de los archivos .pickle generados por este

class Neuron:
    def __init__(self, type: str) -> None:
        self.input: list[Self] = []
        self.output: list[Self] = []
        self.value: float = 0
        self.weight: dict[Self] = {}
        self.type = type
        self.ready = False
    
    def __str__(self):
        return f"""
input = {self.input}
output = {self.output}
weigths = {self.weight}
type = {self.type}
ready = {self.ready}
            """

class NeuralNetwork:
    def __init__(self, genome: Genome):
            self.neuron: dict[int, Neuron] = {}
            self.outputs = []
            self.inputs = []

            # Initialize neurons
            for n in genome.nodes.genes:
                node = Neuron(n.id)
                node.type = n.type
                self.neuron[n.id] = node
                if n.type == "OUTPUT":
                    self.outputs.append(node)
                elif n.type == "INPUT":
                    node.ready = True
                    self.inputs.append(node)
            
            # Print neuron information for debugging
            print("Neurons initialized:", self.neuron.keys())

            # Connect neurons based on genome connections
            for conn in genome.connections.genes.values():
                if not conn.Enabled:
                    continue
                _in = conn.Input
                _out = conn.Output
                _weight = conn.Weight

                # Check if both input and output neurons exist
                if _in in self.neuron and _out in self.neuron:
                    self.neuron[_out].input.append(self.neuron[_in])
                    self.neuron[_in].output.append(self.neuron[_out])
                    self.neuron[_in].weight[self.neuron[_out]] = _weight
                else:
                    missing_neurons = []
                    if _in not in self.neuron:
                        missing_neurons.append(_in)
                    if _out not in self.neuron:
                        missing_neurons.append(_out)
                    print(f"Warning: Missing neurons for connection: {_in} -> {_out}. Missing neuron IDs: {missing_neurons}")
            
            # DEBUG: Print out the connections for verification
            #for neuron_id, neuron in self.neuron.items():
            #    print(f"Neuron {neuron_id}: inputs: {[n.id for n in neuron.input]}, outputs: {[n.id for n in neuron.output]}")
        


    # TODO: fix infinite loop
    def forward(self, _input: dict[int, float]):
        for node_id, value in _input.items():
            if node_id in self.neuron:
                self.neuron[node_id].value = value 
        
        queue = []
        for n in self.outputs:
            queue.append(n)
        
        while queue:
            n = queue.pop()
            n.ready = True
            for i in n.input:
                if i.ready:
                    n.value += i.value * i.weight[n]
                else:
                    n.ready = False
                    queue.insert(0, i)
            if n.ready:
                if n.type != "INPUT":
                    n.value = relu(n.value)  # Change activation function here if needed
        
        # Collect and return the output values, normalized with softmax
        output_values = [x.value for x in self.outputs]
        return softmax(output_values)