import random
from engine import Value

class Neuron:
    
    def __init__(self, nin):
        self.w = [Value(0.8 if i%2==0 else -0.8, label=f'w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        assert len(x) == len(self.w)
        return sum((xi * wi for xi, wi in zip(x, self.w)), 0)
    
    def parameters(self):
        return self.w + [self.b]
    
    def zero_grad(self):
        self.b.grad = 0
        for weight in self.w:
            weight.grad=0 

class Layer:
    
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
        
    def zero_grad(self):
        for neurons in self.neurons:
            neurons.zero_grad()
    
class MLP:
    
    def __init__(self, nin, nout_list):
        updated_nout_list = [nin] + nout_list #append the number of inputs to the list
        self.layers = [Layer(updated_nout_list[i], updated_nout_list[i+1]) for i in range(0, len(nout_list))]
   
    def __call__(self, x):
        inputs = x
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs
        return inputs[0] if len(inputs) == 1 else inputs
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()