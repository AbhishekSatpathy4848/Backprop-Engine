import math

class Value:
    
    def __init__(self, data, children=()):
        self.data = data
        self.children = set(children)
        self.grad = 0.0
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, element):
        assert isinstance(element, (int, float, Value))
        element = element if isinstance(element, Value) else Value(element)
        output = Value(self.data + element.data, (self, element))
        
        def _backward():
            self.grad += output.grad
            element.grad += output.grad
            
        output._backward = _backward
        return output

    def __mul__(self, element):
        assert isinstance(element, (int, float, Value))
        element = element if isinstance(element, Value) else Value(element)
        output = Value(self.data * element.data, (self, element))
            
        def _backward():
            self.grad += output.grad * element.data
            element.grad += output.grad * self.data
            
        output._backward = _backward
        return output
    
    def __pow__(self, element):
        assert isinstance(element, (int, float))
        output = Value(self.data ** element, (self,))
            
        def _backward():
            self.grad += output.grad * (element * (self.data ** ( element - 1 )))
    
        output._backward = _backward
        return output
    
    def tanh(self):
        n = self.data
        out = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        
        def _backward():
            self.grad += output.grad * (1-(out)**2)
        
        output = Value(out, (self,))
        output._backward = _backward
        return output
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, element):
        return self + (-element)
    
    def __truediv__(self, element):
        return self * (element ** -1)
    
    def __radd__(self, element):
        return self + element
    
    def __rsub__(self, element):
        return self - element
    
    def __rmul__(self, element):
        return self * element
    
    def __rtruediv__(self, element):
        return element * (self ** -1)
    
    def backward(self):
        stack = []
        visited = set()
        
        def toposort(v):
            visited.add(v)
            for child in v.children:
                if child not in visited:
                    toposort(child)
            stack.append(v)
        
        toposort(self)
        self.grad = 1

        for node in reversed(stack):
            node._backward()