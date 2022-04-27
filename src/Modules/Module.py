class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        pass

    def forward(self, X):
        pass

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass
