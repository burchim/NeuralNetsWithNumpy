import numpy as np

class Optimizer():

    def __init__(self, params, lr):

        self.params = params
        self.lr = lr
        self.t = 0

    def step(self):
        self.t += 1

    def zero_grad(self):

        for param in self.params.values():
            param.grad = None

    def get_state_dict(self):
        return {"t": self.t}

    def load_state_dict(self, state_dict):
        self.t = state_dict["t"]

class SGD(Optimizer):

    def __init__(self, params, lr, momentum=0, l2_reg=0):
        super(SGD, self).__init__(params, lr)

        self.momentum = momentum
        self.l2_reg = l2_reg
        self.velocity = {}
        for name, param in self.params.items():
            self.velocity[name] = np.zeros_like(param)

    def step(self):
        super(SGD, self).step()
    
        for name, param in self.params.items():

            # L2 Reg
            param.grad += self.l2_reg * param

            # velocity
            self.velocity[name] = self.momentum * self.velocity[name] + param.grad

            # Update
            param -= self.lr * self.velocity[name]

    def get_state_dict(self):
        state_dict = super(SGD, self).get_state_dict()
        state_dict["velocity"] = self.velocity
        return state_dict

    def load_state_dict(self, state_dict):
        super(SGD, self).load_state_dict(state_dict)
        self.velocity = state_dict["velocity"]

class AdaGrad(Optimizer):

    def __init__(self, params, lr=0.01, eps=1e-08, l2_reg=0):
        super(AdaGrad, self).__init__(params, lr)

        self.eps = eps
        self.l2_reg = l2_reg
        self.v = {}
        for name, param in self.params.items():
            self.v[name] = np.zeros_like(param)
        
    def step(self):
        super(AdaGrad, self).step()

        for name, param in self.params.items():

            # L2 Reg
            param.grad += self.l2_reg * param

            # Square Grad
            self.v[name] += param.grad**2

            # Update
            param -= self.lr * param.grad / (np.sqrt(self.v[name]) + self.eps)

    def get_state_dict(self):
        state_dict = super(AdaGrad, self).get_state_dict()
        state_dict["v"] = self.v
        return state_dict

    def load_state_dict(self, state_dict):
        super(AdaGrad, self).load_state_dict(state_dict)
        self.v = state_dict["v"]

class RMSprop(Optimizer): # Root Mean Square Prop

    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-08, l2_reg=0):
        super(RMSprop, self).__init__(params, lr)

        self.beta = beta
        self.eps = eps
        self.l2_reg = l2_reg
        self.v = {}
        for name, param in self.params.items():
            self.v[name] = np.zeros_like(param)
        
    def step(self):
        super(RMSprop, self).step()

        for name, param in self.params.items():

            # L2 Reg
            param.grad += self.l2_reg * param

            # Squared Momentum
            self.v[name] = self.beta * self.v[name] + (1 - self.beta) * param.grad**2

            # Update
            param -= self.lr * param.grad / (np.sqrt(self.v[name]) + self.eps)

    def get_state_dict(self):
        state_dict = super(RMSprop, self).get_state_dict()
        state_dict["v"] = self.v
        return state_dict

    def load_state_dict(self, state_dict):
        super(RMSprop, self).load_state_dict(state_dict)
        self.v = state_dict["v"]

class Adam(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, l2_reg=0):
        super(Adam, self).__init__(params, lr)

        self.betas = betas
        self.eps = eps
        self.l2_reg = l2_reg
        self.moments = {}
        for name, param in self.params.items():
            self.moments[name] = [np.zeros_like(param), np.zeros_like(param)]
        
    def step(self):
        super(Adam, self).step()

        for name, param in self.params.items():

            # L2 Reg
            param.grad += self.l2_reg * param

            # Momentum            
            self.moments[name][0] = self.betas[0] * self.moments[name][0] + (1 - self.betas[0]) * param.grad
            self.moments[name][1] = self.betas[1] * self.moments[name][1] + (1 - self.betas[1]) * param.grad**2
            m_hat = self.moments[name][0] / (1 - self.betas[0]**self.t)
            v_hat = self.moments[name][1] / (1 - self.betas[1]**self.t)

            # Update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_state_dict(self):
        state_dict = super(Adam, self).get_state_dict()
        state_dict["moments"] = self.moments
        return state_dict

    def load_state_dict(self, state_dict):
        super(Adam, self).load_state_dict(state_dict)
        self.moments = state_dict["moments"]

class AdamMax(Optimizer):

    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, l2_reg=0):
        super(AdamMax, self).__init__(params, lr)

        self.betas = betas
        self.eps = eps
        self.l2_reg = l2_reg
        self.moments = {}
        for name, param in self.params.items():
            self.moments[name] = [np.zeros_like(param), np.zeros_like(param)]
        
    def step(self):
        super(AdamMax, self).step()

        for name, param in self.params.items():

            # L2 Reg
            param.grad += self.l2_reg * param

            # Momentum
            self.moments[name][0] = self.betas[0] * self.moments[name][0] + (1 - self.betas[0]) * param.grad
            self.moments[name][1] = np.maximum(self.betas[1] * self.moments[name][1], np.abs(param.grad) + self.eps)
            m_hat = self.moments[name][0] / (1 - self.betas[0]**self.t)

            # Update
            param -= self.lr * m_hat / self.moments[name][1]

    def get_state_dict(self):
        state_dict = super(AdamMax, self).get_state_dict()
        state_dict["moments"] = self.moments
        return state_dict

    def load_state_dict(self, state_dict):
        super(AdamMax, self).load_state_dict(state_dict)
        self.moments = state_dict["moments"]

class AdamW(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        super(AdamW, self).__init__(params, lr)

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.moments = {}
        for name, param in self.params.items():
            self.moments[name] = [np.zeros_like(param), np.zeros_like(param)]
        
    def step(self):
        super(AdamW, self).step()

        for name, param in self.params.items():

            # Weight Decay
            if "weight" in name.split(".")[-1]:
                param -= self.lr * self.weight_decay * param

            # Momentum
            self.moments[name][0] = self.betas[0] * self.moments[name][0] + (1 - self.betas[0]) * param.grad
            self.moments[name][1] = self.betas[1] * self.moments[name][1] + (1 - self.betas[1]) * param.grad**2
            m_hat = self.moments[name][0] / (1 - self.betas[0]**self.t)
            v_hat = self.moments[name][1] / (1 - self.betas[1]**self.t)

            # Update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_state_dict(self):
        state_dict = super(AdamW, self).get_state_dict()
        state_dict["moments"] = self.moments
        return state_dict

    def load_state_dict(self, state_dict):
        super(AdamW, self).load_state_dict(state_dict)
        self.moments = state_dict["moments"]