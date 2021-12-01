import numpy as np
from collections import OrderedDict

# Functions
from nnet.functions import (
    conv2d,
    conv2dBackward,
    batchConv2dBackward
)

class Module:

    def __init__(self):

        self.parameters = OrderedDict()
        self.buffers = OrderedDict()
        self.modules = OrderedDict()
        self.training = True

    def __call__(self, *args, **kargs):

        if not "backward" in kargs:
            backward = False
        else:
            backward = kargs.pop("backward")

        if backward:
            return self.backward(*args, **kargs)
        else:
            return self.forward(*args, **kargs)

    def __setattr__(self, name, value):
        super(Module, self).__setattr__(name, value)

        if isinstance(value, Module) and name != "loss_function":
            self.modules[name] = value
        
    def forward(self, x, *args):

        for module in self.modules.values():
            x = module.forward(x, *args) 
        return x

    def backward(self, gradient=None):

        for module in reversed(self.modules.values()):
            gradient = module.backward(gradient)
        return gradient

    def add_module(self, module, prefix=""):
        self.__setattr__(prefix + module.__class__.__name__, module)

    def register_parameter(self, name, param):
        self.__setattr__(name, param)
        self.parameters[name] = param

    def register_buffer(self, name, buffer):
        self.__setattr__(name, buffer)
        self.buffers[name] = buffer

    def train(self, training=True):
        self.training = training
        for module in self.modules.values():
            module.train(training)

    def get_parameters(self, parameters=None, prefix=""):

        # Init State Dict
        if parameters is None:
            parameters = {}

        # Module Params
        for name, param in self.parameters.items():
            parameters[prefix + name] = param

        # Sub Modules
        for name, module in self.modules.items():
            module.get_parameters(parameters, prefix + name + ".")

        return parameters

    def get_buffers(self, buffers=None, prefix=""):

        # Init State Dict
        if buffers is None:
            buffers = {}

        # Module Buffers
        for name, buffer in self.buffers.items():
            buffers[prefix + name] = buffer

        # Sub Modules
        for name, module in self.modules.items():
            module.get_buffers(buffers, prefix + name + ".")

        return buffers

    def get_state_dict(self, state_dict=None, prefix=""):

        # Init State Dict
        if state_dict is None:
            state_dict = {}

        # Module Params
        for name, param in self.parameters.items():
            state_dict[prefix + name] = param

        # Module Buffers
        for name, buffer in self.buffers.items():
            state_dict[prefix + name] = buffer

        # Sub Modules
        for name, module in self.modules.items():
            module.get_state_dict(state_dict, prefix + name + ".")

        return state_dict

    def load_state_dict(self, state_dict, prefix=""):

        # Module Params
        for name, param in self.parameters.items():
            param.data = state_dict[prefix + name].data

        # Module Buffers
        for name, buffer in self.buffers.items():
            buffer.data = state_dict[prefix + name].data

        # Sub Modules
        for name, module in self.modules.items():
            module.load_state_dict(state_dict, prefix + name + ".")

    def get_num_parameters(self):

        return sum([param.size for param in self.parameters.values()] + [module.get_num_parameters() for module in self.modules.values()])

class ModuleList(Module):

    def __init__(self, modules):
        super(ModuleList, self).__init__()

        # Add Modules
        for i, module in enumerate(modules):
            self.add_module(module, prefix=str(i) + ".")

    def __iter__(self):
        return iter(self.modules.values())

class Tensor(np.ndarray):
    
    def __new__(cls, shape, dtype, *args, **kargs):
        return super().__new__(cls, shape, dtype) 

    def __init__(self, shape, dtype, init, in_features=None, out_features=None):

        # Assert
        assert init in ["zeros", "ones", "uniform", "normal", "scaled_uniform", "scaled_normal", "lecun_uniform", "lecun_normal", "xavier_uniform", "xavier_normal", "he_uniform", "he_normal"]

        # Init
        if init == "zeros":
            data = np.zeros(shape=shape)
        elif init == "ones":
            data = np.ones(shape=shape)
        elif init == "uniform":
            data = np.random.uniform(low=-1.0, high=1.0, size=shape)
        elif init == "normal":
            data = np.random.normal(loc=0.0, scale=1.0, size=shape)

        # Default
        elif init == "scaled_uniform":
            data = np.random.uniform(low=-1.0, high=1.0, size=shape) * np.sqrt(1 / in_features)
        elif init == "scaled_normal":
            data = np.random.normal(loc=0.0, scale=1.0, size=shape) * np.sqrt(1 / in_features)

        # LeCun
        elif init == "lecun_uniform":
            data = np.random.uniform(low=-1.0, high=1.0, size=shape) * np.sqrt(3 / in_features)
        elif init == "lecun_normal":
            data = np.random.normal(loc=0.0, scale=1.0, size=shape) * np.sqrt(1 / in_features)

        # Xavier
        elif init == "xavier_uniform":
            data = np.random.uniform(low=-1.0, high=1.0, size=shape) * np.sqrt(6 / (in_features + out_features))
        elif init == "xavier_normal":
            data = np.random.normal(loc=0.0, scale=1.0, size=shape) * np.sqrt(2 / (in_features + out_features))

        # He
        elif init == "he_uniform":
            data = np.random.uniform(low=-1.0, high=1.0, size=shape) * np.sqrt(6 / in_features)
        elif init == "he_normal":
            data = np.random.normal(loc=0.0, scale=1.0, size=shape) * np.sqrt(2 / in_features)

        # dtype
        self.data = data.astype(dtype)

        # Grad
        self.grad = None

class Identity(Module):

    def __init__(self):
        super(Identity, self).__init__()

class Linear(Module):

    def __init__(self, in_features, out_features, act=Identity, dtype=np.float64, weight_init="lecun_uniform", bias_init="zeros"):
        super(Linear, self).__init__()

        # Weights (Din, Dout)
        self.register_parameter("weight", Tensor(shape=(in_features, out_features), dtype=dtype, init=weight_init, in_features=in_features, out_features=out_features))

        # Bias (Dout)
        self.register_parameter("bias", Tensor(shape=(out_features), dtype=dtype, init=bias_init, in_features=in_features, out_features=out_features))

        # Activation function
        self.act = act()

    def forward(self, x):

        self.x = x

        # (B, ..., Din) -> (B, ..., Dout)
        return self.act(np.matmul(self.x, self.weight) + self.bias)

    def backward(self, gradient):

        # Activation function backward
        gradient = self.act.backward(gradient)

        # Weights Gradients (Din, Dout)
        self.weight.grad = np.tensordot(self.x, gradient, axes=[range(len(gradient.shape)-1)] * 2)

        # Bias Gradients (Dout)
        self.bias.grad = gradient.sum(axis=tuple(range(len(gradient.shape)-1)))

        # Input Gradients (B, ..., Din)
        return np.matmul(gradient, self.weight.transpose())

class Flatten(Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):

        self.x = x

        # (B, D1, D2, ...) -> (B, D1 * D2 * ...)
        return np.reshape(self.x, (self.x.shape[0], -1))

    def backward(self, gradients):

        # (B, D1 * D2 * ...) -> (B, D1, D2, ...)
        return np.reshape(gradients, self.x.shape)

class Reshape(Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = list(shape)

    def forward(self, x):

        self.x = x

        # (B, shape1) -> (B, shape2)
        return np.reshape(self.x, [self.x.shape[0]] + self.shape)

    def backward(self, gradients):

        # (B, shape2) -> (B, shape1)
        return np.reshape(gradients, self.x.shape)

class Transpose(Module):

    def __init__(self, axes):
        super(Transpose, self).__init__()

        self.axes = axes

    def forward(self, x):

        return x.transpose(self.axes)

    def backward(self, gradients):

        return gradients.transpose(self.axes)
    
class GlobalAvgPool1d(Module):

    def __init__(self, axis):
        super(GlobalAvgPool1d, self).__init__()

        self.axis = axis

    def forward(self, x):

        self.repeats = x.shape[self.axis]
        return x.mean(axis=self.axis)

    def backward(self, gradient):

        gradients = np.expand_dims(gradient, axis=self.axis)
        return gradients.repeat(self.repeats, axis=self.axis) / self.repeats

class Sigmoid(Module):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        self.x = x
        return 1 / (1 + np.exp(-self.x))

    def backward(self, gradient):

        return (np.exp(-self.x)) / ((1 + np.exp(-self.x))**2) * gradient

class Tanh(Module):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):

        self.tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.tanh

    def backward(self, gradient):

        return (1 - self.tanh**2) * gradient

class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):

        self.x = x
        return np.maximum(self.x, 0)

    def backward(self, gradient):

        return np.where(self.x > 0, gradient, 0)

class LeakyReLU(Module):

    def __init__(self, alpha=0.01):
        super(LeakyReLU, self).__init__()

        self.alpha = alpha

    def forward(self, x):

        self.x = x
        return np.where(self.x > 0, self.x, self.alpha * self.x)

    def backward(self, gradient):

        return np.where(self.x > 0, gradient, self.alpha * gradient)

class PReLU(Module):

    def __init__(self, in_features, dtype=np.float64):
        super(PReLU, self).__init__()

        self.register_parameter("alpha", Tensor(shape=(in_features,), dtype=dtype, init="zeros"))

    def forward(self, x):

        self.x = x
        return np.where(self.x > 0, self.x, self.alpha * self.x)

    def backward(self, gradient):

        self.alpha.grad = np.where(self.x > 0, 0, self.x * gradient).sum(axis=tuple(range(len(gradient.shape)-1)))

        return np.where(self.x > 0, gradient, self.alpha * gradient)

class ELU(Module):

    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def forward(self, x):

        self.x = x
        return np.where(self.x > 0, self.x, self.alpha * (np.exp(self.x) - 1))

    def backward(self, gradient):

        return np.where(self.x > 0, gradient, self.alpha * np.exp(self.x) * gradient)

class Swish(Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):

        self.x = x
        return self.x / (1 + np.exp(-self.x))

    def backward(self, gradient):

        return (1 + np.exp(-self.x) + self.x * np.exp(-self.x)) / ((1 + np.exp(-self.x))**2) * gradient

class Softmax(Module):

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):

        x_exp = np.exp(x - x.max())
        self.s = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
        return self.s

    def backward(self, gradient, jacobian=False):

        if jacobian:

            # Softmax Outer Product (B, ..., D, D)
            outer = np.einsum('i...j, i...k -> i...jk', self.s, self.s) 

            # Softmax Identity (B, ..., D, D)
            identity = np.einsum('i...j, jk -> i...jk', self.s, np.eye(self.s.shape[-1]))

            # Input Gradients
            return np.einsum('i...jk , i...k -> i...j', identity - outer, gradient)

        else:

            # Element Wise Product
            product = self.s * gradient

            # Input Gradients
            return product - self.s * np.sum(product, axis=-1, keepdims=True)

class BatchNorm(Module):

    def __init__(self, in_features, eps=1e-5, momentum=0.99, dtype=np.float64):
        super(BatchNorm, self).__init__()

        # Gamma (D)
        self.register_parameter("gamma", Tensor(shape=(in_features,), dtype=dtype, init="ones"))

        # Beta (D)
        self.register_parameter("beta", Tensor(shape=(in_features,), dtype=dtype, init="zeros"))

        # Moving Var (D)
        self.register_buffer("moving_var", Tensor(shape=(in_features,), dtype=dtype, init="ones"))

        # Moving Mean (D)
        self.register_buffer("moving_mean", Tensor(shape=(in_features,), dtype=dtype, init="zeros"))

        self.momentum = momentum
        self.eps = eps

    def forward(self, x):

        self.x = x

        if self.training:

            self.axis = tuple(range(len(self.x.shape)-1))
            self.m = np.prod(self.x.shape[:-1])

            # Compute mean / var
            self.x_mean = self.x.mean(axis=self.axis, keepdims=True)
            self.x_var = self.x.var(axis=self.axis, keepdims=True)

            # Update Moving Average
            self.moving_mean *= self.momentum
            self.moving_mean += (1 - self.momentum) * self.x_mean.reshape(-1)
            self.moving_var *= self.momentum
            self.moving_var += (1 - self.momentum) * self.x_var.reshape(-1)

            self.x_norm = (self.x - self.x_mean) / np.sqrt(self.x_var + self.eps)
        
        else:

            self.x_norm = (self.x - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
            
        return self.gamma * self.x_norm + self.beta

    def backward(self, gradient):

        # x norm grad
        x_norm_grad = self.gamma * gradient

        # x var grad
        x_var_grad = - np.sum(x_norm_grad * (self.x - self.x_mean) * 1/2 * np.power(self.x_var + self.eps, -3/2), axis=self.axis, keepdims=True)

        # x mean grad
        x_mean_grad = - np.sum(x_norm_grad / np.sqrt(self.x_var + self.eps), axis=self.axis, keepdims=True) - 2 * x_var_grad * np.mean(self.x - self.x_mean, axis=self.axis, keepdims=True)

        # Gamma grad
        self.gamma.grad = (self.x_norm * gradient).sum(axis=self.axis)

        # Beta grad
        self.beta.grad = gradient.sum(axis=self.axis)

        return x_norm_grad  / np.sqrt(self.x_var + self.eps) + x_var_grad * 2 * (self.x - self.x_mean) / self.m + x_mean_grad / self.m

class LayerNorm(Module):

    def __init__(self, norm_shape, eps=1e-5, dtype=np.float64):
        super(LayerNorm, self).__init__()

        # Normalized Shape
        if isinstance(norm_shape, int):
            self.norm_shape = (norm_shape,)
        else:
            self.norm_shape = norm_shape

        # Gamma (D)
        self.register_parameter("gamma", Tensor(shape=self.norm_shape, dtype=dtype, init="ones"))

        # Bata (D)
        self.register_parameter("beta", Tensor(shape=self.norm_shape, dtype=dtype, init="zeros"))

        self.eps = eps
        self.axis = tuple(range(-len(self.norm_shape), 0))
        self.m = np.prod(self.norm_shape)

    def forward(self, x):

        # Compute mean / var
        self.x = x
        self.x_mean = self.x.mean(axis=self.axis, keepdims=True)
        self.x_var = self.x.var(axis=self.axis, keepdims=True)

        self.x_norm = (self.x - self.x_mean) / np.sqrt(self.x_var + self.eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, gradient):

        # x norm grad
        x_norm_grad = self.gamma * gradient

        # x var grad
        x_var_grad = - np.sum(x_norm_grad * (self.x - self.x_mean) * 1/2 * np.power(self.x_var + self.eps, -3/2), axis=self.axis, keepdims=True)

        # x mean grad
        x_mean_grad = - np.sum(x_norm_grad / np.sqrt(self.x_var + self.eps), axis=self.axis, keepdims=True) - 2 * x_var_grad * np.mean(self.x - self.x_mean, axis=self.axis, keepdims=True)

        # Gamma grad
        axis_affine = tuple(range(len(gradient.shape)-len(self.norm_shape)))
        self.gamma.grad = (self.x_norm * gradient).sum(axis=axis_affine)

        # Beta grad
        self.beta.grad = gradient.sum(axis=axis_affine)

        return x_norm_grad  / np.sqrt(self.x_var + self.eps) + x_var_grad * 2 * (self.x - self.x_mean) / self.m + x_mean_grad / self.m

class Dropout(Module):

    def __init__(self, rate):
        super(Dropout, self).__init__()

        self.rate = rate

    def forward(self, x):

        if self.training:
            self.drop = np.random.uniform(0, 1, size=x.shape) > self.rate
            return (x * self.drop) / (1 - self.rate)
        else:
            return x 

    def backward(self, gradient):

        return gradient / (1 - self.rate) * self.drop 

class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, dtype=np.float64, embedding_init="normal"):
        super(Embedding, self).__init__()

        # Params
        self.num_embeddings = num_embeddings # V
        self.embedding_dim = embedding_dim # D

        # Embedding Weights (V, D)
        self.register_parameter("embeddings", Tensor(shape=(self.num_embeddings, self.embedding_dim), dtype=dtype, init=embedding_init, in_features=self.num_embeddings, out_features=self.embedding_dim))

        # Padding id
        self.padding_idx = padding_idx
        if self.padding_idx != None:
            self.embeddings[self.padding_idx] = 0

    def forward(self, x):

        # Input Shape
        x_shape = x.shape

        # Flatten (D1, ..., Dn) -> (D1 * ... * Dn)
        self.x_flatten = x.reshape(-1)

        # Embeddings (D1 * ... * Dn, D)
        emb = self.embeddings[self.x_flatten]

        # Reshape (D1 * ... * Dn, D) -> (D1, ..., Dn, D)
        emb = emb.reshape(x_shape + (self.embedding_dim,))

        return emb

    def backward(self, gradient):

        # Reshape (D1, ..., Dn, D) -> (D1 * ... * Dn, D)
        gradient = gradient.reshape((-1, self.embedding_dim))

        # Zeros Grad (V, D)
        self.embeddings.grad = np.zeros_like(self.embeddings)

        # Embedding Grad
        for v in np.unique(self.x_flatten):
            self.embeddings.grad[v] = gradient[np.equal(self.x_flatten, v)].sum(axis=0)

        # Padding id
        if self.padding_idx != None:
            self.embeddings.grad[self.padding_idx] = 0

class PatchEmbedding(Module):

    def __init__(self, in_height, in_width, in_dim, patch_size, embedding_dim, dtype=np.float64):
        super(PatchEmbedding, self).__init__()

        # Reshape (B, H, W, Din) -> (B, H // P, P, W // P, P, Din)
        self.reshape1 = Reshape(shape=(in_height // patch_size, patch_size, in_width // patch_size, patch_size, in_dim))

        # Transpose (B, H // P, P, W // P, P, Din) -> (B, H // P, W // P, P, P, Din)
        self.transpose = Transpose(axes=(0, 1, 3, 2, 4, 5))

        # Reshape (B, H // P, W // P, P, P, Din) -> (B, (H * W) // P**2, Din * P**2)
        self.reshape2 = Reshape(shape=((in_height * in_width) // (patch_size**2), in_dim * patch_size**2))

        # Linear Projection (B, N, Din * P**2) -> (B, N, D)
        self.linear = Linear(in_dim * patch_size**2, embedding_dim, act=Identity, dtype=dtype)

class Conv2d(Module):

    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), act=Identity, dtype=np.float64, weight_init="lecun_uniform", bias_init="zeros"):
        super(Conv2d, self).__init__()

        # Weights (Kh, Kw, Din, Dout)
        self.register_parameter("weight", Tensor(shape=(kernel_size[0], kernel_size[1], in_features, out_features), dtype=dtype, init=weight_init, in_features=in_features * kernel_size[0] * kernel_size[1], out_features=out_features))

        # Bias (Dout)
        self.register_parameter("bias", Tensor(shape=(out_features), dtype=dtype, init=bias_init, in_features=in_features, out_features=out_features))

        # Activation function
        self.act = act()

        # Padding
        self.padding = ((kernel_size[0] // 2, (kernel_size[0] - 1) // 2), (kernel_size[1] // 2, (kernel_size[1] - 1) // 2))
        self.padding_backward = (((kernel_size[1] - 1) // 2, kernel_size[1] // 2), ((kernel_size[0] - 1) // 2, kernel_size[0] // 2))

        # Conv Stride
        self.stride = stride

    def forward(self, x):

        self.x = x

        # (B, H, W, Din) -> (B, H, W, Dout)
        return self.act(conv2d(self.x, self.weight, stride=self.stride, padding=self.padding) + self.bias)

    def backward(self, gradient):

        # Activation function backward
        gradient = self.act.backward(gradient)

        # Weights Gradients (Kh, Kw, Din, Dout)
        self.weight.grad = batchConv2dBackward(self.x, gradient, stride=self.stride, padding=self.padding).sum(axis=0)

        # Bias Gradients (Dout)
        self.bias.grad = gradient.sum(axis=(0, 1, 2))

        # Upsample Gradient
        gradient_up = np.zeros(shape=self.x.shape[:-1] + gradient.shape[-1:], dtype=gradient.dtype)
        gradient_up[:, ::self.stride[0], ::self.stride[1]] = gradient

        # Input Gradients (B, H, W, Din)
        return conv2dBackward(gradient_up, self.weight.transpose(1, 0, 3, 2), padding=self.padding_backward)

class MultiHeadAttention(Module):

    def __init__(self, dim_model, num_heads, dtype=np.float64):
        super(MultiHeadAttention, self).__init__()

        # Assert
        assert dim_model % num_heads == 0

        # Attention Params
        self.dim_model = dim_model # D
        self.num_heads = num_heads # H
        self.dim_head = dim_model // num_heads # d

        # Linear Layers
        self.query_layer = Linear(self.dim_model, self.dim_model, dtype=dtype)
        self.key_layer = Linear(self.dim_model, self.dim_model, dtype=dtype)
        self.value_layer = Linear(self.dim_model, self.dim_model, dtype=dtype)
        self.output_layer = Linear(self.dim_model, self.dim_model, dtype=dtype)

        # Softmax Layer
        self.softmax = Softmax()

    def forward(self, Q, K, V, mask=None):

        # Batch size B
        batch_size = Q.shape[0]

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        self.Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)
        self.K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)
        self.V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)

        # Att scores (B, H, T, T)
        att_scores = np.matmul(self.Q, self.K.transpose(0, 1, 3, 2)) / np.sqrt(self.K.shape[-1])

        # Apply mask
        self.mask = mask
        if self.mask is not None:
            att_scores = np.where(self.mask, att_scores, -1e9)

        # Att weights (B, H, T, T)
        self.att_w = self.softmax(att_scores)

        # Att output (B, H, T, d)
        O = np.matmul(self.att_w, self.V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(0, 2, 1, 3).reshape(batch_size, -1,  self.dim_model)

        # Output Linear Layer
        O = self.output_layer(O)

        return O

    def backward(self, gradient):

        # Batch size B
        batch_size = gradient.shape[0]

        # Output Linear Layer grad
        gradient = self.output_layer(gradient, backward=True)

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        gradient = gradient.reshape(batch_size, -1, self.num_heads,  self.dim_head).transpose(0, 2, 1, 3)

        # Att weights grad (B, H, T, T)
        att_w_grad = np.matmul(gradient, self.V.transpose(0, 1, 3, 2))

        # Value grad (B, H, T, d)
        V_grad = np.matmul(self.att_w.transpose(0, 1, 3, 2), gradient)

        # Softmax Backward
        att_scores_grad = self.softmax(att_w_grad, backward=True)

        # Apply mask
        if self.mask is not None:
            att_scores_grad = np.where(self.mask, 0, att_scores_grad)

        # Q grad (B, H, T, d)
        Q_grad = np.matmul(att_scores_grad, self.K) / np.sqrt(self.K.shape[-1])

        # K grad (B, H, T, d)
        K_grad = np.matmul(self.Q.transpose(0, 1, 3, 2), att_scores_grad).transpose(0, 1, 3, 2) / np.sqrt(self.K.shape[-1])

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        Q_grad = Q_grad.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.dim_model)
        K_grad = K_grad.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.dim_model)
        V_grad = V_grad.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.dim_model)

        # Linear Layers grad
        Q_grad = self.query_layer(Q_grad, backward=True)
        K_grad = self.key_layer(K_grad, backward=True)
        V_grad = self.value_layer(V_grad, backward=True)

        return Q_grad, K_grad, V_grad