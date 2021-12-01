from nnet.modules import Flatten, PatchEmbedding, Embedding, ModuleList
from nnet.blocks import MLP, MLPMixer, CNN, TransformerBlock
from nnet.model import Model
import numpy as np


class MLPModel(Model):

    def __init__(self, in_height, in_width, in_dim, dim_layers, hidden_function, out_function, norm, drop_rate, loss_function, dtype):
        super(MLPModel, self).__init__(loss_function)

        # Flatten Layer
        self.flatten = Flatten()

        # MLP block
        self.mlp = MLP(
            dim_input=in_height * in_width * in_dim,
            dim_layers=dim_layers,
            hidden_function=hidden_function,
            out_function=out_function,
            norm=norm,
            drop_rate=drop_rate,
            dtype=dtype
        )

class CNNModel(Model):

    def __init__(self, in_height, in_width, in_dim, dim_cnn_layers, kernel_size, strides, dim_mlp_layers, hidden_function, out_function, norm, drop_rate, loss_function, dtype):
        super(CNNModel, self).__init__(loss_function)

        # Conv Layers
        self.cnn = CNN(
            dim_input=in_dim,
            dim_layers=dim_cnn_layers,
            kernel_size = kernel_size,
            strides = strides,
            hidden_function=hidden_function,
            norm=norm,
            drop_rate=drop_rate,
            dtype=dtype
        )

        # Flatten Layer
        self.flatten = Flatten()

        # MLP block
        self.mlp = MLP(
            dim_input=in_height * in_width * dim_cnn_layers[-1] // np.prod(strides),
            dim_layers=dim_mlp_layers,
            hidden_function=hidden_function,
            out_function=out_function,
            norm=norm,
            drop_rate=drop_rate,
            dtype=dtype
        )

class MLPMixerModel(Model):

    def __init__(self, in_height, in_width, in_dim, patch_size, num_layers, dim_feat, dim_expand_feat, dim_expand_seq, out_layers, out_norm, hidden_function, out_function, drop_rate, loss_function, dtype):
        super(MLPMixerModel, self).__init__(loss_function)

        # Patch Embedding
        self.embedding = PatchEmbedding(in_height, in_width, in_dim, patch_size, dim_feat, dtype=dtype)

        # MLP Mixer Layers
        self.layers = ModuleList([MLPMixer(
            dim_seq=(in_height * in_width) // (patch_size**2),
            dim_expand_seq=dim_expand_seq, 
            dim_feat=dim_feat, 
            dim_expand_feat=dim_expand_feat, 
            act=hidden_function, 
            drop_rate=drop_rate,
            dtype=dtype
        ) for layer_id in range(num_layers)])

        # Flatten Layer
        self.flatten = Flatten()

        # MLP block
        self.mlp_out = MLP(
            dim_input=dim_feat * (in_height * in_width) // (patch_size**2),
            dim_layers=out_layers,
            hidden_function=hidden_function,
            out_function=out_function,
            norm=out_norm,
            drop_rate=drop_rate,
            dtype=dtype
        )

class VisionTransformer(Model):

    def __init__(self, in_height, in_width, in_dim, patch_size, num_blocks, dim_model, ff_ratio, num_heads, drop_rate, dim_mlp_layers, mlp_norm, hidden_function, out_function, loss_function, dtype):
        super(VisionTransformer, self).__init__(loss_function)

        # Patch Embedding
        self.embedding = PatchEmbedding(in_height, in_width, in_dim, patch_size, dim_model, dtype=dtype)

        # Positional Encodings
        self.pos_encoding = Embedding(num_embeddings=(in_height * in_width) // (patch_size**2), embedding_dim=dim_model, dtype=dtype)

        # Transformer Blocks
        self.blocks = ModuleList([TransformerBlock(
            dim_model=dim_model, 
            ff_ratio=ff_ratio, 
            num_heads=num_heads, 
            drop_rate=drop_rate, 
            act=hidden_function, 
            dtype=dtype
        ) for block_id in range(num_blocks)])
        
        # Flatten Layer
        self.flatten = Flatten()

        # MLP block
        self.mlp = MLP(
            dim_input=dim_model * (in_height * in_width) // (patch_size**2),
            dim_layers=dim_mlp_layers,
            hidden_function=hidden_function,
            out_function=out_function,
            norm=mlp_norm,
            drop_rate=drop_rate,
            dtype=dtype
        )

    def forward(self, x):

        # Patch Embedding
        x = self.embedding(x)

        # Positional Encodings
        x += self.pos_encoding(np.arange(x.shape[1]))

        # Blocks
        x = self.blocks(x)

        # Flatten
        x = self.flatten(x)

        # MLP
        x = self.mlp(x)

        return x

    def backward(self, gradient=None):

        # Loss Function
        gradient = self.loss_function(backward=True)

        # MLP
        gradient = self.mlp(gradient, backward=True)

        # Flatten
        gradient = self.flatten(gradient, backward=True)

        # Blocks
        gradient = self.blocks(gradient, backward=True)

        # Positional Encodings
        self.pos_encoding(gradient.sum(axis=0), backward=True)

        # Patch Embedding
        self.embedding(gradient, backward=True)