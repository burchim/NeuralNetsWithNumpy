from nnet.modules import Module, ModuleList, Linear, Identity, Dropout, PReLU, Transpose, LayerNorm, Conv2d, MultiHeadAttention

class MLP(Module):

    def __init__(self, dim_input, dim_layers, hidden_function, out_function, norm, drop_rate, dtype):
        super(MLP, self).__init__()

        # Fully Connected Layers
        for layer_id in range(len(dim_layers)):

            # Linear
            self.add_module(Linear(
                in_features=dim_input if layer_id==0 else dim_layers[layer_id-1], 
                out_features=dim_layers[layer_id], 
                act=Identity,
                dtype=dtype
            ), prefix=str(layer_id) + ".")

            # Norm Layer
            if layer_id<len(dim_layers)-1 and norm != Identity:
                self.add_module(norm(dim_layers[layer_id], dtype=dtype), prefix=str(layer_id) + ".")

            # Act Function
            if layer_id<len(dim_layers)-1:
                if hidden_function == PReLU:
                    self.add_module(hidden_function(dim_layers[layer_id], dtype), prefix=str(layer_id) + ".")
                else:
                    self.add_module(hidden_function(), prefix=str(layer_id) + ".")
            else:
                self.add_module(out_function(), prefix=str(layer_id) + ".")

            # Dropout
            if layer_id<len(dim_layers)-1 and drop_rate > 0:
                self.add_module(Dropout(drop_rate), prefix=str(layer_id) + ".")

class CNN(Module):

    def __init__(self, dim_input, dim_layers, kernel_size, strides, hidden_function, norm, drop_rate, dtype):
        super(CNN, self).__init__()

        # CNN Layers
        for layer_id in range(len(dim_layers)):

            # Conv2d
            self.add_module(Conv2d(
                in_features=dim_input if layer_id==0 else dim_layers[layer_id-1], 
                out_features=dim_layers[layer_id], 
                kernel_size=kernel_size,
                stride=strides[layer_id],
                act=Identity,
                dtype=dtype
            ), prefix=str(layer_id) + ".")

            # Norm Layer
            if norm != Identity:
                self.add_module(norm(dim_layers[layer_id], dtype=dtype), prefix=str(layer_id) + ".")

            # Act Function
            if hidden_function == PReLU:
                self.add_module(hidden_function(dim_layers[layer_id], dtype), prefix=str(layer_id) + ".")
            else:
                self.add_module(hidden_function(), prefix=str(layer_id) + ".")

            # Dropout
            if drop_rate > 0:
                self.add_module(Dropout(drop_rate), prefix=str(layer_id) + ".")

class MLPMixer(Module):

    def __init__(self, dim_seq, dim_expand_seq, dim_feat, dim_expand_feat, act, drop_rate, dtype):
        super(MLPMixer, self).__init__()

        # Layer Norm 1
        self.layernorm1 = LayerNorm(dim_feat, dtype=dtype)

        # Transpose 1
        self.transpose1 = Transpose(axes=(0, 2, 1))

        # MLP 1
        self.mlp1 = MLP(
            dim_input=dim_seq,
            dim_layers=[dim_expand_seq, dim_seq],
            hidden_function=act,
            out_function=Identity,
            norm=Identity,
            drop_rate=0,
            dtype=dtype
        )

        # Transpose 2
        self.transpose2 = Transpose(axes=(0, 2, 1))

        # Dropout 1
        if drop_rate > 0:
            self.dropout1 = Dropout(drop_rate)
        else:
            self.dropout1 = Identity()

        # Layer Norm 2
        self.layernorm2 = LayerNorm(dim_feat, dtype=dtype)

        # MLP 2
        self.mlp2 = MLP(
            dim_input=dim_feat,
            dim_layers=[dim_expand_feat, dim_feat],
            hidden_function=act,
            out_function=Identity,
            norm=Identity,
            drop_rate=0,
            dtype=dtype
        )

        # Dropout 2
        if drop_rate > 0:
            self.dropout2 = Dropout(drop_rate)
        else:
            self.dropout2 = Identity()

    def forward(self, x):

        # Module 1
        residual = x
        x = self.layernorm1(x)
        x = self.transpose1(x)
        x = self.mlp1(x)
        x = self.transpose2(x)
        x = self.dropout1(x)
        x += residual

        # Module 2
        residual = x
        x = self.layernorm2(x)
        x = self.mlp2(x)
        x = self.dropout2(x)
        x += residual

        return x

    def backward(self, gradient):

        # Module 2
        residual = gradient
        gradient = self.dropout2(gradient, backward=True)
        gradient = self.mlp2(gradient, backward=True)
        gradient = self.layernorm2(gradient, backward=True)
        gradient += residual

        # Module 1
        residual = gradient
        gradient = self.dropout1(gradient, backward=True)
        gradient = self.transpose2(gradient, backward=True)
        gradient = self.mlp1(gradient, backward=True)
        gradient = self.transpose1(gradient, backward=True)
        gradient = self.layernorm1(gradient, backward=True)
        gradient += residual

        return gradient

class TransformerBlock(Module):

    def __init__(self, dim_model, ff_ratio, num_heads, drop_rate, act, dtype):
        super(TransformerBlock, self).__init__()

        # Muti-Head Self-Attention Module
        self.mhsa_module = MultiHeadSelfAttentionModule(
            dim_model=dim_model,
            num_heads=num_heads,
            drop_rate=drop_rate,
            dtype=dtype
        )

        # Feed Forward Module
        self.ff_module = FeedForwardModule(
            dim_model=dim_model,
            dim_ffn=dim_model * ff_ratio,
            drop_rate=drop_rate,
            act=act,
            dtype=dtype
        )

    def forward(self, x, mask=None):

        # Muti-Head Self-Attention Module
        x = x + self.mhsa_module(x, mask)

        # Feed Forward Module
        x = x + self.ff_module(x)

        return x

    def backward(self, gradient):

        # Feed Forward Module
        gradient = gradient + self.ff_module(gradient, backward=True)

        # Muti-Head Self-Attention Module
        gradient = gradient + self.mhsa_module(gradient, backward=True)

        return gradient

class MultiHeadSelfAttentionModule(Module):

    def __init__(self, dim_model, num_heads, drop_rate, dtype):
        super(MultiHeadSelfAttentionModule, self).__init__()

        # Layer Norm
        self.layernorm = LayerNorm(dim_model, dtype=dtype)

        # MHSA
        self.mhsa = MultiHeadAttention(dim_model, num_heads, dtype=dtype)

        # Dropout
        self.dropout = Dropout(drop_rate)

    def forward(self, x, mask=None):

        # Layer Norm
        x = self.layernorm(x)

        # MHSA
        x = self.mhsa(x, x, x, mask)

        # Dropout
        x = self.dropout(x)

        return x

    def backward(self, gradient):

        # Dropout
        gradient = self.dropout(gradient, backward=True)

        # MHSA
        Q_grad, K_grad, V_grad = self.mhsa(gradient, backward=True)
        gradient = Q_grad + K_grad + V_grad

        # Layer Norm
        gradient = self.layernorm(gradient, backward=True)

        return gradient

class FeedForwardModule(Module):

    def __init__(self, dim_model, dim_ffn, drop_rate, act, dtype, inner_dropout=False):
        super(FeedForwardModule, self).__init__()

        # Layers
        self.layers = ModuleList([
            LayerNorm(dim_model, dtype=dtype),
            Linear(dim_model, dim_ffn, act=act, dtype=dtype),
            Dropout(drop_rate) if inner_dropout else Identity(),
            Linear(dim_ffn, dim_model, dtype=dtype),
            Dropout(drop_rate)
        ])