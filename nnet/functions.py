# Numpy
import numpy as np

def conv2d(x, w, stride=(1, 1), padding=((0, 0), (0, 0))):

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    
    # Padding
    x = np.pad(x, pad_width=((0, 0), padding_h, padding_w, (0, 0)))

    # Input (B, H, W, Di)
    B, H, W, _ = x.shape

    # Weight (Kh, Kw, Di, Do)
    Kh, Kw, _, Do = w.shape

    # Output (B, Ho, Wo, Do)
    y = np.zeros((B, (H - Kh) // stride_h  + 1, (W - Kw) // stride_w + 1, Do))

    # Output Loop
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):

            # Kernel loop
            
            for ki in range(Kh):
                for kj in range(Kw):

                    y[:, i, j] += np.matmul(x[:, i * stride_h + ki, j * stride_w + kj], w[ki, kj])
    return y

def conv2dBackward(x, w, stride=(1, 1), padding=((0, 0), (0, 0))):

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    
    # Padding
    x = np.pad(x, pad_width=((0, 0), padding_h, padding_w, (0, 0)))

    # Input (B, H, W, Di)
    B, H, W, _ = x.shape

    # Weight (Kh, Kw, Di, Do)
    Kh, Kw, _, Do = w.shape

    # Output (B, Ho, Wo, Do)
    y = np.zeros((B, H - Kh  + 1, W - Kw + 1, Do))

    # Output Loop
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):

            # Kernel loop
            for ki in range(Kh):
                for kj in range(Kw):
                    
                    if np.count_nonzero(x[:, i + ki, j + kj]):
                        y[:, i, j] += np.matmul(x[:, i + ki, j + kj], w[ki, kj])
    return y

def batchConv2dBackward(x, w, stride=(1, 1), padding=((0, 0), (0, 0))):

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    
    # Padding
    x = np.pad(x, pad_width=((0, 0), padding_h, padding_w, (0, 0)))

    # Input (B, H, W, Di)
    B, H, W, Di = x.shape

    # Weight (B, Kh, Kw, Do)
    B, Kh, Kw, Do = w.shape

    # Output (B, Ho, Wo, Di, Do)
    y = np.zeros((B, H - Kh * stride_h + 1, W - Kw * stride_w + 1, Di, Do))

    # Output Loop
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):

            # Kernel Loop
            for ki in range(Kh):
                for kj in range(Kw):

                    y[:, i, j] += np.einsum("bi, bo -> bio", x[:, i + stride_h * ki, j + stride_w * kj], w[:, ki, kj])
    return y