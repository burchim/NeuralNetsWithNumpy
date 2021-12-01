# Numpy
import numpy as np

class Scheduler:

    def __init__(self, optimizer):

        # Model Optimizer
        self.optimizer = optimizer

    def step(self):
        pass

class ConstantScheduler(Scheduler):

    def __init__(self, optimizer, lr_value):
        super(ConstantScheduler, self).__init__(optimizer)

        # Scheduler Params
        self.lr_value = lr_value

    def step(self):

        # Update LR
        self.optimizer.lr = self.lr_value

class ConstantDecayScheduler(Scheduler):

    def __init__(self, optimizer, lr_values, decay_steps):
        super(ConstantDecayScheduler, self).__init__(optimizer)

        # Scheduler Params
        self.lr_values = lr_values
        self.decay_steps = decay_steps

    def step(self):

        # Update LR
        lr_value = self.lr_values[0]
        for i, step in enumerate(self.decay_steps):
            if self.optimizer.t >= step:
                lr_value = self.lr_values[i]
            else:
                break
        self.optimizer.lr = lr_value

class WarmupNoamDecayScheduler(Scheduler):

    def __init__(self, optimizer, warmup_steps, dim_decay, lr_factor):
        super(WarmupNoamDecayScheduler, self).__init__(optimizer)

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.dim_decay = dim_decay
        self.lr_factor = lr_factor

    def step(self):

        # Update LR
        step = self.optimizer.t + 1
        arg1 = step * (self.warmup_steps**-1.5) # Warmup phase
        arg2 = step**-0.5 # Decay phase
        self.optimizer.lr = self.lr_factor * self.dim_decay**-0.5 * min(arg1, arg2)

class WarmupExpDecayScheduler(Scheduler):

    def __init__(self, optimizer, warmup_steps, lr_max, alpha, end_step):
        super(WarmupExpDecayScheduler, self).__init__(optimizer)

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.alpha = alpha
        self.end_step = end_step

    def step(self):

        # Update LR
        step = self.optimizer.t + 1
        arg1 = step / self.warmup_steps * self.lr_max # Warmup phase
        arg2 = self.lr_max * self.alpha**((step - self.warmup_steps) / (self.end_step - self.warmup_steps)) # Decay phase
        self.optimizer.lr = min(arg1, arg2)

class WarmupCosineAnnealingScheduler(Scheduler):

    def __init__(self, optimizer, warmup_steps, lr_max, lr_min, end_step):
        super(WarmupCosineAnnealingScheduler, self).__init__(optimizer)

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.end_step = end_step

    def step(self):

        # Update LR
        step = self.optimizer.t + 1
        if step <= self.warmup_steps: # Warmup phase
            self.optimizer.lr = step / self.warmup_steps * self.lr_max
        elif step <= self.end_step: # Annealing phase
            self.optimizer.lr = (self.lr_max - self.lr_min) * 0.5 * (1 + np.cos(np.pi * (step - self.warmup_steps) / (self.end_step - self.warmup_steps))) + self.lr_min
        else: # End phase
            self.optimizer.lr = self.lr_min