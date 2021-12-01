from nnet.modules import Module

import numpy as np
import pickle
import os
from tqdm import tqdm


class Model(Module):

    def __init__(self, loss_function):
        super(Model, self).__init__()

        # Loss Function
        self.loss_function = loss_function()

        # Training
        self.train()

    def backward(self, gradient=None):
        return super(Model, self).backward(self.loss_function.backward())

    def summary(self):

        # Number Params
        print(self.get_num_parameters(), "parameters")

        # Params
        state_dict = self.get_state_dict()
        max_len = max([len(key) for key in state_dict.keys()])
        for key, param in state_dict.items():
            print("{:<40} shape {:<16} mean {:<12.4f} std {:<12.4f} dtype {:<12}".format(key + " " * (max_len - len(key)), str(param.shape), param.mean(), param.std(), str(param.dtype)))

    def save(self, save_path, save_optimizer=True):

        # Save model checkpoint
        pickle.dump(
            {
                "model_state_dict": self.get_state_dict(),
                "optimizer_state_dict": self.optimizer.get_state_dict() if save_optimizer else None
            }, 
            open(save_path, "bw"),
            protocol=pickle.HIGHEST_PROTOCOL
        )

        # Print
        print("Model saved at step {}".format(self.optimizer.t))

    def load(self, load_path):

        # Load model checkpoint
        try:
            checkpoint = pickle.load(open(load_path, "br"))
        except:
            print("Couldn't load model: {}".format(load_path))
            return

        # Model State Dict
        self.load_state_dict(checkpoint["model_state_dict"])

        # Optimizer State Dict
        if checkpoint["optimizer_state_dict"] != None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Print
        print("Model loaded at step {}".format(self.optimizer.t))

    def fit(self, dataset_train, epochs, dataset_val=None, initial_epoch=0, callback_path=None):

        # Callbacks
        if callback_path is not None:

            # Callback Dir
            if not os.path.isdir(callback_path):
                os.makedirs(callback_path)

        # Extract Dataset
        x_train, y_train = dataset_train

        # Training Loop
        for epoch in range(initial_epoch, epochs):

            # Shuffle Training Set
            shuffle = np.arange(len(x_train))
            np.random.shuffle(shuffle)
            x_train = x_train[shuffle]
            y_train = y_train[shuffle]

            # Epoch Init
            print("Epoch {}/{}:".format(epoch + 1, epochs))
            epoch_loss = 0
            epoch_acc = 0
            iterator_train = tqdm(zip(x_train, y_train), total=len(x_train), dynamic_ncols=True)

            # Training Mode
            self.train()

            # Epoch training
            for step, batch in enumerate(iterator_train):

                # Batch
                x, y = batch

                # Forward 
                y_pred = self.forward(x)

                # Compute Loss
                batch_loss = self.loss_function(y, y_pred)

                # Update Epoch Loss
                epoch_loss += batch_loss

                # Compute Accuracy
                batch_acc = np.mean(y.argmax(axis=-1)==y_pred.argmax(axis=-1))

                # Update Accuracy
                epoch_acc += batch_acc

                # Backward
                self.backward()

                # Update Scheduler
                self.scheduler.step()

                # Update Params
                self.optimizer.step()

                # Zero Gradients
                self.optimizer.zero_grad()

                # Step Print
                iterator_train.set_description("mean loss: {:.4f} - batch loss: {:.4f} - mean acc: {:.2f} - batch acc: {:.2f} - lr: {:.6f} - step: {}".format(epoch_loss / (step + 1), batch_loss, 100 * epoch_acc / (step + 1), 100 * batch_acc, self.optimizer.lr, self.optimizer.t))

            # Validation
            if dataset_val is not None:
                self.evaluate(dataset_val)

            # Saving Checkpoint
            if callback_path != None:
                self.save(os.path.join(callback_path, "checkpoints_" + str(epoch + 1) + ".ckpt"))
            print()

    def evaluate(self, dataset_val):

        # Extract Dataset
        x_val, y_val = dataset_val

        # Init
        iterator_val = tqdm(zip(x_val, y_val), total=len(x_val), dynamic_ncols=True)
        val_acc = 0
        val_loss = 0

        # Eval Mode
        self.train(False)

        # Validation Loop
        for step, batch in enumerate(iterator_val):

            # Batch
            x, y = batch

            # Forward
            y_pred = self.forward(x)

            # Compute Loss
            batch_loss = self.loss_function(y, y_pred)

            # Update loss
            val_loss += batch_loss

            # Compute Accuracy
            batch_acc = np.mean(y.argmax(axis=-1)==y_pred.argmax(axis=-1))

            # Update Accuracy
            val_acc += batch_acc

            # Step Print
            iterator_val.set_description("mean loss: {:.4f} - batch loss: {:.4f} - mean acc: {:.2f} - batch acc: {:.2f}".format(val_loss / (step + 1), batch_loss, 100 * val_acc / (step + 1), 100 * batch_acc))

        # Print Accuracy
        print("validation loss: {:.4f}".format(val_loss / (len(x_val))))
        print("validation accuracy: {:.2f}%".format(100 * val_acc / len(x_val)))