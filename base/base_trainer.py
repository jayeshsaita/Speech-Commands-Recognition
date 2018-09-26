"""
base_trainer.py has one class BaseTrainer.

BaseTrainer provides base functionality for any trainer object.
It provides functionality to:
    - train for one epoch
    - validation 
    - testing
    - freezing model
    - unfreezing model
    - saving checkpoint
    - loading checkpoint
"""
import torch
from torchvision import models
import math
import matplotlib.pyplot as plt
import copy
import os
import datetime
from base import base_trainer


class BaseTrainer:
    """Base Class for trainer/train.py."""

    def __init__(self, model, train_dl, valid_dl, test_dl, criterion):
        """Initialize the BaseTrainer object."""
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.criterion = criterion
        self.opt_lrs = []

    def train_epoch(self, optimizer, scheduler):
        """
        Perform one training epoch.

        Parameters:
            optimizer - optimizer to use while training
            scheduler - scheduler to use while training
        Returns: training loss after epoch

        """
        self.model.train()

        final_loss = None

        for inputs, labels in self.train_dl:

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            loss = None

            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            self.opt_lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            scheduler.step()

            final_loss = float(loss)

            del(loss)

            torch.cuda.empty_cache()

        del(inputs)
        del(labels)
        del(outputs)

        return final_loss

    def valid_epoch(self):
        """
        Perform one validation epoch.

        Returns:
            val_loss - Validation loss 
            val_acc - Validation accuracy

        """
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():

            for val_inputs, val_labels in self.valid_dl:
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)

                val_loss = None

                val_outputs = self.model(val_inputs)
                val_preds, val_indices = torch.max(val_outputs, dim=1)

                val_loss = self.criterion(val_outputs, val_labels)

                # float(val_loss) => the float() is the most important thing
                # to avoid cuda out of memory error
                # https://pytorch.org/docs/stable/notes/faq.html#my-model-reports-cuda-runtime-error-2-out-of-memory
                running_loss += float(val_loss)
                running_corrects += torch.sum(val_indices == val_labels)

                del(val_loss)

                torch.cuda.empty_cache()

        del(val_inputs)
        del(val_labels)
        del(val_outputs)

        epoch_loss = running_loss / len(self.valid_dl.dataset)
        epoch_acc = running_corrects.double() / len(self.valid_dl.dataset)

        return [epoch_loss, epoch_acc]

    def test_epoch(self):
        """
        Perform a test epoch using current model.

        Returns accuracy on test set
        """
        self.model.eval()

        corrects = 0
        print('Performing Test Epoch')
        for test_inputs, test_labels in self.test_dl:
            test_inputs = test_inputs.to(self.device)
            test_labels = test_labels.to(self.device)

            test_output = self.model(test_inputs)
            test_output_labels = torch.max(test_output, dim=1)[1]
            corrects += int(torch.sum(test_output_labels == test_labels))

        del(test_inputs)
        del(test_labels)
        del(test_output)

        return corrects/len(self.test_dl.dataset)

    def unfreeze(self):
        """Unfreeze all layers of model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze(self):
        """Freeze all layers of model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def save_checkpoint(self, path, optimizer, scheduler, cycle, train_loss,
                        valid_loss, valid_acc):
        """
        Save checkpoint into given path.

        Parameters:
            path - path where checkpoint would be saved
            optimizer - optimizer's current state_dict to save
            scheduler - scheduler's current state_dict to save
            cycle - current cycle number (in SGDR) used for filename
            train_loss - training loss at the end of cycle 
            valid_loss - validation loss at the end of cycle
            valid_acc - validation accuracy at the end of cycle

        """
        state = {
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'cycle': cycle,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc
        }

        filename = f'cycle_{cycle}'
        torch.save(state, f'{path}/{filename}')

    def load_checkpoint(self, path):
        """
        Load checkpoint from given path.

        Returns state dictionary with keys:
            model - model's state dictionary
            optimizer - optimizer's state dictionary
            scheduler - scheduler's state dictionary
            cycle - cycle number when checkpoint was saved
            train_loss - training loss when checkpoint was saved
            valid_loss - validation loss when checkpoint was saved
            valid_acc - validation accuracy when checkpoint was saved

        """
        state = torch.load(path)
        return state
