"""
train.py module has one class ModelTrainer.

ModelTrainer extends functionality of BaseTrainer by providing
    - Training using SGDR
    - Testing using Snapshot ensembling
"""
import torch
from torchvision import models
import math
import matplotlib.pyplot as plt
import copy
import os
import datetime
from base import base_trainer
from IPython.display import FileLink


class ModelTrainer(base_trainer.BaseTrainer):
    """ModelTrainer helps to do transfer learning on given model."""

    def __init__(self, model, train_dl, valid_dl, test_dl,
                 criterion, freeze=True):
        """
        Initialize the ModelTrainer Object.

        Parameters:
            model - model to initialize the object
            train_dl - training dataloader
            valid_dl - validation dataloader
            test_dl - test dataloader
            criterion - criteria to use for measuring accuracy
            freeze - freeze all layers of model except fc layer
                     (default True)

        """
        super().__init__(model, train_dl, valid_dl, test_dl, criterion)  

        if freeze:
            self.freeze()

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(train_dl.dataset.classes))

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        if self.device == 'cuda:0':
            print('Model moved to GPU')
        else:
            print('Model is on CPU')

    def dummy_sgdr(self, learning_rate, n_cycles, cycle_len=1, cycle_mult=1):
        """Demonstrate learning rate annealing in SGDR."""
        opt_lrs = []
        for curr_cycle in range(n_cycles):
            n_epochs = cycle_len * (cycle_mult ** curr_cycle)
            temp_optimizer = torch.optim.SGD(params=[torch.rand(5, 5),
                                             torch.rand(5,2)], lr=learning_rate)

            temp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                             optimizer=temp_optimizer,
                             T_max=n_epochs*len(self.train_dl))
            temp_scheduler.step()

            for curr_epoch in range(n_epochs):
                for _ in range(len(self.train_dl)):
                    opt_lrs.append(temp_optimizer.state_dict()
                                   ['param_groups'][0]['lr'])
                    temp_optimizer.step()
                    temp_scheduler.step()

        plt.plot(opt_lrs)
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')

    def fit(self, n_cycles, cycle_len=1, cycle_mult=1, learning_rate=None,
            optimizer=None, save_snapshot=False, snapshot_dir='saved/'):
        """
        Train the model using SGDR.

        Parameters:
            n_cycles - number of cycles for SGDR
            cycle_len - initial length of cycle (in epochs) for SGDR
            cycle_mult - multiplier to increase cycle length after every cycle
            learning_rate - learning rate to start cosine annealing from
            optimizer - optimizer to use for training
            save_snapshot - boolean which allows to save snapshot
                            after every cycle (default False)
            snapshot_dir - directory in which to save snapshots
                           (default saved/)

        """
        if save_snapshot:
            curr_data = datetime.datetime.now()
            curr_date = datetime.datetime.date(curr_data)
            curr_time = datetime.datetime.time(curr_data)
            dir_name = f'{curr_date}_{str(curr_time)[:8]}'
            os.makedirs(f'{snapshot_dir}{dir_name}')

        self.opt_lrs = []

        for curr_cycle in range(n_cycles):
            print(f'Cycle:{curr_cycle+1}')
            n_epochs = cycle_len * (cycle_mult ** curr_cycle)

            if optimizer:
                curr_optimizer = copy.deepcopy(optimizer)
            else:
                curr_optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 lr=learning_rate, momentum=0.9)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=curr_optimizer,
                        T_max=n_epochs*len(self.train_dl))
            scheduler.step()

            for curr_epoch in range(n_epochs):
                print(f'\tEpoch:{curr_epoch+1}/{n_epochs}')
                train_loss = self.train_epoch(optimizer=curr_optimizer, 
                                              scheduler=scheduler)
                print(f'\tTrain Loss:{train_loss}')
                valid_loss, valid_acc = self.valid_epoch()
                print(f'\tValid Loss:{valid_loss} Valid Accuracy:{valid_acc}')
                print('\t' + '-'*50)

            if save_snapshot:
                self.save_checkpoint(path=f'{snapshot_dir}{dir_name}',
                                     optimizer=curr_optimizer,
                                     scheduler=scheduler, cycle=curr_cycle,
                                     train_loss=train_loss,
                                     valid_loss=valid_loss, valid_acc=valid_acc)
                print('\tSnapshot saved!')

    def test_batch(self, input):
        """
        Perform test on given batch.

        Returns predicted label for current batch.

        """
        input = input.to(self.device)
        output = self.model(input)
        output_labels = torch.max(output, dim=1)[1]

        return output_labels

    def predict(self, path, n_models):
        """
        Use models from saved snapshots for prediction.

        Parameters:
            path - path to directory containing checkpoint files
            n_models - number of models to use for prediction
                       (top n_models will be used,
                       models sorted based on validation accuracy)

        """
        models = []
        checkpoints = os.listdir(path)
        for curr in checkpoints:
            state = self.load_checkpoint(f'{path}/{curr}')
            model = state['model']
            valid_acc = state['valid_acc']
            models.append((model, valid_acc))

        models.sort(key=lambda x: x[1], reverse=True)

        models = models[:n_models]

        corrects = 0

        for iteration, (test_input, test_label) in enumerate(self.test_dl, 1):
            test_output_label = []
            for curr_model, valid_acc in models:
                self.model.load_state_dict(curr_model)
                self.model.to(self.device)
                test_output_label.append((self.test_batch(test_input)))
            test_output_label = torch.tensor([x.cpu().numpy()
                                             for x in test_output_label])
            final_votes = torch.mode(test_output_label, dim=0)[0]
            corrects += torch.sum(final_votes == test_label)

        return corrects.item()/len(self.test_dl.dataset)