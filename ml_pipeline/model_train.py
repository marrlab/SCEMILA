import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import sys
import time
import copy
import numpy as np


class ModelTrainer:
    '''class containing all the info about the training process and handling the actual
    training function'''

    def __init__(
            self,
            model,
            dataloaders,
            epochs,
            optimizer,
            scheduler,
            class_count,
            device,
            early_stop=20):
        self.model = model
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_count = class_count
        self.early_stop = early_stop
        self.device = device

    def launch_training(self):
        '''initializes training process.'''

        best_loss = 10  # high value, so that future loss values will always be lower
        no_improvement_for = 0
        best_model = copy.deepcopy(self.model.state_dict())

        for ep in range(self.epochs):
            # perform train/val iteration
            loss, acc, conf_matrix, data_obj = self.dataset_to_model(
                ep, 'train')
            loss, acc, conf_matrix, data_obj = self.dataset_to_model(ep, 'val')
            no_improvement_for += 1

            loss = loss.cpu().numpy()

            # if improvement, reset counter
            if(loss < best_loss):
                best_model = copy.deepcopy(self.model.state_dict())
                best_loss = loss
                no_improvement_for = 0
                print("Best!")

            # break if X times no improvement
            if(no_improvement_for == self.early_stop):
                break

            # scheduler (optional)
            if not (self.scheduler is None):
                if isinstance(
                        self.scheduler,
                        optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

        # load best performing model, and launch on test set
        self.model.load_state_dict(best_model)
        loss, acc, conf_matrix, data_obj = self.dataset_to_model(ep, 'test')
        return self.model, conf_matrix, data_obj

    def dataset_to_model(self, epoch, split, backprop_every=20):
        '''launch iteration for 1 epoch on specific dataset object, with backprop being optional
        - epoch: epoch count, is only printed and thus not super important
        - split: if equal to 'train', apply backpropagation. Otherwise, don`t.
        - backprop_every: only apply backpropagation every n patients. Allows for gradient accumulation over
          multiple patients, like in batch processing in a regular neural network.'''

        if(split == 'train'):
            backpropagation = True
            self.model.train()
        else:
            backpropagation = False
            self.model.eval()

        # initialize data structures to store results
        corrects = 0
        train_loss = 0.
        time_pre_epoch = time.time()
        confusion_matrix = np.zeros(
            (self.class_count, self.class_count), np.int16)
        data_obj = DataMatrix()

        self.optimizer.zero_grad()
        backprop_counter = 0

        for batch_idx, (bag, label, path_full) in enumerate(
                self.dataloaders[split]):

            # send to gpu
            label = label.to(self.device)
            bag = bag.to(self.device)

            # forward pass
            prediction, att_raw, att_softmax, bag_feature_stack = self.model(
                bag)

            # calculate and store loss
            # loss_func = nn.BCELoss()
            # loss_out = loss_func(prediction, label)

            # pred_log = torch.log(prediction)
            loss_func = nn.CrossEntropyLoss()

            loss_out = loss_func(prediction, label[0])
            train_loss += loss_out.data

            # apply backpropagation if indicated
            if(backpropagation):
                loss_out.backward()
                backprop_counter += 1
                # counter makes sure only every X samples backpropagation is
                # excluded (resembling a training batch)
                if(backprop_counter % backprop_every == 0):
                    self.optimizer.step()
                    # print_grads(self.model)
                    self.optimizer.zero_grad()

            # transforms prediction tensor into index of position with highest
            # value
            label_prediction = torch.argmax(prediction, dim=1).item()
            label_groundtruth = label[0].item()

            # store patient information for potential later analysis
            data_obj.add_patient(
                label_groundtruth,
                path_full[0],
                att_raw,
                att_softmax,
                label_prediction,
                F.softmax(
                    prediction,
                    dim=1),
                loss_out,
                bag_feature_stack)

            # store predictions accordingly in confusion matrix
            if(label_prediction == label_groundtruth):
                corrects += 1
            confusion_matrix[label_groundtruth, label_prediction] += int(1)

            # print('----- loss: {:.3f}, gt: {} , pred: {}, prob: {}'.format(loss_out, label_groundtruth, label_prediction, prediction.detach().cpu().numpy()))

        samples = len(self.dataloaders[split])
        train_loss /= samples

        accuracy = corrects / samples

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, {}s, {}'.format(
            epoch + 1, self.epochs, train_loss.cpu().numpy(),
            accuracy, int(time.time() - time_pre_epoch), split))

        return train_loss, accuracy, confusion_matrix, data_obj.return_data()


class DataMatrix():
    '''DataMatrix contains all information about patient classification for later storage.
    Data is stored within a dictionary:

    self.data_dict[true entity] contains another dictionary with all patient paths for
                                the patients of one entity (e.g. AML-PML-RARA, SCD, ...)

    --> In this dictionary, the paths form the keys to all the data of that patient
        and it's classification, stored as a tuple:

        - attention_raw:    attention for all single cell images before softmax transform
        - attention:        attention after softmax transform
        - prediction:       Numeric position of predicted label in the prediction vector
        - prediction_vector:Prediction vector containing the softmax-transformed activations
                            of the last AMiL layer
        - loss:             Loss for that patients' classification
        - out_features:     Aggregated bag feature vectors after attention calculation and
                            softmax transform. '''

    def __init__(self):
        self.data_dict = dict()

    def add_patient(
            self,
            entity,
            path_full,
            attention_raw,
            attention,
            prediction,
            prediction_vector,
            loss,
            out_features):
        '''Add a new patient into the data dictionary. Enter all the data packed into a tuple into the dictionary as:
        self.data_dict[entity][path_full] = (attention_raw, attention, prediction, prediction_vector, loss, out_features)

        accepts:
        - entity: true patient label
        - path_full: path to patient folder
        - attention_raw: attention before softmax transform
        - attention: attention after softmax transform
        - prediction: numeric bag label
        - prediction_vector: output activations of AMiL model
        - loss: loss calculated from output actiations
        - out_features: bag features after attention calculation and matrix multiplication

        returns: Nothing
        '''

        if not (entity in self.data_dict):
            self.data_dict[entity] = dict()
        self.data_dict[entity][path_full] = (
            attention_raw.detach().cpu().numpy(),
            attention.detach().cpu().numpy(),
            prediction,
            prediction_vector.data.cpu().numpy()[0],
            float(
                loss.data.cpu()),
            out_features.detach().cpu().numpy())

    def return_data(self):
        return self.data_dict
