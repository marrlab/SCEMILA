import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import sys
import time
import copy
import numpy as np

class trainer:
    '''class containing all the info about the training process and handling the actual 
    training function'''

    def __init__(self, model, dataloaders, epochs, optimizer, scheduler, class_count, device, early_stop=20):
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

        best_loss = 10          #high value, so that future loss values will always be lower
        no_improvement_for = 0
        best_model = copy.deepcopy(self.model.state_dict())

        for ep in range(self.epochs):
            # perform train/val iteration
            loss, acc, conf_matrix, att = self.dataset_to_model(ep, 'train')
            loss, acc, conf_matrix, att = self.dataset_to_model(ep, 'val')
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
                if type(self.scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

        # load best performing model, and launch on test set
        self.model.load_state_dict(best_model)
        loss, acc, conf_matrix, att = self.dataset_to_model(ep, 'test')
        return self.model, conf_matrix, att


    def dataset_to_model(self, epoch, split, backprop_every=20):
        '''launch iteration for 1 epoch on specific dataset object, with backprop being optional
        - epoch: epoch count, is only printed and thus not super important
        - split: if equal to 'train', apply backpropagation. Otherwise, don`t.
        - backprop_every: only apply backpropagation every n patients. Allows for gradient accumulation over
          multiple patients, like in batch processing in a regular neural network.'''
        
        if(split == 'train'):
            backpropagation=True
            self.model.train()
        else:
            backpropagation = False
            self.model.eval()

        # initialize data structures to store results
        corrects = 0
        train_loss = 0.
        time_pre_epoch = time.time()
        confusion_matrix = np.zeros((self.class_count, self.class_count), np.int16)
        att_matrix = attention_matrix()

        self.optimizer.zero_grad()
        backprop_counter = 0

        for batch_idx, (bag, label, path_full) in enumerate(self.dataloaders[split]):

            # send to gpu
            label = label.to(self.device)
            bag = bag.to(self.device)

            # forward pass
            prediction, att_raw, att_softmax, bag_feature_stack = self.model(bag)

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
                # counter makes sure only every X samples backpropagation is excluded (resembling a training batch)
                if(backprop_counter%backprop_every == 0):
                    self.optimizer.step()
                    #print_grads(self.model)
                    self.optimizer.zero_grad()
            
            # transforms prediction tensor into index of position with highest value
            label_prediction = torch.argmax(prediction, dim=1).item()
            label_groundtruth = label[0].item()
            
            # store patient information for potential later analysis
            att_matrix.add_patient(label_groundtruth, path_full[0], att_raw, att_softmax, label_prediction, F.softmax(prediction, dim=1), loss_out, bag_feature_stack)

            # store predictions accordingly in confusion matrix
            if(label_prediction == label_groundtruth):
                corrects += 1
            confusion_matrix[label_groundtruth, label_prediction] += int(1)

            # print('----- loss: {:.3f}, gt: {} , pred: {}, prob: {}'.format(loss_out, label_groundtruth, label_prediction, prediction.detach().cpu().numpy()))

        samples = len(self.dataloaders[split])
        train_loss /= samples

        accuracy = corrects/samples

        print('- ep: {}/{}, loss: {:.3f}, acc: {:.3f}, {}s, {}'.format(
            epoch+1, self.epochs, train_loss.cpu().numpy(), 
            accuracy, int(time.time()-time_pre_epoch), split))

        return train_loss, accuracy, confusion_matrix, att_matrix.return_data()


class attention_matrix():
    '''stores all information about attention, predicted values, attention-pooled feature vectors, loss, ...
    Great to be pickelized later.'''
    
    def __init__(self):
        self.matrix = dict()

    def add_patient(self, entity, path_full, attention_raw, attention, prediction, prediction_vector, loss, out_features):
        '''stores all data including attention, predicted label, prediction probabilities, loss.
        Object is later saved in pickle.'''

        if not (entity in self.matrix):
            self.matrix[entity] = dict()
        self.matrix[entity][path_full] = (attention_raw.detach().cpu().numpy(), attention.detach().cpu().numpy(), prediction, 
                                    prediction_vector.data.cpu().numpy(), float(loss.data.cpu()), out_features.detach().cpu().numpy())

    def return_data(self):
        return self.matrix
