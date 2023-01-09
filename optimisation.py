# model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# learning and optimisation helper libraries
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.callbacks import Callback
from hyperopt import fmin, tpe, Trials, hp, rand
from hyperopt.pyll.stochastic import sample

import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

def define_model(lr, n_steps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(n_features))
    model.add(Dense(n_features))
    
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model

def evaluate_model(trainX, trainY, testX, testY, max_epochs, learning_rate, momentum, batch_size, model=None, callbacks=[]):
    if model == None:
        model = define_model(learning_rate, momentum)
    history = model.fit(trainX, trainY, epochs=max_epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0, callbacks = callbacks)
    return model, history

def evaluate_model_cross_validation(trainX, trainY, max_epochs, learning_rate, momentum, batch_size, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for trainFold_ix, testFold_ix in kfold.split(trainX):
        # select rows for train and test
        trainFoldsX, trainFoldsY, testFoldX, testFoldY = trainX[trainFold_ix], trainY[trainFold_ix], trainX[testFold_ix], trainY[testFold_ix]
        # fit model
        model = define_model(learning_rate, momentum)
        history = model.fit(trainFoldsX, trainFoldsY, epochs=max_epochs, batch_size=batch_size, validation_data=(testFoldX, testFoldY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testFoldX, testFoldY, verbose=0)
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories

#### VISUALISATION OF TRAINING OUTCOME

def summarize_diagnostics(histories):
    fig, (ax1, ax2) = plt.subplots(2,1)
    for i in range(len(histories)):
        # plot loss
        ax1.set_title('Cross Entropy Loss')
        ax1.plot(histories[i]['loss'], color='blue', label='train_loss')
        ax1.plot(histories[i]['val_loss'], color='orange', label='val_loss')
        # plot accuracy
        ax2.set_title('Accuracy')
        ax2.plot(histories[i]['accuracy'], color='blue', label='train_acc')
        ax2.plot(histories[i]['val_accuracy'], color='orange', label='val_acc')
    fig.canvas.draw()

### Custom callback

class PlotProgress(Callback):
    def on_train_begin(self, logs={}):
        self.fig, self.ax1, self.ax2 = self.prepare_plot()
        self.loss_history = list()
        self.val_loss_history = list()
        self.accuracy_history = list()
        self.val_accuracy_history = list()
        
    # plot diagnostic learning curve
    def prepare_plot(self):
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title('Cross Entropy Loss')
        ax2.set_title('Accuracy')
        plt.tight_layout()
        return fig, ax1, ax2

    def update_plot(self):
        # plot loss
        self.ax1.plot(self.loss_history, color='blue', label='train_acc')
        self.ax1.plot(self.val_loss_history, color='orange', label='val_acc')
        # plot accuracy
        self.ax2.plot(self.accuracy_history, color='blue', label='train_acc')
        self.ax2.plot(self.val_accuracy_history, color='orange', label='val_acc')
        self.fig.canvas.draw()

### Hyperparameter optimisation

def grid_search(trainX, trainY, testX, testY, max_epochs, learning_rates, momentums, batch_sizes):
    hyperparameter_sets, scores = list(), list()
    callbacks = [PlotProgress()]
    i = 1
    total_runs = len(learning_rates) * len(momentums) *  len(batch_sizes)
    running_time = 0
    start = time.time()
    for lr in learning_rates:
        for momentum in momentums:
            for bs in batch_sizes:
                if i > 1:
                    time_remaining = running_time/(i-1)*(total_runs-(i-1))
                    print('{} of {} done so far so far in {}s. Estimated time remaining: {}s'.format(i-1, total_runs, running_time, time_remaining))
                print('Starting run {} of {}'.format(i, total_runs))
                print('Evaluating Hyperparameters: learning_rate: {}, momentum: {}, batch_size: {}'.format(lr, momentum, bs))
                accuracies, _ = evaluate_model_cross_validation(trainX, trainY, max_epochs=max_epochs, learning_rate=lr, momentum=momentum, batch_size=bs, n_folds=5)
                score = np.log10(1 - np.mean(accuracies))
                scores.append(score)
                with open('grid_scores.pickle', 'wb') as file:
                    pickle.dump(scores, file)
                hyperparameter_sets.append({'learning_rate': lr, 'momentum': momentum, 'batch_size': bs})
                with open('grid_hpsets.pickle', 'wb') as file:
                    pickle.dump(hyperparameter_sets, file)
                running_time = time.time() - start
                i+=1
    return hyperparameter_sets, scores