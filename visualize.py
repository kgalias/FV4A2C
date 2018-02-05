# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt

import numpy as np
import chainer
#from chainer import Variable
#from chainer import datasets, iterators, optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
#from chainer import training
#from chainer.training import extensions
from chainer.dataset import convert
from chainer.datasets import TupleDataset

def get_mnist(n_train=100, n_test=100, n_dim=1, with_label=True, classes=None):
    """

    :param n_train: nr of training examples per class
    :param n_test: nr of test examples per class
    :param n_dim: 1 or 3 (for convolutional input)
    :param with_label: whether or not to also provide labels
    :param classes: if not None, then it selects only those classes, e.g. [0, 1]
    :return:
    """

    train_data, test_data = chainer.datasets.get_mnist(ndim=n_dim, withlabel=with_label)

    if not classes:
        classes = np.arange(10)
    n_classes = len(classes)

    if with_label:

        for d in range(2):

            if d==0:
                data = train_data._datasets[0]
                labels = train_data._datasets[1]
                n = n_train
            else:
                data = test_data._datasets[0]
                labels = test_data._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i==0:
                    idx = lidx
                else:
                    idx = np.hstack([idx,lidx])

            L = np.concatenate([i*np.ones(n) for i in np.arange(n_classes)]).astype('int32')

            if d==0:
                train_data = TupleDataset(data[idx],L)
            else:
                test_data = TupleDataset(data[idx],L)

    else:

        tmp1, tmp2 = chainer.datasets.get_mnist(ndim=n_dim, withlabel=True)

        for d in range(2):

            if d == 0:
                data = train_data
                labels = tmp1._datasets[1]
                n = n_train
            else:
                data = test_data
                labels = tmp2._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i == 0:
                    idx = lidx
                else:
                    idx = np.hstack([idx, lidx])

            if d == 0:
                train_data = data[idx]
            else:
                test_data = data[idx]

    return train_data, test_data

batchsize = 32
n_units = 10
n_epochs = 50  # not much is visible with 20, so we upped it to 50
max_hid_layers = 3

class CNN(Chain):
    def __init__(self, activation="sigmoid", dropout=False, batch_norm=False):
        
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=1, 
                                        out_channels=5, 
                                        ksize=5, 
                                        stride=1, 
                                        pad=0)
            if self.batch_norm: 
                self.bn = L.BatchNormalization(5)
            self.fc = L.Linear(None, 10)
        
            
    def __call__(self, x):
        x = self.conv(x)
        if self.dropout:
            x = F.dropout(x, ratio=0.4)
        if self.activation == "relu":
            h = F.relu(x)
        #elif self.activation == "sigmoid":
        h = F.sigmoid(x)
        h = F.max_pooling_2d(h, 2, 2)
        if self.batch_norm: 
            h = self.bn(h)
        with chainer.using_config('train', True):
            if self.dropout:
                h = F.dropout(h, ratio=0.4)
            return self.fc(h)
        return F.softmax(self.fc(h))
    
# train, test = chainer.datasets.get_mnist()
train_conv, test_conv = get_mnist(n_dim=3)

train_iter_conv = chainer.iterators.SerialIterator(train_conv, batchsize)
test_iter_conv = chainer.iterators.SerialIterator(test_conv, batchsize,
                                             repeat=False, 
                                             shuffle=False)

train_count_conv = len(train_conv)
test_count_conv = len(test_conv)

def train_CNN(train_iter_conv, test_iter_conv, model, optimizer, n_epochs, train_count, test_count):
    
    train_losses = []
    test_losses = []
    
    sum_accuracy = 0
    sum_loss = 0
    
    while train_iter_conv.epoch < n_epochs:
        batch = train_iter_conv.next()

        x_array, t_array = convert.concat_examples(batch, -1)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

        if train_iter_conv.is_new_epoch:
            print('epoch: ', train_iter_conv.epoch)
            train_loss = sum_loss / train_count
            train_losses.append(train_loss)
            print('train mean loss: {}, accuracy: {}'.format(
                  train_loss, sum_accuracy / train_count))

            sum_accuracy = 0
            sum_loss = 0
            model.predictor.train = False
            for batch in test_iter_conv:
                x_array, t_array = convert.concat_examples(batch, -1)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            test_iter_conv.reset()
            model.predictor.train = True
            test_loss = sum_loss / test_count
            test_losses.append(test_loss)
            print('test mean  loss: {}, accuracy: {}'.format(
                  test_loss, sum_accuracy / test_count))
            sum_accuracy = 0
            sum_loss = 0
            
        return train_losses, test_losses




#%%
net = CNN()
model = L.Classifier(net)    
train_iter_conv.reset()
test_iter_conv.reset()

net.cleargrads()
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)


train_losses_def, test_losses_def = train_CNN(train_iter_conv, test_iter_conv, 
                                                model, optimizer, n_epochs, train_count_conv, test_count_conv)

#%%

test_iter_conv.reset()
image = test_iter_conv.next()

x_array, t_array = convert.concat_examples(image, -1)
tempx = x_array[0,:,:,:]
tempx1 = tempx.reshape(1,1,28,28)
orix = chainer.Variable(x_array)
x = chainer.Variable(tempx1)
t = chainer.Variable(t_array)

plt.imshow(x.data.reshape(28,28))

randomImage = np.random.rand(28,28).astype(np.float32)

output = net(randomImage.reshape(1,1,28,28))

# backprop???

# disables the updates for the conv layer
# net.conv.disable_update()

plt.subplot(1,2,1)
plt.imshow(output.data)

plt.subplot(1,2,2)
plt.imshow(randomImage)

# backprop steps to determine which pixels need which values 
# stimulate the conv layer the most

#%%



#%%
# =============================================================================
# plt.plot(test_losses_def)
# plt.plot(train_losses_def)
# plt.legend(["test loss", "train loss"])
# plt.title("conv")
# plt.ylabel("loss")
# plt.xlabel("epochs")
# # to enforce same yticks for all plots
# ls = np.linspace(0, 2.5, 6)
# plt.yticks(ls)
# # to enforce integer xticks
# eps = np.linspace(1, n_epochs, n_epochs)
# _ = plt.xticks(eps)
# =============================================================================

#%%
# =============================================================================
# i.e., take the network from pa2, freeze the weights in some group of neurons 
# (one neuron, a layer, etc.) and do backprop onto a randomly-initialized group 
# picture (as described in the Feature Visualization distill article)?
# =============================================================================


#%%