import os
import scipy.io
import pandas as pd
import numpy as np

def preprocess_input_clf(_input):
    #pertube input
    n_aug = 10
    xs = _input[:,:13]
    ys = _input[:,13:26]
    max_x = np.max(np.array(xs).reshape(-1))
    min_x = np.min(np.array(xs).reshape(-1))
    max_y = np.max(np.array(ys).reshape(-1))
    min_y = np.min(np.array(ys).reshape(-1))
    x_aug = [-i*(min_x - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_x)/n_aug for i in range(0, n_aug+1)]
    y_aug = [-i*(min_y - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_y)/n_aug for i in range(0, n_aug+1)]
    move_x = np.random.choice(x_aug)
    move_y = np.random.choice(y_aug)
    new_xs = np.array(xs)+move_x*np.ones(shape=(xs.shape[0], xs.shape[1]))
    new_ys = np.array(ys)+move_y*np.ones(shape=(ys.shape[0], ys.shape[1]))
    result = np.append(new_xs, new_ys, axis=1)
    result = np.append(result, _input[:,-13:], axis=1)
    return result

def get_input_clf(_id, train_input):
    result = preprocess_input_clf(train_input[_id])
    return result


def get_output_clf(_id, train_label):
    return train_label[_id]


def data_generator_clf(train_input, train_label, batch_size = 64, augmented=True):   
    while True:
        batch_idx = np.random.choice(range(0, train_input.shape[0]), 
                                     size = batch_size)
        shape = (1, train_input.shape[1], 13*3)
        batch_input = np.zeros(shape=shape)
        batch_output = np.zeros(shape=(1, train_label.shape[1])) 
        if augmented:
            for idx in batch_idx:
                _input = get_input_clf(idx, train_input).reshape(1, batch_input.shape[1], batch_input.shape[2])
                _output = get_output_clf(idx, train_label).reshape(1, batch_output.shape[1])
                batch_input = np.append(batch_input, _input, axis=0)
                batch_output = np.append(batch_output, _output, axis=0)
        else:
            for idx in batch_idx:
                _input = train_input[idx].reshape(1, batch_input.shape[1], batch_input.shape[2])
                _output = train_label[idx].reshape(1, batch_output.shape[1])
                batch_input = np.append(batch_input, _input, axis=0)
                batch_output = np.append(batch_output, _output, axis=0)
        batch_x = batch_input[1:]
        batch_y = batch_output[1:]
        
        yield batch_x, batch_y

####################################################################################################################################

def preprocess_input_nf(_input):
    #pertube input
    n_aug = 10
    xs = _input[:,:13]
    ys = _input[:,13:26]
    max_x = np.max(np.array(xs).reshape(-1))
    min_x = np.min(np.array(xs).reshape(-1))
    max_y = np.max(np.array(ys).reshape(-1))
    min_y = np.min(np.array(ys).reshape(-1))
    x_aug = [-i*(min_x - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_x)/n_aug for i in range(0, n_aug+1)]
    y_aug = [-i*(min_y - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_y)/n_aug for i in range(0, n_aug+1)]
    move_x = np.random.choice(x_aug)
    move_y = np.random.choice(y_aug)
    new_xs = np.array(xs)+move_x*np.ones(shape=(xs.shape[0], xs.shape[1]))
    new_ys = np.array(ys)+move_y*np.ones(shape=(ys.shape[0], ys.shape[1]))
    result = np.append(new_xs, new_ys, axis=1)
    result = np.append(result, _input[:,-13:], axis=1)
    return result, move_x, move_y

def get_input_nf(_id, train_input):
    result, move_x, move_y = preprocess_input_nf(train_input[_id])
    return result, move_x, move_y


def get_output_nf(_id, train_label, move_x, move_y):
    xs = train_label[_id][:13]
    ys = train_label[_id][13:26]
    new_xs = np.array(xs)+move_x*np.ones(shape=(xs.shape[0]))
    new_ys = np.array(ys)+move_y*np.ones(shape=(ys.shape[0]))
    result = np.append(new_xs.reshape(1, new_xs.shape[0]), new_ys.reshape(1, new_ys.shape[0]), axis=1)
    result = np.append(result, train_label[_id,-13:].reshape(1,13), axis=1)
    return result


def data_generator_nf(train_input, train_label, batch_size = 64, augmented=True):   
    while True:
        batch_idx = np.random.choice(range(0, train_input.shape[0]), 
                                     size = batch_size)
        shape = (1, train_input.shape[1], 13*3)
        batch_input = np.zeros(shape=shape)
        batch_output = np.zeros(shape=(1, train_label.shape[1])) 
        if augmented:
            for idx in batch_idx:
                _input, move_x, move_y = get_input_nf(idx, train_input)
                _input = _input.reshape(1, batch_input.shape[1], batch_input.shape[2])
                _output = get_output_nf(idx, train_label, move_x, move_y).reshape(1, 39)
                batch_input = np.append(batch_input, _input, axis=0)
                batch_output = np.append(batch_output, _output, axis=0)
        else:
            for idx in batch_idx:
                _input = train_input[idx].reshape(1, batch_input.shape[1], batch_input.shape[2])
                _output = train_label[idx].reshape(1, 39)
                batch_input = np.append(batch_input, _input, axis=0)
                batch_output = np.append(batch_output, _output, axis=0)
        batch_x = batch_input[1:]
        batch_y = batch_output[1:]
        
        yield batch_x, batch_y


####################################################################################################################################

def preprocess_input_en(_input):
    #pertube input
    n_aug = 10
    xs = _input[:13]
    ys = _input[13:26]
    max_x = np.max(np.array(xs).reshape(-1))
    min_x = np.min(np.array(xs).reshape(-1))
    max_y = np.max(np.array(ys).reshape(-1))
    min_y = np.min(np.array(ys).reshape(-1))
    x_aug = [-i*(min_x - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_x)/n_aug for i in range(0, n_aug+1)]
    y_aug = [-i*(min_y - 0)/n_aug for i in range(0, n_aug+1)] + [i*(1-max_y)/n_aug for i in range(0, n_aug+1)]
    move_x = np.random.choice(x_aug)
    move_y = np.random.choice(y_aug)
    new_xs = np.array(xs)+move_x*np.ones(shape=(1,13))
    new_ys = np.array(ys)+move_y*np.ones(shape=(1,13))
    result = np.append(new_xs, new_ys, axis=1)
    result = np.append(result, _input[-13:].reshape(1,13), axis=1)
    return result, move_x, move_y

def get_input_en(_id, train_input):
    result, move_x, move_y = preprocess_input_en(train_input[_id])
    return result, move_x, move_y


def get_output_en(_id, train_label, move_x, move_y):
    xs = train_label[_id, :13]
    ys = train_label[_id, 13:26]
    new_xs = np.array(xs)+move_x*np.ones(shape=(xs.shape[0]))
    new_ys = np.array(ys)+move_y*np.ones(shape=(ys.shape[0]))
    result = np.append(new_xs.reshape(1, new_xs.shape[0]), new_ys.reshape(1, new_ys.shape[0]), axis=1)
    result = np.append(result, train_label[_id,-13:].reshape(1,13), axis=1)
    return result


def data_generator_en(train_input, train_label, batch_size = 64, augmented=True):   
    while True:
        batch_idx = np.random.choice(range(0, train_input.shape[0]), 
                                     size = batch_size)
        shape = (1, 13*3)
        batch_input = np.zeros(shape=shape)
        batch_output = np.zeros(shape=shape) 
        if augmented:
            for idx in batch_idx:
                _input, move_x, move_y = get_input_en(idx, train_input)
                _input = _input.reshape(1, 39)
                _output = get_output_en(idx, train_label, move_x, move_y).reshape(1, 39)
                batch_input = np.append(batch_input, _input, axis=0)
                batch_output = np.append(batch_output, _output, axis=0)
        else:
            for idx in batch_idx:
                _input = train_input[idx].reshape(1, 39)
                _output = train_label[idx].reshape(1, 39)
                batch_input = np.append(batch_input, _input, axis=0)
                batch_output = np.append(batch_output, _output, axis=0)
        batch_x = batch_input[1:]
        batch_y = batch_output[1:]
        
        yield batch_x, batch_y