#coding=utf-8
import numpy as np
import h5py
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    #print ("训练集大小: " + str(train_set_x_orig.shape))
    #print ("训练集标签大小: " + str(train_set_y_orig.shape))

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    #print ("测试集大小: " + str(test_set_x_orig.shape))
    #print ("测试集标签大小: " + str(test_set_y_orig.shape))

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    #print ("分类：" + str(classes))
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    print ("训练集标签reshape后大小: " + str(train_set_y_orig.shape))
    print ("测试集标签reshape大小: " + str(test_set_y_orig.shape))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
