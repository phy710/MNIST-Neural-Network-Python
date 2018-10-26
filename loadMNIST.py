# -*- coding: utf-8 -*-
"""
Load MNIST Dataset

Created on Sat Oct 20 17:01:00 2018

@author: Zephyr
"""

import numpy as np

def loadMNISTLabels(filename):
    f = open(filename, 'rb')
    assert f!=-1, 'Could not open '+filename
    labels = np.frombuffer(f.read(), np.uint8, offset=8)
    f.close()
    return labels

def loadMNISTImages(filename):
    f = open(filename, 'rb')
    assert f!=-1, 'Could not open '+filename
    images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    f.close()
    images = images.T.astype(np.float64)/255
    return images
