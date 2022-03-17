# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:53:06 2022

@author: gmahi
"""

import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json
from time import time
import pickle
import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.preprocessing import image
from keras import Model
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add
from tensorflow.keras.utils import to_categorical
import flask
from flask import Flask, request, render_template, redirect
import img_captioning_model
from img_captioning_model import predict_it




app = Flask(__name__)

@app.route('/')
def initialize():    
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def caption_it():
    if request.method == 'POST':
        
        f = request.files['userfile']
        path = "./static/uploads/{}".format(f.filename)
        f.save(path)
        
        caption = predict_it().predict_caption(path)
        
        result_dict = {'image': path, 'Caption': caption}
    
    return render_template("index.html", img_result = result_dict)


if __name__ == '__main__':
    app.run(debug = True)
    
    
    
    
    
    

