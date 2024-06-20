#!/bin/bash

python model_train.py --ds x15x12 --stride 4 --loss mae --n_epoch 100

python model_train.py --ds x15x12 --stride 4 --loss huber --n_epoch 100

python model_train.py --ds x15x12 --stride 4 --loss mse --n_epoch 100
