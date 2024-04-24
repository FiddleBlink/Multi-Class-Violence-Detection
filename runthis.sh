#!/bin/bash
python main.py --optimizer SGD --online-mode Binary --weights Normal --max-epoch 50 
python main.py --optimizer Adam --online-mode Multi --weights Normal --max-epoch 50 
python main.py --optimizer SGD --online-mode Multi --weights Normal --max-epoch 50 
python main.py --optimizer Adam --online-mode Binary --weights Inverse --max-epoch 50 
python main.py --optimizer Adam --online-mode Multi --weights Inverse --max-epoch 50 
python main.py --optimizer SGD --online-mode Multi --weights Inverse --max-epoch 50 
python main.py --optimizer SGD --online-mode Binary --weights Inverse --max-epoch 50 