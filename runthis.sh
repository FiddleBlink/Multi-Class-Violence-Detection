#!/bin/bash
python main.py --optimizer SGD --online-mode Binary --weights Normal --max-epoch 50 --model-name wsanodet_SGD_Binary_Normal_50
python main.py --optimizer Adam --online-mode Multi --weights Normal --max-epoch 50 --model-name wsanodet_Adam_Multi_Normal_50 
python main.py --optimizer SGD --online-mode Multi --weights Normal --max-epoch 50 --model-name wsanodet_SGD_Multi_Normal_50 
python main.py --optimizer Adam --online-mode Binary --weights Inverse --max-epoch 50 --model-name wsanodet_Adam_Binary_Inverse_50 
python main.py --optimizer Adam --online-mode Multi --weights Inverse --max-epoch 50 --model-name wsanodet_Adam_Multi_Inverse_50 
python main.py --optimizer SGD --online-mode Multi --weights Inverse --max-epoch 50 --model-name wsanodet_SGD_Multi_Inverse_50 
python main.py --optimizer SGD --online-mode Binary --weights Inverse --max-epoch 50 --model-name wsanodet_SGD_Binary_Inverse_50 