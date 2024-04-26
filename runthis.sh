#!/bin/bash
python main.py --optimizer Adam --online-mode Binary --weights Normal --max-epoch 100 --model-name wsanodet_SGD_Binary_Normal_100
python main.py --optimizer Adam --online-mode Binary --weights Normal --max-epoch 150 --model-name wsanodet_SGD_Binary_Normal_150
python main.py --optimizer Adam --online-mode Binary --weights Inverse --max-epoch 100 --model-name wsanodet_Adam_Binary_Inverse_100 
python main.py --optimizer Adam --online-mode Binary --weights Inverse --max-epoch 150 --model-name wsanodet_Adam_Binary_Inverse_150 
python main.py --optimizer Adam --online-mode Multi --weights Normal --max-epoch 100 --model-name wsanodet_Adam_Multi_Normal_100 
python main.py --optimizer Adam --online-mode Multi --weights Normal --max-epoch 150 --model-name wsanodet_Adam_Multi_Normal_150 
python main.py --optimizer SGD --online-mode Multi --weights Normal --max-epoch 100 --model-name wsanodet_SGD_Multi_Normal_100 
python main.py --optimizer SGD --online-mode Multi --weights Normal --max-epoch 150 --model-name wsanodet_SGD_Multi_Normal_150 
python main.py --optimizer SGD --online-mode Binary --weights Normal --max-epoch 100 --model-name wsanodet_SGD_Binary_Normal_100
python main.py --optimizer SGD --online-mode Binary --weights Normal --max-epoch 150 --model-name wsanodet_SGD_Binary_Normal_150