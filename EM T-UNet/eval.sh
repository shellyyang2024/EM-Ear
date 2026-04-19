#!/bin/bash

config=config/eval.yaml
dataset_name=LJSpeech
vocoder_ckpt=../checkpoint-400000steps-22k.pkl
vocoder_config=../config-22k.yml
list_val=examples/alienware-ljspeech/alienware_ljspeech_test.csv
audio_path=examples/ljspeech-usrp/audio_wave_22k
load_best_model=examples/alienware-ljspeech/net_best.pth
save_wave_path=examples/alienware-ljspeech/audio

CUDA_VISIBLE_DEVICES=1 python vitunet_vocoder_eval_22k_dnsmos.py \
 --config=$config \
 --vocoder_ckpt=$vocoder_ckpt \
 --vocoder_config=$vocoder_config \
 --dataset_name=$dataset_name \
 --list_val=$list_val \
 --audio_path=$audio_path \
 --load_best_model=$load_best_model \
 --save_wave_path=$save_wave_path