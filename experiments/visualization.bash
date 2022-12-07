#! /bin/bash
tensorboard_path='./log/full/'
tensorboard --logdir=${tensorboard_path} --host 0.0.0.0 --port 30001