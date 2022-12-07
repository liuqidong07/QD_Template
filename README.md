# Qidong's DL Model Template

I create the template for fast development of the DL model, including training and inference based on my taste.

____

I will introduce the template according to the Module.

As we know, a complete DL architecture can be divided into `data`, `model`, `train` and `eval` modules.

### 1. Data
You can refer to the `generators\` folder to implemnt all functions about data.

In `data.py`, you can implement your dataset, where the most important thing is to return a batch of data.

The `generator.py` manage the data processing. On the one hand, it can load data and make dataloader. On the other hand, it is a manager which bridge the _data_ and _train_ module.

### 2. Model
All the models are implemented in `models\` folder. The `model.py` is a basic class that you can implement some general functions. Then, create each model in this folder and inherit from the basic class.

### 3. Train
The `trainer.py` implement the basic training process. You should implement your trainer that inherit from the basic trainer. Note that the key of your own trainer is "how to train one batch".

### 4. Eval
The evluation process is embedded in the `trainer.py`, because the evaluation varies much across different tasks.

By $\mathcal{LIU Qidong}$

