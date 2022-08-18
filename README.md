# SSIL_Food
Repository for Self-Supervised Incremental Learning in Food Recognition.

This project is done in partial fulfilment of the requirements for the Master of Data Science Programme, Univerity of Malaya.

---
## Data

The data should be stored and structured in the following way:
 ```
data 
|-- food-101 
|   |-- images 
|	|-- {dir of all 101 image classes} 
|   |-- meta 
|	|-- {classes, labels, train test split txt/json files} 
|   |-- food101_split.yaml 
|   |-- food101_split_off_inc.py 
|
|-- uec256 
	  |-- meta 
	  |--- UECFOOD256 
	      |--- {dir of all 256 image classes} 
	      |--- category.txt 

```
---

# Model Training

As the name suggest: \
1a and 1b are scripts for offline transfer learning. \
2a and 2b are scripts for incremental learning. 

To reproduce the result, run all of the 4 scripts. Some of the parameters that can be changed (in script) are:

- ds: specify dataset to be used. 'f1' for Food101 and 'uec' for UECFood256
- state_path: specify name of model state to continue training. If training from scratch, specify it as None.
- to_train (boolean): Specify as True to continue training the model given in state_path.
- eval_perf (boolean): Specify as True to evaluate top-1 accuracy of the model given in state_path.
- show_train_loss (boolean): Specify as True to show the Loss Curve of the model given in state_path.
- max_epoch, wu_epoch: Specify the number of maximum epoch and warmup epoch.
- batch_s: batch size. To be changed according to GPU VRAM.
- lr: base learning rate, i.e. learning rate at epoch = wu_epoch.
