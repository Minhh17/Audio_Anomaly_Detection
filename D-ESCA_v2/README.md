# Environment Sound Collection & Analysis Platform (ESCA)

ESCA has represented the real-time abnormal environment sound detection.




## Contents

1. [Requirements](#-Requirements)
2. [Features](#-Features)
3. [Prepare data and Training](#-Prepare-data-and-Training)
4. [Results](#-Results)
## Requirements (ThinkEdge SE70)
* Python 3.7 or higher
* Setup the environment according to the steps in the [Environment_Setting_Guide.docx](./Docs/Environment_Setting_Guide.docx) documentation.


## Features

### Main Functionality
* **Create Dataset:** Create audio dataset 
* **Base Training:** Training model with labelled environment audio dataset
* **Transfer Learning:** Use pre-trained model for update model with new audio dataset. The new model contains the features of both the old and the new dataset
* **Real-time Dectection:** Model implementation to detect anomalies in audio tracks

**Input:** Audio recording files as format.wav

**Output:** Loss value, Normal/Abnormal classificaion of files splitted from the original audio file.

## Create dataset
* Change specific parameter in [config_file](./config/params.yaml)

```bash
RECORD: 
  DEVICE_INDEX_INPUT: '''# Index number of recording device #'''
  ABNOMALY: '''# [Default = False] = True if create anomaly dataset #'''
  SECOND: '''# Length of a recording file #'''
  DATASET_PATH: '''# Dataset save path #'''
  ...
```
*To run create dataset:
```bash
    python ./tool/create_dataset.py -cfg ./config/params.yaml
```

## Prepare data and Training

### Prepare
* Change specific PATH in [config_file](./config/params.yaml)

```bash
DATASET:
  PATH:
    TFRECORD:'''# TFRecord save path #'''
    NORMAL: '''#  directory of normal training data [source data]#'''
    ABNORMAL: '''# directory of anomaly dataset (only use for testing)[source data] #'''
    TEST: '''# directory to test dataset. If this is None then part of normal training data will be used for test#'''

```

* The model expect audio data length is 2 seconds in TFRecord type:
```bash
  python ./tools/prepare_data.py -cfg ./config/params.yaml
```
### Base Training
* Change specific parameter in [config_file](./config/params.yaml)

```bash
TRAINING: 
  LOG_FOLDER : '''# log save path #'''
  EPOCH :   '''# Number of epoch #'''
```
*To run base-training after prepare data: 
```bash
  python ./tools/base_training.py -cfg ./config/params.yaml
```
### Transfer Learning
* Change specific parameter in [config_file](./config/params.yaml)

```bash
TRANSFER_LEARNING: 
  NORMAL_DATA_DIRS:  '''#  directory of normal training data [target data]#'''
  ANOMALY_DATA_DIRS: '''# directory of anomaly dataset [target data] #'''
  BASED_WEIGHTS: '''# base model download path #'''
```
*To run transferlearing: 
```bash
  python ./tools/tl_training.py -cfg ./config/params.yaml
```

### Real-time
* Change specific parameter in [config_file](./config/params.yaml)

```bash
REALTIME: 
  TRANSFER_LEARNING = : '''# [Default = False] Run real-time with transfer learning model #'''
  LOG_PATH :'''# log path #'''
  MANUAL_THRESHOLD :'''# [Default = None] #'''
  RUNTIME : '''# Time of run #'''
```
*To run realtime-detection:
```bash
   python ./tools/rt_test.py -cfg ./config/params.yaml  
```
## Results
## Documentation

[Documentation](./Docs/)
