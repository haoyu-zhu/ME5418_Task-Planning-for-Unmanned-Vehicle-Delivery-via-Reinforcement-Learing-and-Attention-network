# Task Planning for Unmanned Vehicle Delivery
This project focuses on the path planning of unmanned vehicles (UVs) for parcels 
delivery across different communities within an urban environment.
- We generate 4 depots and 100 delivery task points with rewards.
- We run an attention-based policy network to choose where the robot goes next (currently random-initialized weights).
- We visualize the robot driving around, unloading cargo, and drawing its path in 3D world.
- We use the REINFORCE algorithm for training and compare it with two baseline methods: EMA and Greedy.
- We test the trained model in two ways: greedy algorithm and sampling to get the maximum reward.

## 1.Environment Setup
There are two ways to set up the environment.
### 1.1 If you ALREADY have the previous environment
```
bash
conda env update -n OPRL -f environment.yml
conda activate OPRL
```

### 1.2 If you are starting from a CLEAN environment
1. Create and install the base dependencies from environment.yml.
2. Activate conda environment.
3. Install torch manually (same command as above) so it matches your GPU.
```
bash
conda env create -f environment.yml -n OPRL
conda activate OPRL
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 2.Run
There are two baselines to train the model.
### 2.1 EMA baseline
```
bash
python train_exp.py
```
### 2.2 Greedy baseline
```
bash
python train_greedy.py
```

### 2.3 Dynamic train greedy baseline
```
bash
python train_dynamic.py
```

### 2.4 Evaluate Model
Change the parameter VISIUALIZE_TEST in eval.py.
VISIUALIZE_TEST == 0: No visualization, evaluate and contrast by sampling and greedy.
VISIUALIZE_TEST == 1: Visualization using greedy policy.
VISIUALIZE_TEST == 2: Visualization using sampling policy.
We currently set VISIUALIZE_TEST == 1

ENVS_NUMBER change the number of environments when evaluating 
```
bash
python eval.py
```


## 3.Repository Structure
### 3.1 valid.py
Builds a fixed (reproducible) verification environment for testing during training.

### 3.2 train_exp.py
Trains the model by REINFORCE algorithm in EMA baseline.

### 3.3 train_greedy.py
Trains the model by REINFORCE algorithm in Greedy baseline.

### 3.4 eval.py
Evaluates and visualizes the trained model.

### 3.5 train_greedy.py
Trains the model by in a dynamic map,task location changing.