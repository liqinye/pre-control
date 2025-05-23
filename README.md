# Pre-Control
Precise Attribute Intensity Control in Large Language Models via Targeted Representation Editing
![image](figure/pc_mainfig.png)

## Environment Setup
Setup Conda Environment: 
```bash
conda create -n pre-control python=3.10
conda activate pre-control
pip install -r requirements.txt
```

## Run Value Function Training 

### Step 1. Generate Training Dataset 

First, get the LLM responses and the corresponding hidden states:
```bash
bash scripts/get_vf_train_dataset.sh
```

Then, label the generated responses with a trained Reward Model: 
```bash
bash scripts/get_vf_train_dataset_labels.sh
```

### Step 2. Train the Value Function (VF) 
Train the VF on the generated dataset:
```bash
bash scripts/train_vf.sh
```

### Step 3. Evaluate the VF

First, get the LLM responses with and without hidden state editing using the trained VF:
```bash
bash scripts/run_inference_intervention.sh
```

Then, get the reward scores of the generated responses:
```bash
bash scripts/get_vf_edited_response_scores.sh
```