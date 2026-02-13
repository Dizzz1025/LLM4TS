# LLMODE: Aligning ODEs with LLMs for Spatio-Temporal Forecasting

This document provides a comprehensive guide to the training pipeline of the **LLMODE** framework, based on the paper "LLMODE: Aligning ODEs with LLMs via Gated Token Injection for Irregular Spatio-Temporal Forecasting". The goal of this project is to tackle irregular spatio-temporal forecasting by combining continuous-time models (ODEs) with Large Language Models (LLMs).

## Architecture Overview

LLMODE is a **two-stage pipeline**. An **ODE-based dynamics encoder** first extracts continuous-time latent states from irregular spatio-temporal data, and a **frozen LLM forecaster** then performs forecasting by consuming compressed evidence tokens plus textual priors.

1. **ODE Encoder (Latent Dynamics Encoding)**
   - **Code**: `models/ode/gru_ode_bayes/models.py`
   - **Role**: processes raw, irregularly-sampled spatio-temporal observations and produces a latent trajectory `h(t)` as **dynamics evidence**. It models continuous evolution between observations and applies discrete updates when new observations arrive.

2. **LLM Forecaster (Token Compression + Gated Injection)**
   - **Code**:
     - Token compression: `models/resampler/PerceiverResampler.py`
     - LLM injection + forecasting: `models/LLMODE/STLlama.py`
   - **Role**: the frozen LLM does **not** take raw time series as input. Instead, it uses:
     - **Dynamics evidence tokens**: compressed latent states exported from the ODE encoder
     - **Auxiliary priors**: text descriptor files (e.g., statistics summaries)
     - **Instruction prompts**: natural-language task instructions (forecast horizon, targets, etc.)
   - These external tokens are injected into the frozen LLM through the gated injection modules defined in `models/LLMODE/STLlama.py`.

---

## End-to-End Training Pipeline

The training process is a two-stage procedure. First, we generate the dynamics evidence using the ODE Encoder. Second, we train the LLM Forecaster using this evidence.

### **Step 1: Generate Dynamics Evidence with the ODE Encoder**

The LLM does not learn from raw time-series data. Its understanding of the system's dynamics comes entirely from the latent states pre-computed by the ODE Encoder.

#### Train the ODE Encoder (NYC example)
```bash
cd models/ode/experiments/NYC
python NYC_train.py
```
After training the ODE model, you can run the **models/ode/experiments/NYC/NYC_test.py** to automatically generate and save the following outputs for each time-series instance:

- `h`: latent states over the historical window (dynamics evidence), shape `[num_samples, total_length, hidden_size]`
- `mask`: observation mask for the future window
- `gt`: ground-truth values for the future window (LLM labels)

These arrays are saved together as a single NumPy archive (e.g., `h_mask_gt_merged.npz`), which will be used in the next stage.

> Note: The test script is responsible for saving `h/mask/gt`. In this open-source release, we currently provide **only the training script**.


### **Step 2: Prepare Textual Data**

The LLM requires textual prompts to understand the task and context. You need to prepare two JSON files:

1.  **Instruction Prompts** (e.g., `NYC_train_prompts.json`):
    *   This file contains the natural language instructions for the model. Each entry should have a unique `id` corresponding to a sample in your `.npz` file.
    *   The prompt describes the task and contains time-aware query slots where predictions are expected.
    *   *Example Entry*:
        ```json
        {
            "id": "0",
            "conversations": [
                {
                    "from": "human",
                    "value": "Task: Predict future inflow and outflow at the query slots given historical observations over NYC regions, where each node corresponds to a region described by its POI profile. \nRegion: Bronx area with dominant POIs: Cultural Facility, Residential, Education Facility. \nQuery Slots: <q1> <q2> ... <qN>."
                }
            ]
        }
        ```

2.  **Auxiliary Statistical Priors** (e.g., `NYC_text_stats.json`):
    *   This file provides auxiliary priors (e.g., segment-level statistics and dataset-specific descriptors) that complement the dynamics evidence.
    *   *Example Entry*:
        ```json
        {
            "0": "[SEG_STATS K=4] seg1: inflow_mean=-0.34, outflow_mean=-0.26, inflow=[max: -0.34, min: -0.34], outflow=[max: -0.26, min: -0.26], obs_rate=83.33% seg2: inflow_mean=-0.29, outflow_mean=-0.26, inflow=[max: -0.27, min: -0.32], outflow=[max: -0.25, min: -0.26], obs_rate=100.00% seg3: inflow_mean=-0.23, outflow_mean=-0.25, inflow=[max: -0.18, min: -0.26], outflow=[max: -0.25, min: -0.25], obs_rate=50.00% seg4: inflow_mean=-0.08, outflow_mean=-0.24, inflow=[max: -0.07, min: -0.11], outflow=[max: -0.24, min: -0.24], obs_rate=83.33% max_gap=5.00 [/SEG_STATS]"
        }
        ```

### **Step 3: Configure and Run the LLM Training Script**

With the dynamics evidence and text files prepared, you can now train the main LLM Forecaster.

- **Training script**: `/home/zhangdi24/LLM4TS/models/LLMODE/llmode_train.sh` (referred to as `train.sh` below) 
1.  **Configure `train.sh`**:
    *   Open your training script (`train.sh` or similar).
    *   Set the correct paths for all your data files and the base model.

    ```bash
    #!/bin/bash

    # 1. Core Configurations
    MODEL_PATH="./checkpoints/vicuna-7b-v1.5-16k" # Path to base pre-trained LLM
    OUTPUT_DIR="./checkpoints/NYC"           # Where the final model will be saved

    # 2. Data Paths (from Steps 1 & 2)
    TRAIN_PROMPTS="./data/NYC/train/NYC_train_prompts.json"
    TRAIN_ST_DATA="./data/NYC/train/h_mask_gt_merged.npz" # Dynamics evidence
    TRAIN_TEXT_STATS="./data/NYC/train/NYCmerged_text_stats.json"

    # ... (evaluation paths and other hyperparameters) ...
    ```

2.  **Execute the Script**: Run the shell script to start the training.
    ```bash
    bash train.sh
    ```

Upon completion, the trained model, containing the weights for the adapters and the regression head, will be saved in the specified `OUTPUT_DIR`.


### Acknowledgments

This implementation includes code adapted from the following open-source projects:

- GRU-ODE-Bayes (De Brouwer et al., MIT License)
- UrbanGPT (Apache License 2.0)
- flamingo-pytorch (lucidrains, MIT License)

We thank the authors and maintainers of these projects for making their code publicly available.

> **Anonymous review note:** For the double-blind submission, we release only the core code in this repository.  
> We will release the full codebase and processed data upon acceptance.
