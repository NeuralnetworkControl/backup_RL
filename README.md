# Reinforcement Learning for Adaptive Information Acquisition in Founder Success Prediction

This project implements a reinforcement learning framework for
sequential information acquisition in founder success prediction.

Instead of observing all features at once, an agent adaptively queries
information slots (education, roles, executive signals, industry, etc.)
and decides when to stop, balancing predictive accuracy and information
cost.

------------------------------------------------------------------------

## Overview

Traditional prediction models assume full feature availability.\
In practice, information is often incomplete and costly to obtain.

We model founder prediction as a sequential decision process:

-   The agent starts with no information.
-   It can query one information slot at a time.
-   After each query, it may query another slot or stop and make a
    prediction.
-   Rewards depend on prediction correctness and query cost.

The objective is to learn a policy that maximizes reward while
minimizing unnecessary information acquisition.

------------------------------------------------------------------------

## Information Slots

-   edu -- Education background\
-   role -- Professional roles and job titles\
-   exec -- Executive or leadership signals\
-   industry -- Industry exposure\
-   depth -- Experience depth and seniority

------------------------------------------------------------------------

## Project Structure

-   data_store.py\
-   get_observation.py\
-   Tree_value_map.py\
-   main_loop.py\
-   network_trainers.py\
-   Networks.py\
-   train.py\
-   pretrain_classifier.py\
-   cluster_and_clean_train_set.py\
-   llm_next_action_supervisor.py\
-   policy_diagnostics.py\
-   plots_policy_states.py\
-   run_multi_seeds.py

------------------------------------------------------------------------

## Data Format

data2/\
founder_index.csv\
founder_edu_state.npy\
founder_role_vecs.npy\
founder_exec_vecs.npy\
founder_industry_vecs.npy\
founder_depth_feats.npy\
labels_train.csv\
labels_val.csv

------------------------------------------------------------------------

## Training

Clean training data:

    python cluster_and_clean_train_set.py

Pretrain classifier:

    python pretrain_classifier.py --epochs 10

Train RL policy:

    python train.py --n_success_train 500 --n_fail_train 5000 --val_ratio 0.5

Checkpoints, logs, and metrics are automatically saved to:

    runs/YYYYMMDD_HHMMSS/

------------------------------------------------------------------------

## Multi-Seed Testing

To evaluate robustness across different test splits:

    python run_multi_seeds.py         --ckpt runs/.../latest.pt         --labels_pool data2/labels_val.csv         --n_success 90         --n_fail 910         --test_seeds 1 2 3 4 5 6 7 8 9 10

Outputs:

-   multi_test_results.csv\
-   decision_paths_last_seed.json

------------------------------------------------------------------------

## Experimental Results

All experiment artifacts are stored in the `runs/` directory.

Each run folder contains:

-   Model checkpoints (`.pt` files)
-   Configuration file (`config.json`)
-   Training logs
-   Validation metrics
-   Policy diagnostics (entropy, KL)
-   Replay buffer statistics (if enabled)

This structure ensures full reproducibility of experiments and easy
comparison across different training settings and random seeds.
