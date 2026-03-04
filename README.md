# 🔍 Founder Success Predictor --- Adaptive RL Policy with LLM Supervision

An adaptive information-gathering system that predicts startup founder
success using a learned policy network guided by Monte Carlo tree search
and optional LLM supervision. Instead of consuming all features upfront,
the agent dynamically decides which information to query next and stops
once sufficient signal is collected.

------------------------------------------------------------------------

## 🧠 How It Works

The system formulates founder evaluation as a sequential decision
problem:

1.  A **PolicyNet** observes a partially revealed founder state and
    selects the next information slot to query (or STOP).
2.  A **Classifier** predicts success probability based on the
    accumulated state.
3.  **Monte Carlo Tree Search (Tree_value_map)** estimates Q-values for
    possible actions and provides soft targets for policy learning.
4.  An optional **LLM supervisor** provides weak action preferences when
    the policy is uncertain.

------------------------------------------------------------------------

## 📁 Repository Structure

    ├── main_loop.py
    ├── Networks.py
    ├── network_trainers.py
    ├── Tree_value_map.py
    ├── get_observation.py
    ├── data_store.py
    ├── train.py
    ├── pretrain_classifier.py
    ├── Cluster_and_clean_train_set.py
    ├── run_multi_seeds.py
    ├── llm_next_action_supervisor.py
    ├── label.py
    ├── policy_diagnostics.py
    ├── plots_policy_states.py

------------------------------------------------------------------------

## 🗂 Data Details

In the `data/` directory:

-   The `.npy` files are feature vectors embedded from the full
    (private) dataset.\
-   `labels_train.csv` and `labels_val.csv` are the training and
    validation splits.\
-   `labels_train_clean.csv` is the cleaned training set after
    clustering-based outlier removal.

⚠️ The complete dataset used to generate embeddings is private and
cannot be publicly released.

------------------------------------------------------------------------

## 🧠 About `train.py` and LLM Supervision

In `train.py`, you can toggle:

``` python
USE_LLM = True / False
```

### ✅ Training Without LLM (Fully Reproducible)

``` bash
python train.py
```

This trains a pure RL decision model using only embedded feature
vectors.\
Fully reproducible.

### ⚠️ Training With LLM Supervisor

The LLM supervisor requires original textual background information as
observed input.

Because the full dataset is private, exact reproduction of the
LLM-supervised training setting in the paper is not possible.

------------------------------------------------------------------------

## 🧪 Using Public Data to Try LLM Supervision

To experiment with LLM supervision:

1.  Use the provided public dataset: `vcbench_final_public`
2.  Generate embedding vectors from it.
3.  When calling `llm_next_action_supervisor`, provide:
    -   The original textual information
    -   The corresponding embedded vectors for RL training

This enables a partially reproducible LLM-supervised setup.

------------------------------------------------------------------------

## 🧪 Reproducing Experimental Results

The `runs/` directory contains saved experiment checkpoints, including:

-   `model_llm_supervisor/`
-   `model_without_llm/`

To reproduce reported results:

``` bash
python run_multi_seeds.py \
  --ckpt runs/model_llm_supervisor/final_model.pt \
  --test_seeds 0 1 2 3 4 5 6 7 8 9
```

This outputs evaluation performance and generates:

    decision_paths_last_seed.json

which records decision-making chains for the final seed.

------------------------------------------------------------------------

## 📊 Visualization

Run:

``` bash
python plots_policy_states.py
```

This generates entropy and policy activity comparison plots (LLM vs
non-LLM).

------------------------------------------------------------------------

## 📦 Dependencies

    torch
    numpy
    pandas
    scikit-learn
    matplotlib

------------------------------------------------------------------------

## 📄 License

MIT
