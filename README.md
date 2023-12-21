# Readme

Code implementation for paper submited to DSN2024.

# Dataset:
As the SWaT dataset is not publicly available, please request the SWaT dataset from iTrust lab at https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

# Preparation:
* 1. After obtaining the dataset, please use the given attack list in SWaT folder to manually mark the attack dataset's label column to the corresponding attack ID, e.g., A1, A2...A41 etc. Keep the non-attack data label as it is, e.g., 0.
* 2. Dataset preprocessing and training of GDN model, please refer to https://github.com/d-ailin/GDN
* 3. Dataset preprocessing and training of 1dCNN, LSTM, KNN, RFR, and LR, please refer to 'model_evaluate_1dcnn.py' and 'model_evaluate_lstm_knn_rfr_lr.py'
* 4. Please refer to the respective environment file for the models.

# Execution:
## For GDN model:
```
    # Replace the python file name in the run.sh with the target file name listed below, and add pretrained model path to load_model_path in the run.sh.
    bash run.sh cpu swat
    # For GDN model performance check: gdn_model_performance_check.py
    # For evasion attack on GDN model: gdn_ppo_caller.py, adjust the corresponding parameters to meet the experiment needs
    # For calculating the permutation: gdn_feature_permutation.py
    # For delayed or expedited attack: gdn_stealthy.py, adjust the corresponding parameters to meet the experiment needs
    # For testing the max and min perturbation: gdn_action_scale.py
```
## For rest of the model:
* 1. For model train, test, max-min perturbation test, stealthy attack, and feature permutation, please refer to model_evalute_1dcnn.py for 1dCNN model, and model_evaluate_lstm_knn_rfr_lr.py for LSTM, KNN, RFR, and LR models.
```
Execution syntax (same syntax for model_evaluate_lstm_knn_rfr_lr.py as well):
model_evalute_1dcnn.py model_name model_mode evaluate_mode attack_type
Example:
* test the performance of 1dcnn model: model_evalute_1dcnn.py 1dcnn test any any
* test the delayed stealthy attack on rfr model: model_evaluate_lstm_knn_rfr_lr.py rfr stealthy delayed
* obtain the permutation result on knn model: model_evaluate_lstm_knn_rfr_lr.py knn permutation any

```
* 2. For evasion attack, please refer to ppo_1dcnn.py for 1dCNN model, and ppo_lstm_knn_rfr_lr.py for LSTM, KNN, RFR, and LR models.
```
Execution syntax (same syntax for model_evaluate_lstm_knn_rfr_lr.py as well):
ppo_1dcnn.py model_name attack_id false feature_portion_numerator feature_portion_denominator
Example:
ppo_1dcnn.py lstm A38 false 3 3
```