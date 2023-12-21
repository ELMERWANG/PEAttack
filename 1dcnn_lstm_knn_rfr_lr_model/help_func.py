from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Activation
import numpy as np
from mylibrary import load_variable, series_to_supervised
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
import random 
import pandas as pd
import sys
import tensorflow as tf
import gc

def augment_data(data, lag=1):
    """
    Augment the data by computing the difference between the current value and a past value with a given time lag.
    Returns augmented_data: numpy ndarray containing the original data along with the new augmented features
    """
    # Calculate the differences
    diff = data[lag:] - data[:-lag]
    # Stack the original data and differences horizontally (column-wise)
    augmented_data = np.hstack((data[lag:], diff))
    
    return augmented_data

def delay_attack_payload_generator(data, change_level, trim_to_input):
    '''<Locked>
    Generate attack payload in a delayed mode
    Input: 
    - data: data stream, type: list, if input is df, use tolist()
    - change level: (int), from 1 to n. 1 means, insert 1 after each 1, 3 means insert 1 after each 3.
    - trim to input: after delay attack, the generated data length is more than input, if true, keep the output with the same length as input
    - window size: unimplemented
    - randomness ratio: unimplemented
    - distribution comparison: unimplemented

    Output:
    - converted data: modified data stream according to the change ratio.
    '''

    converted_data = [data[0]] # initialize with the first value in the data stream
    accumulate_res = 0 # accumulated residual for each data
    for i in range(1, len(data)-1): # calculate diff beforehand, but can be easily convert to online mode
        diff = data[i]-data[i-1]
        converted_data.append(converted_data[-1] + (1-1 / change_level) * diff)
        accumulate_res += 1 / change_level * diff

        if (len(converted_data)-1) % change_level == 0:
            converted_data.append(accumulate_res+converted_data[-1])
            accumulate_res = 0

    if accumulate_res != 0:
        converted_data.append(accumulate_res+converted_data[-1]) # if there is any residual left, and the left number of data is not enough for one more round, add directly to the end of the list

    if trim_to_input:
        converted_data = converted_data[0:len(data)]

    return converted_data

def expedited_attack_payload_generator(ori_data, change_level, drop_amount, rand_coeff):
    '''<Locked>
    Generate attack payload in a expedited mode
    Input: 
    - ori_data: data stream, type: list, if input is df, use tolist()
    - change level: (int), from 1 to n. 1 means, every 1 drop drop_amount data 
    - drop_amount: the number of data not copied
    - rand_coeff: the randomness introduced to the data

    Output:
    - converted data: modified data stream according to the change ratio.
    '''

    converted_data = []
    ori_counter = 0

    while ori_counter < len(ori_data):
        for _ in range(change_level):
            if ori_counter < len(ori_data):
                value = ori_data[ori_counter] * (1 + random.uniform(-rand_coeff, rand_coeff))
                converted_data.append(value)
                ori_counter += 1
        # Skip drop_amount elements
        ori_counter += drop_amount

    return converted_data

def various_label_check(predicted_label, real_label):
    '''Give the predicted label list, and the real labels contains various labels, this function can print how many type of attacks are detected, also print in percentage'''

    # Ensure the lists are of the same length
    if len(predicted_label) != len(real_label):
        print('len(predicted_label): ', len(predicted_label))
        print('len(real_label): ', len(real_label))
        raise ValueError("The two lists must be of the same length.")

    # Using a dictionary to track the counts of each "A" prefixed item in real_label
    a_counts = {}

    total_as = 0  # Total occurrences of all A-prefixed items
    total_ones_for_as = 0  # Total number of 1s corresponding to all A-prefixed items

    # Iterate over both lists simultaneously
    for pred_val, real_val in zip(predicted_label, real_label):
        # Check if the value in real_label starts with 'A'
        if str(real_val).startswith('A'):
            # Initialize the counts if this is a new "A" prefixed item
            if real_val not in a_counts:
                a_counts[real_val] = {'total': 0, 'ones': 0}
            a_counts[real_val]['total'] += 1
            total_as += 1
            
            # Check if the corresponding value in predicted_label is 1
            if pred_val == 1:
                a_counts[real_val]['ones'] += 1
                total_ones_for_as += 1

    all_attack_id_list = ['A1', 'A2', 'A3', 'A4', 'A6', 'A7', 'A8', 'A10', 'A11', 'A13', 'A14', 'A16', 'A17', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41']

    for key, values in a_counts.items():
        if key in all_attack_id_list:
            percentage = (values['ones'] / values['total']) * 100
            print(f"For {key}:")
            print(f"Total occurrences: {values['total']}")
            print(f"Number of corresponding '1's in predicted_label: {values['ones']}")
            print(f"Percentage: {percentage:.3f}%")
            print("-------------------------")


    overall_percentage = (total_ones_for_as / total_as) * 100 if total_as != 0 else 0
    print(f"Overall for all A-prefixed items:")
    print(f"Total occurrences: {total_as}")
    print(f"Total number of corresponding '1's: {total_ones_for_as}")
    print(f"Overall Percentage of attack detected: {overall_percentage:.5f}%")

def find_indices(lst, value):
    '''Used with get_attack_data function'''
    start_index = None
    end_index = None

    for i, v in enumerate(lst):
        if v == value and start_index is None:
            start_index = i
        elif start_index is not None and v != value:
            end_index = i - 1  # The last occurrence of the consecutive sequence
            break

    # If the sequence goes till the end of the list
    if start_index is not None and end_index is None:
        end_index = len(lst) - 1
    return start_index, end_index

def get_attack_data(model_name, attack_id, step_size, passivity):
    '''This function returns the complete data in the attack dataset for the given attack ID, step_size is used to preserve the data ahead of the complete data chunk.
    
    output: dataset contains only the attack instance ID
    '''
    new_attack_dataset = None
    new_attack_label = None
    new_attack_label_various = None

    if model_name == '1dcnn':
        dataset_attack = load_variable('swat_dataset_attack_v0_various_label')

        values_attack = dataset_attack.drop('label', axis=1).values
        label_attack = np.asarray(dataset_attack['label'])

        test_labels_list_various = label_attack.tolist() # 449919

        label_attack = [1 if isinstance(item, str) and item.startswith('A') else item for item in test_labels_list_various]
        values_attack = values_attack.astype('float32')

        scaler = load_variable('scaler_for_1dcnn')
        test_dataset = scaler.transform(values_attack)

    elif model_name in ['lstm', 'knn', 'rfr', 'lr']:
        dataset_attack = pd.read_csv('./dataset/swat/test_all_label.csv', sep=',', index_col=0)

        test_labels_list_various = dataset_attack['attack'].tolist()
        label_attack_0_1 = [1 if isinstance(x, str) and x.startswith('A') else 0 for x in test_labels_list_various] # convert to 1s and 0s only

        test_dataset = dataset_attack.drop('attack', axis=1).values
        label_attack = label_attack_0_1

    else:
        print('No model provided, exiting..')
        sys.exit()

    attack_start_index, attack_end_index = find_indices(test_labels_list_various, attack_id)

    if model_name == '1dcnn':
        attack_end_index += 1999
        attack_start_index -= 1

    attack_end_index += 1

    if passivity == True:
        attack_end_index += step_size # preserve enough data after the end index for passivity check
    
    print('attack start index is: ', attack_start_index)
    print('attack end index is: ', attack_end_index)
    print('Selected length is: ', attack_end_index-attack_start_index)

    if attack_start_index - step_size >= 0:
        new_attack_dataset = test_dataset[attack_start_index-step_size:attack_end_index] # Step size: amount of extra data is included
        new_attack_label = label_attack[attack_start_index-step_size:attack_end_index]
        new_attack_label_various = test_labels_list_various[attack_start_index-step_size:attack_end_index]
    else:
        raise ValueError('No extra step size data can be found')
    
    return new_attack_dataset, new_attack_label, new_attack_label_various

def find_consecutive_zeros(lst, chunk_length):

    sequence_length = chunk_length

    for i in range(len(lst) - sequence_length + 1):
        # Check if the next sequence_length elements are all zeros
        if all(lst[i + j] == 0 for j in range(sequence_length)):
            return i

    # Return -1 if no such sequence is found
    return -1

def feature_permutation(model, model_name, label_attack, values_attack, threshold, n_obs, look_back_window, no_of_input_features):
    '''Feature permutation for lstm, knn, rfr, lr.'''
    sensor_feature_index = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
    per_result = []

    for i in sensor_feature_index:
        X_test_permuted = values_attack.copy()
        np.random.shuffle(X_test_permuted[:, i])
        reframed_attack_dataset = series_to_supervised(X_test_permuted, look_back_window, 1)

        attack_values = reframed_attack_dataset.values
        attack_x = attack_values[:,:n_obs]
        attack_y = attack_values[:,n_obs:]
        if model_name == 'lstm':
            attack_x = attack_x.reshape((attack_x.shape[0], look_back_window, no_of_input_features))
        y_pred_attack = model.predict(attack_x)
        mse_values = [mean_squared_error([true], [pred]) for true, pred in zip(attack_y, y_pred_attack)]
        y_pred_binary = [1 if mse > threshold else 0 for mse in mse_values]
        n = 20
        y_pred_binary_refined = []

        for j in range(len(y_pred_binary)):
            if y_pred_binary[j] == 1:
                if j + n < len(y_pred_binary):
                    subsequent_anomalies = sum(y_pred_binary[j+1:j+n+1]) # Count how many of the next n data points are anomalies
                    if subsequent_anomalies >= n / 8:
                        y_pred_binary_refined.append(1)  # Confirm as anomaly
                    else:
                        y_pred_binary_refined.append(0)  # Mark as normal
                else:
                    y_pred_binary_refined.append(y_pred_binary[j])
            else:
                y_pred_binary_refined.append(0)

        f1 = f1_score(label_attack, y_pred_binary_refined)
        precision = precision_score(label_attack, y_pred_binary_refined)
        recall = recall_score(label_attack, y_pred_binary_refined)
        per_result.append([i, f1, precision, recall])
        
        del attack_x, attack_y, reframed_attack_dataset, X_test_permuted
        del mse_values, y_pred_binary, y_pred_binary_refined
        gc.collect()    

    df = pd.DataFrame(per_result, columns=["Feature", "f1", "pre", "rec"])
    df.to_excel(f"{model_name}_output_permutation.xlsx", index=False, engine='openpyxl')
    print(f"Feature permutation for model {model_name}' is calculated and saved!")

def max_perturbation_scale_check(target_data, data_y, model, model_name, scale):
    '''Max-min perturbation scale for LSTM, KNN, RFR, and LR.'''
	
    feature_list = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
    target_data_max = target_data.copy()
    target_data_min = target_data.copy()
    max_scale = scale
    min_scale = scale*-1
    if model_name == 'lstm':
        for i in feature_list:
            target_data_max[0, -1, i] *= (1+max_scale)
            target_data_min[0, -1, i] *= (1+min_scale)
    else:
        for i in feature_list:
            target_data_max[-1, i+561] *= (1+max_scale) # [1, 561:612], 612 = 12*51 = step_size * num of feature
            target_data_min[-1, i+561] *= (1+min_scale)
            
    predict_normal = model.predict(target_data)
    predict_max = model.predict(target_data_max)
    predict_min = model.predict(target_data_min)

    mse_normal = mean_squared_error(data_y, predict_normal[0])
    mse_max = mean_squared_error(data_y, predict_max[0])
    mse_min = mean_squared_error(data_y, predict_min[0])

    print('The mse diff for max is :', mse_normal-mse_max)
    print('The mse diff for min is :', mse_normal-mse_min)

def re_evaluate(model, attack_x, threshold, new_data_label, data_y, look_beyond):

    y_pred_attack = model.predict(attack_x)
    mse_values = [mean_squared_error([true], [pred]) for true, pred in zip(data_y, y_pred_attack)]
    y_pred_binary = [1 if mse > threshold else 0 for mse in mse_values]

    n = look_beyond # 20
    y_pred_binary_refined_after = []

    for i in range(len(y_pred_binary)):
        if y_pred_binary[i] == 1:
            if i + n < len(y_pred_binary):
                subsequent_anomalies = sum(y_pred_binary[i+1:i+n+1])
                if subsequent_anomalies >= n / 8:
                    y_pred_binary_refined_after.append(1)
                else:
                    y_pred_binary_refined_after.append(0)
            else:
                y_pred_binary_refined_after.append(y_pred_binary[i])
        else:
            y_pred_binary_refined_after.append(0)

    f1 = f1_score(new_data_label, y_pred_binary_refined_after)
    precision = precision_score(new_data_label, y_pred_binary_refined_after)
    recall = recall_score(new_data_label, y_pred_binary_refined_after)

    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return y_pred_binary_refined_after

def lstm_anomaly_score_check(lstm_config, overall_counter, data_x):
    '''Anomaly score check for lstm only'''

    attack_y = lstm_config["data_y"][overall_counter]
    y_pred_attack = lstm_config["model"].predict(data_x.reshape(1, 100, 51), verbose=0)
    y_pred_attack = y_pred_attack.reshape(-1)
    mse_values = mean_squared_error(attack_y, y_pred_attack)
    return mse_values

def anomaly_score_check(lstm_config, overall_counter, data_x):
    '''Anomaly score check for knn, rfr, lr'''
    
    data_x = series_to_supervised(data_x, 11, 1) # 11 = step_size - 1, as we are looking at 2d instead of 3d
    data_x = data_x.to_numpy()
    attack_y = lstm_config["data_y"][overall_counter]
    y_pred_attack = lstm_config["model"].predict(data_x)
    y_pred_attack = y_pred_attack.reshape(-1)
    mse_values = mean_squared_error(attack_y, y_pred_attack)
    return mse_values

def train_agent_with_clipping(tf_agent, experience):

    batch_size = 64
    trajectories = tf.data.Dataset.from_tensor_slices(experience).batch(batch_size)
    iterator = iter(trajectories)
    experience_batch = next(iterator)
    traj_shape = experience_batch.reward.shape[0]
    weights = tf.ones(shape=(traj_shape,), dtype=tf.float32)
    weights = tf.reshape(weights, (traj_shape,))
    with tf.GradientTape() as tape:
        train_loss = tf_agent._train(experience_batch, weights)

    gradients = tape.gradient(train_loss, tf_agent.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0) 
    tf_agent._optimizer.apply_gradients(zip(clipped_gradients, tf_agent.trainable_variables))
    
    return train_loss

def find_index(sensor_name):
    # List of sensor and actuator names
    index_list = ["FIT101", "LIT101", "MV101", "P101", "P102", "AIT201", "AIT202", "AIT203",
                  "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205", "P206", "DPIT301",
                  "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302",
                  "AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403", "P404",
                  "UV401", "AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502",
                  "FIT503", "FIT504", "P501", "P502", "PIT501", "PIT502", "PIT503", "FIT601",
                  "P601", "P602", "P603"]
    
    # Try to find the index of the sensor_name in the list
    try:
        return index_list.index(sensor_name)
    except ValueError:
        # If the sensor_name is not in the list, return -1
        return -1
    
def remove_element_by_value(input_list, value):
    # Try to remove the element if it exists in the list
    try:
        input_list.remove(value)
    except ValueError:
        # If the value is not in the list, pass
        pass
    return input_list

def mask_importance_list(attack_id, importance_list):
    '''Remove the features related to the attack itself'''

    removal_mapping = {
        "A3": 'LIT101',
        "A6": 'AIT202',
        "A7": 'LIT301',
        "A8": 'DPIT301',
        "A10": 'FIT401',
        "A11": 'FIT401',
        "A16": 'LIT301',
        "A19": 'AIT504',
        "A20": 'AIT504',
        "A21": 'LIT101',
        "A22": 'AIT502',
        "A23": 'DPIT301',
        "A25": 'LIT401',
        "A26": 'LIT301',
        "A27": 'LIT401',
        "A30": 'LIT101',
        "A31": 'LIT401',
        "A32": 'LIT301',
        "A33": 'LIT101',
        "A36": 'LIT101',
        "A37": 'FIT502',
        "A38": ['AIT402', 'AIT502'],
        "A39": ['FIT401', 'AIT502'],
        "A40": 'FIT401',
        "A41": 'LIT301'
    }

    # Processing based on attack_id
    if attack_id in removal_mapping:
        elements_to_remove = removal_mapping[attack_id]
        if isinstance(elements_to_remove, list):
            for element in elements_to_remove:
                importance_list = remove_element_by_value(importance_list, find_index(element))
        else:
            importance_list = remove_element_by_value(importance_list, find_index(elements_to_remove))
    elif attack_id in {"A1", "A2", "A4", "A5", "A9", "A12", "A13", "A14", "A15", "A17", "A18", "A24", "A28", "A29", "A34", "A35"}:
        pass # either the associated features are actuators or there is no actual attack happened

    return importance_list

def z_score_anomaly_detection_single_check(realtime_predictions, realtime_actuals, mu_e, sigma_e):
    """
    Detect anomalies in real-time data using z-scores and apply a time window refinement.
    """
    min_length = min(len(realtime_predictions), len(realtime_actuals))
    realtime_predictions = realtime_predictions[:min_length].copy()
    realtime_actuals = realtime_actuals[:min_length].copy()

    errors = np.abs(realtime_predictions - realtime_actuals)
    epsilon = 1e-10

    z_scores = np.abs(errors - mu_e) / (sigma_e + epsilon) # (2000, 51)
     
    return np.sum(z_scores[0])

def onedcnn_max_perturbation_scale_check(target_data, data_y, model, scale, mu_e, sigma_e):
	
    feature_list = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
    target_data_max = target_data.copy()
    target_data_min = target_data.copy()
    max_scale = scale
    min_scale = scale*-1

    for i in feature_list:
        target_data_max[256][i] *= (1+max_scale)
        target_data_min[256][i] *= (1+min_scale)

    augmented_normal = augment_data(target_data)
    augmented_max = augment_data(target_data_max)
    augmented_min = augment_data(target_data_min)

    eval_generator_normal = TimeSeriesGeneratorWithTargets(augmented_normal, 256, 256)
    eval_generator_max = TimeSeriesGeneratorWithTargets(augmented_max, 256, 256)
    eval_generator_min = TimeSeriesGeneratorWithTargets(augmented_min, 256, 256)

    predict_normal = model.predict(eval_generator_normal, verbose=0)
    predict_max = model.predict(eval_generator_max, verbose=0)
    predict_min = model.predict(eval_generator_min, verbose=0)

    zscore_normal = z_score_anomaly_detection_single_check(predict_normal, data_y, mu_e, sigma_e)
    zscore_max = z_score_anomaly_detection_single_check(predict_max, data_y, mu_e, sigma_e)
    zscore_min = z_score_anomaly_detection_single_check(predict_min, data_y, mu_e, sigma_e)

    print('The anomaly score diff for max is :', zscore_normal-zscore_max)
    print('The anomaly score diff for min is :', zscore_normal-zscore_min)
    print('Finished max min perturbation check, exiting the program..')
    sys.exit()

class TimeSeriesGeneratorWithTargets(Sequence):
    def __init__(self, data, sequence_length, batch_size):
        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.feature_count = data.shape[1] // 2  # Augmented data has doubled the feature count
        self.indices = np.arange(data.shape[0] - sequence_length - 1) # Adjust indices to account for the target data point

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, index):

        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # Generate sequences for each index and corresponding target
        X = np.zeros((len(batch_indices), self.sequence_length, self.data.shape[1]))
        y = np.zeros((len(batch_indices), self.feature_count))  # Target has the original number of features
        for i, row_idx in enumerate(batch_indices):
            X[i] = self.data[row_idx:row_idx+self.sequence_length]
            y[i] = self.data[row_idx+self.sequence_length, :self.feature_count]  # Extract only the original features
        
        return X, y

def z_score_anomaly_detection(realtime_predictions, realtime_actuals, mu_e, sigma_e, threshold, window_size, window_num_check):
    """Detect anomalies in real-time data using z-scores and apply a time window refinement."""
    errors = np.abs(realtime_predictions - realtime_actuals)
    epsilon = 1e-10
    z_scores = np.abs(errors - mu_e) / (sigma_e + epsilon)
    anomalies = []
    for idx in range(len(z_scores)):
        if np.sum(z_scores[idx]) > threshold:
            # Check if within the window there's a continuation of anomalies
            if idx + window_size < len(z_scores):
                window_anomalies = np.where(np.sum(z_scores[idx: idx + window_size], axis=1) > threshold, 1, 0)
                if np.sum(window_anomalies) >= window_size / window_num_check:
                    anomalies.append(1)
                else:
                    anomalies.append(0)
            else:
                anomalies.append(0)
        else:
            anomalies.append(0)
                
    return z_scores, anomalies

def build_model(step_size):
    '''Build 1dcnn model'''
    model = Sequential()
    input_shape = (step_size, 102)
    kernel_size = 2
    num_filters = 32 # Initial filter size
    # Add the convolutional layers
    for i in range(8): # 8 layers
        if i == 0:
            model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', input_shape=input_shape))
        else:
            model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        num_filters *= 2 # Double the filter size for the next layer
        
    model.add(Flatten())
    # Add dropout layer for regularization
    model.add(Dropout(0.5))
    # Add the fully connected layer
    model.add(Dense(51, activation='relu'))

    return model

def onedcnn_feature_permutation(model, scaled_attack, step_size, batch_size, mu_e, sigma_e, threshold, window_size, window_num_check, label_attack):

    sensor_feature_index = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
    per_result = []

    for i in sensor_feature_index:
        X_test_permuted = scaled_attack.copy()
        np.random.shuffle(X_test_permuted[:, i])
        augmented_data_attack = augment_data(X_test_permuted)
        eval_generator = TimeSeriesGeneratorWithTargets(augmented_data_attack, step_size, batch_size)

        eval_y = []
        for batch_num in range(len(eval_generator)):
            _, y_batch = eval_generator[batch_num]
            eval_y.append(y_batch)
        eval_y = np.concatenate(eval_y, axis=0)
        eval_y = eval_y.tolist()
        y_pred_eval = model.predict(eval_generator)

        _, detected_result_all = z_score_anomaly_detection(y_pred_eval, eval_y, mu_e, sigma_e, threshold, window_size, window_num_check)
        min_length = min(len(label_attack), len(detected_result_all))
        label_attack = label_attack[:min_length]
        detected_result_all = detected_result_all[:min_length]

        pre = precision_score(label_attack, detected_result_all)
        rec = recall_score(label_attack, detected_result_all)
        f1 = f1_score(label_attack, detected_result_all)

        print('F1 score: ',f1)
        print('Precision: ',pre)
        print('Recall: ',rec)

        per_result.append([i, f1, pre, rec])

    df = pd.DataFrame(per_result, columns=["Feature", "f1", "pre", "rec"])
    df.to_excel("onedcnn_output_permutation.xlsx", index=False, engine='openpyxl')
    print('The feature permutation for 1dcnn model is done, exiting the program..')
    sys.exit()

def stealthy_attack(model, model_name, values_attack, y_pred_binary_refined, step_size, look_back_window, n_obs, attack_y, threshold, attack_type, target_feature_index, no_of_input_features, n):
    '''Apply stealthy attack for lstm, knn, rfr, and lr.'''
    chunk_length = 104 # one quarter of the reading period

    values_attack_alter = values_attack.copy()
    start_index = find_consecutive_zeros(y_pred_binary_refined, chunk_length)

    chunk_data_to_process = values_attack_alter[step_size+start_index:step_size+start_index+chunk_length, target_feature_index].copy()

    if attack_type == 'delay':
        values_attack_alter[step_size+start_index:step_size+start_index+chunk_length, target_feature_index] = delay_attack_payload_generator(chunk_data_to_process, 2, True)
    else:
        converted_data = expedited_attack_payload_generator(chunk_data_to_process, 1, 1, 0.005)
        values_attack_alter[step_size+start_index:step_size+start_index+len(converted_data), target_feature_index] = converted_data

    attack_dataset_alter = series_to_supervised(values_attack, look_back_window, 1).values
    attack_x_alter = attack_dataset_alter[:,:n_obs]
    if model_name == 'lstm':
        attack_x_alter = attack_x_alter.reshape((attack_x_alter.shape[0], look_back_window, no_of_input_features))
    y_pred_attack_alter = model.predict(attack_x_alter)
    mse_values_alter = [mean_squared_error([true], [pred]) for true, pred in zip(attack_y, y_pred_attack_alter)]
    y_pred_binary_alter = [1 if mse > threshold else 0 for mse in mse_values_alter]

    y_pred_binary_refined_alter = []

    for i in range(len(y_pred_binary_alter)):
        if y_pred_binary_alter[i] == 1:  # Current data point is initially marked as an anomaly
            if i + n < len(y_pred_binary_alter):  # Check if we have enough subsequent data points
                # Count how many of the next n data points are anomalies
                subsequent_anomalies = sum(y_pred_binary_alter[i+1:i+n+1])
                if subsequent_anomalies >= n / 8: # 8
                    y_pred_binary_refined_alter.append(1)
                else:
                    y_pred_binary_refined_alter.append(0)
            else:
                y_pred_binary_refined_alter.append(y_pred_binary_alter[i])
        else:
            y_pred_binary_refined_alter.append(0)

    counter = 0
    counter_normal = 0
    for i in range(len(y_pred_binary_refined)):
        if y_pred_binary_refined[i] != y_pred_binary_refined_alter[i]:
            counter += 1
        else:
            counter_normal += 1

    print(f'The detection result changed percentage is: {counter/chunk_length*100} %')
    print(f'The stealthy attack for {attack_type} is conducted, exiting the program..')
    sys.exit()