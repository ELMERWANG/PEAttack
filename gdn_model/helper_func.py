import numpy as np
import torch
from util.data import get_err_median_and_iqr
import pandas as pd
import random



def various_label_check(predicted_label, real_label):
    '''Give the predicted label list, and the real labels contains various labels, this function can print how many type of attacks are detected, also print in percentage'''

    # Ensure the lists are of the same length
    if len(predicted_label) != len(real_label):
        print('len(predicted_label): ', len(predicted_label))
        print('len(real_label): ', len(real_label))
        raise ValueError("The two lists must be of the same length.")

    a_counts = {} # Using a dictionary to track the counts of each "A" prefixed item in real_label
    total_as = 0
    total_ones_for_as = 0

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

    over_under_flow_attack_id_list = ['A1', 'A2', 'A3', 'A4', 'A6', 'A7', 'A8', 'A10', 'A11', 'A13', 'A14', 'A16', 'A17', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41']
    for key, values in a_counts.items():
        if key in over_under_flow_attack_id_list:
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


def get_attack_data(model_name, attack_id, step_size, passivity):
    '''This function returns the complete data in the attack dataset for the given attack ID, step_size is used to preserve the data ahead of the complete data chunk.
    output: dataset of only the attack instance ID
    '''
    new_attack_dataset = None
    new_attack_label = None
    new_attack_label_various = None

    if model_name == 'GDN':
        test_dataset = pd.read_csv(f'./data/swat/test_all_label.csv', sep=',', index_col=0)

        test_labels_list_various = test_dataset['attack'].tolist()
        test_dataset = test_dataset.reset_index(drop=True) # replace the time index as sequence number
        test_dataset = test_dataset.iloc[:, :-1]

        label_attack = [1 if isinstance(x, str) and x.startswith('A') else 0 for x in test_labels_list_various] # convert to 1s and 0s only
        test_dataset['attack'] = label_attack # original attack column is various label, now replace it with 1 or 0 label
        

    attack_start_index, attack_end_index = find_indices(test_labels_list_various, attack_id)

    print('Found attack start index is: ', attack_start_index)
    print('Found attack end index is: ', attack_end_index)

    attack_start_index -= 3 # as we need 3 extra for the score smooth function
    attack_end_index += 1 # we need add one, as the last index is not counted.

    if passivity == True:
        attack_end_index += step_size # preserve enough data after the end index for passivity check

    if attack_start_index - step_size >= 0:
        new_attack_dataset = test_dataset[attack_start_index-step_size:attack_end_index]
        new_attack_label = label_attack[attack_start_index-step_size:attack_end_index]
        new_attack_label_various = test_labels_list_various[attack_start_index-step_size:attack_end_index]
    else:
        raise ValueError('No extra step size data can be found')
    
    return new_attack_dataset, new_attack_label, new_attack_label_various

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


def altered_result_check_ppo(test_result, single_check):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)
    all_scores =  None
    feature_num = np_test_result.shape[-1]
    para_list_all_test = [0.0340476930141449, 0.030281655490398407, 0.017242267727851868, 0.03566721826791763, 0.008933395147323608, 0.019947320222854614, 0.01802733540534973, 0.004210375249385834, 0.0018786638975143433, 0.0009308904409408569, 2.305459499359131, 2.696294814348221, 0.30325765907764435, 0.3488807901740074, 0.037966009229421616, 0.06902330042794347, 0.0349506139755249, 0.03651968389749527, 0.019668519496917725, 0.01806042343378067, 0.016895443201065063, 0.013939663767814636, 0.009604275226593018, 0.00045733898878097534, 0.016164630651474, 0.025830477476119995, 0.014166727662086487, 0.0045279860496521, 0.01459529995918274, 0.016994863748550415, 0.005035564303398132, 0.0025365129113197327, 0.011150360107421875, 0.01804201304912567, 0.006132751703262329, 0.03951841592788696, 0.029582351446151733, 0.054810985922813416, 0.0049295127391815186, 0.01884138584136963, 0.010411947965621948, 0.010396033525466919, 0.0063134729862213135, 0.0013577714562416077, 0.010552048683166504, 0.00142601877450943, 0.006651431322097778, 0.0014042407274246216, 0.011830419301986694, 0.010372772812843323, 0.021661758422851562, 0.024722829461097717, 0.15903190337121487, 0.10623387157102115, 0.01512196660041809, 0.00535312294960022, 0.0380932092666626, 0.03692331910133362, 0.005101799964904785, 0.0005407184362411499, 0.012979954481124878, 0.0010611414909362793, 0.009819209575653076, 0.0007431879639625549, 0.011462032794952393, 0.0003506690263748169, 0.010029464960098267, 0.00783132016658783, 0.0338405966758728, 0.046367138624191284, 0.10952365421690047, 0.07713164482265711, 0.06852343678474426, 0.15918677300214767, 0.004874696023762226, 0.004463938530534506, 0.01105421781539917, 0.007791385054588318, 0.01532965898513794, 0.012232720851898193, 0.014491647481918335, 0.01453825831413269, 0.046957194805145264, 0.014056220650672913, 0.014759957790374756, 0.013457223773002625, 0.013816595077514648, 0.0011324584484100342, 0.00956752896308899, 0.009589701890945435, 0.043571799993515015, 0.052660245448350906, 0.012064218521118164, 0.0054275840520858765, 0.011920935685338918, 0.0021867416508030146, 0.006061851978302002, 0.00042732805013656616, 0.010233476758003235, 0.0046311840415000916, 0.009731411933898926, 8.412450551986694e-05]

    for i in range(feature_num):
        if single_check:
            test_re_list = np_test_result[:2,-4:,i]
        else:
            test_re_list = np_test_result[:2,:,i]

        test_predict, test_gt = test_re_list
        n_err_mid = para_list_all_test[2*i]
        n_err_iqr = para_list_all_test[2*i+1]

        test_delta = np.abs(np.subtract(
                            np.array(test_predict).astype(np.float64), 
                            np.array(test_gt).astype(np.float64)
                        ))
        epsilon=1e-2
        err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)
        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3

        for i in range(before_num, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

        if all_scores is None:
            all_scores = smoothed_err_scores
        else:
            all_scores = np.vstack((
                all_scores,
                smoothed_err_scores
            ))

    total_features = all_scores.shape[0]
    topk = 1
    topk_indices = np.argpartition(all_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    total_topk_err_scores = []
    total_topk_err_scores = np.sum(np.take_along_axis(all_scores, topk_indices, axis=0), axis=0)
    threshold = 35.458921639159946

    if single_check:
        if total_topk_err_scores[-1] > threshold:
            return 1, total_topk_err_scores[-1]
        else:
            return 0, total_topk_err_scores[-1]
        

def delay_attack_payload_generator(data, change_level, trim_to_input):
    '''
    Generate attack payload in a delayed mode
    Input: 
    - data: data stream, type: list, if input is df, use tolist()
    - change level: (int), from 1 to n. 1 means, insert 1 after each 1, 3 means insert 1 after each 3.
    - trim to input: after delay attack, the generated data length is more than input, if true, keep the output with the same length as input

    Output:
    - converted data: modified data stream according to the change ratio.
    '''
    converted_data = [data[0]] # initialize with the first value in the data stream
    accumulate_res = 0 # accumulated residual for each data
    for i in range(1, len(data)-1): # calculate diff beforehand, but can be easily converted to online mode
        diff = data[i]-data[i-1]
        converted_data.append(converted_data[-1] + (1-1 / change_level) * diff)
        accumulate_res += 1 / change_level * diff

        if (len(converted_data)-1) % change_level == 0: # every change_level insert 1 data
            converted_data.append(accumulate_res+converted_data[-1])
            accumulate_res = 0

    if accumulate_res != 0:
        converted_data.append(accumulate_res+converted_data[-1]) # if there is any residual left, and the left number of data is not enough for one more round, add to the end of the list

    if trim_to_input:
        converted_data = converted_data[0:len(data)]

    return converted_data


def expedited_attack_payload_generator(ori_data, change_level, drop_amount, rand_coeff):
    '''
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
        ori_counter += drop_amount

    return converted_data


def do_prediction(target_model, data_loader, env_config):

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []
    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    target_model.eval()
    counter = 0

    while counter < len(data_loader.dataset):

        x_data = data_loader.dataset[counter][0].to(env_config['device']).float()
        x_data = x_data.reshape(1, x_data.shape[0], x_data.shape[1])
        y_data = data_loader.dataset[counter][1].to(env_config['device']).float()
        y_data = y_data.reshape(1, y_data.shape[0])
        labels = data_loader.dataset[counter][2].to(env_config['device']).float()
        labels = labels.view(torch.Size([1]))
        edge_index = data_loader.dataset[counter][3].to(env_config['device']).float()
        edge_index = edge_index.reshape(1, edge_index.shape[0], edge_index.shape[1])

        with torch.no_grad():

            predicted = target_model(x_data, edge_index).float().to(env_config['device'])
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y_data
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y_data), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        counter += 1

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()


    test_result = [test_predicted_list, test_ground_list, test_labels_list]
    np_test_result = np.array(test_result)
    all_scores =  None
    feature_num = np_test_result.shape[-1]
            
    for i in range(feature_num):

        test_re_list = np_test_result[:2,:,i]
        test_predict, test_gt = test_re_list
        n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
        test_delta = np.abs(np.subtract(
                            np.array(test_predict).astype(np.float64), 
                            np.array(test_gt).astype(np.float64)
                        ))
        epsilon=1e-2
        err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) + epsilon)
        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3

        for j in range(before_num, len(err_scores)):
            smoothed_err_scores[j] = np.mean(err_scores[j-before_num:j+1])

        if all_scores is None:
            all_scores = smoothed_err_scores
        else:
            all_scores = np.vstack((
                all_scores,
                smoothed_err_scores
            ))

    total_features = all_scores.shape[0]
    topk = 1
    topk_indices = np.argpartition(all_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    total_topk_err_scores = []
    total_topk_err_scores = np.sum(np.take_along_axis(all_scores, topk_indices, axis=0), axis=0)
    threshold = 35.458921639159946
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    return pred_labels

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

def mask_importance_list(attack_id):

    importance_list = [47, 46, 16, 27, 35, 26, 39, 38, 44, 37, 17, 34, 0, 1, 5, 6, 7, 8, 18, 25, 28, 36, 40, 41, 45]
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


def find_consecutive_zeros(lst, chunk_len):
    
    sequence_length = chunk_len # Length of the sequence to find
    for i in range(len(lst) - sequence_length + 1):
        # Check if the next sequence_length elements are all zeros
        if all(lst[i + j] == 0 for j in range(sequence_length)):
            return i

    return -1 # Return -1 if no such sequence is found

def altered_result_check_action_scale(test_result, single_check):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)
    all_scores =  None
    feature_num = np_test_result.shape[-1]
    para_list_all_test = [0.0340476930141449, 0.030281655490398407, 0.017242267727851868, 0.03566721826791763, 0.008933395147323608, 0.019947320222854614, 0.01802733540534973, 0.004210375249385834, 0.0018786638975143433, 0.0009308904409408569, 2.305459499359131, 2.696294814348221, 0.30325765907764435, 0.3488807901740074, 0.037966009229421616, 0.06902330042794347, 0.0349506139755249, 0.03651968389749527, 0.019668519496917725, 0.01806042343378067, 0.016895443201065063, 0.013939663767814636, 0.009604275226593018, 0.00045733898878097534, 0.016164630651474, 0.025830477476119995, 0.014166727662086487, 0.0045279860496521, 0.01459529995918274, 0.016994863748550415, 0.005035564303398132, 0.0025365129113197327, 0.011150360107421875, 0.01804201304912567, 0.006132751703262329, 0.03951841592788696, 0.029582351446151733, 0.054810985922813416, 0.0049295127391815186, 0.01884138584136963, 0.010411947965621948, 0.010396033525466919, 0.0063134729862213135, 0.0013577714562416077, 0.010552048683166504, 0.00142601877450943, 0.006651431322097778, 0.0014042407274246216, 0.011830419301986694, 0.010372772812843323, 0.021661758422851562, 0.024722829461097717, 0.15903190337121487, 0.10623387157102115, 0.01512196660041809, 0.00535312294960022, 0.0380932092666626, 0.03692331910133362, 0.005101799964904785, 0.0005407184362411499, 0.012979954481124878, 0.0010611414909362793, 0.009819209575653076, 0.0007431879639625549, 0.011462032794952393, 0.0003506690263748169, 0.010029464960098267, 0.00783132016658783, 0.0338405966758728, 0.046367138624191284, 0.10952365421690047, 0.07713164482265711, 0.06852343678474426, 0.15918677300214767, 0.004874696023762226, 0.004463938530534506, 0.01105421781539917, 0.007791385054588318, 0.01532965898513794, 0.012232720851898193, 0.014491647481918335, 0.01453825831413269, 0.046957194805145264, 0.014056220650672913, 0.014759957790374756, 0.013457223773002625, 0.013816595077514648, 0.0011324584484100342, 0.00956752896308899, 0.009589701890945435, 0.043571799993515015, 0.052660245448350906, 0.012064218521118164, 0.0054275840520858765, 0.011920935685338918, 0.0021867416508030146, 0.006061851978302002, 0.00042732805013656616, 0.010233476758003235, 0.0046311840415000916, 0.009731411933898926, 8.412450551986694e-05]

    for i in range(feature_num):
        if single_check:
            test_re_list = np_test_result[:2,-4:,i]
        else:
            test_re_list = np_test_result[:2,:,i]

        test_predict, test_gt = test_re_list
        n_err_mid = para_list_all_test[2*i]
        n_err_iqr = para_list_all_test[2*i+1]

        test_delta = np.abs(np.subtract(
                            np.array(test_predict).astype(np.float64), 
                            np.array(test_gt).astype(np.float64)
                        ))
        epsilon=1e-2

        err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) + epsilon)
        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3

        for j in range(before_num, len(err_scores)):
            smoothed_err_scores[j] = np.mean(err_scores[j-before_num:j+1])

        if all_scores is None:
            all_scores = smoothed_err_scores
        else:
            all_scores = np.vstack((
                all_scores,
                smoothed_err_scores
            ))

    total_features = all_scores.shape[0]
    topk = 1
    topk_indices = np.argpartition(all_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    total_topk_err_scores = []
    total_topk_err_scores = np.sum(np.take_along_axis(all_scores, topk_indices, axis=0), axis=0)

    return total_topk_err_scores[-1]