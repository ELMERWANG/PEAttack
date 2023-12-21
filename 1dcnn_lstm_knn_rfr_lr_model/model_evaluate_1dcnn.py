
from mylibrary import *
import numpy as np
from pandas import concat
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from help_func import *

'''Include 1DCNN model train, test, max-min perturbation test, stealthy attack, and feature permutation'''


start_time = time.time()

seed = 6666
tf.random.set_seed(seed)

model_name = sys.argv[1]
model_mode = sys.argv[2]
evaluate_mode = sys.argv[3]
attack_type = sys.argv[4] # default value: ignore


epochs = 100
batch_size = 256
step_size = 256
train_test_ratio = 0.8
mu_e = load_variable('1dcnn_mu') # mean of prediction residual on the evaluation set
sigma_e = load_variable('1dcnn_sigma') # variability of prediction residual on the evaluation set
threshold = load_variable('1dcnn_threshold') # threshold is determined using the max anomaly score after the prediction residual is normalized using mu and sigma with a minor adjustment
window_num_check = 2
window_size = 2000
target_feature = 'LIT301'

dataset_normal = pd.read_excel('./dataset/swat/swat_normal.xlsx', header=0, index_col=0) # swat_normal_v0_16000_trimmed_w_label
dataset_attack = pd.read_excel('./dataset/swat/swat_attack.xlsx', header=0, index_col=0)

target_feature_index = dataset_attack.columns.get_loc(target_feature)
values_normal = dataset_normal.drop('label', axis=1).values
label_normal = np.asarray(dataset_normal['label'])
values_attack = dataset_attack.drop('label', axis=1).values
label_attack = np.asarray(dataset_attack['label'])
label_attack_various = label_attack.tolist()[batch_size:]
label_attack = [1 if isinstance(item, str) and item.startswith('A') else 0 for item in label_attack_various]

# ensure all data is float
values_normal = values_normal.astype('float32')
values_attack = values_attack.astype('float32')

scaler = MinMaxScaler()
scaled_normal = scaler.fit_transform(values_normal)
scaled_attack = scaler.transform(values_attack)
scaled_attack_alter = scaled_attack.copy() # for stealthy attack

augmented_data_normal = augment_data(scaled_normal)
augmented_data_attack = augment_data(scaled_attack)

no_of_input_features = augmented_data_normal.shape[1]
training_size = round(train_test_ratio * augmented_data_normal.shape[0])
train_data = augmented_data_normal[:training_size]
test_data = augmented_data_normal[training_size:]

train_generator = TimeSeriesGeneratorWithTargets(train_data, step_size, batch_size)
test_generator = TimeSeriesGeneratorWithTargets(test_data, step_size, batch_size)
eval_generator = TimeSeriesGeneratorWithTargets(augmented_data_attack, step_size, batch_size) # Creating time series generators

if model_mode == 'train':
    patience = 15
    learning_rate = 0.00001
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99)
    print('Start training model..')
    model = build_model(step_size)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=patience, restore_best_weights=True)
    history = model.fit(train_generator, batch_size=batch_size, validation_data=test_generator, epochs=epochs, verbose=1, shuffle=False, callbacks=[es])
    model_name = '1dcnn_zscore_epo_' + str(epochs) + '_batch' + str(batch_size) + '_step' + str(step_size) + '_patience' + str(patience) + '_lr' + str(learning_rate)
    history_name = '1dcnn_zscore_history_epo_' + str(epochs) + '_batch' + str(batch_size) + '_step' + str(step_size) + '_patience' + str(patience) + '_lr' + str(learning_rate)

    save_model(model, model_name, True)
    save_variable(history.history, history_name, True)
    print(model_name)
    print(np.mean(history.history['val_loss']))
    print('Model training finished.')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

else:
    model = load_model('1dcnn_zscore_epo_100_batch256_step256_patience15_fullepoch')

eval_y = []
for batch_num in range(len(eval_generator)):
    x_batch, y_batch = eval_generator[batch_num]
    eval_y.append(y_batch)
eval_y = np.concatenate(eval_y, axis=0)


if evaluate_mode == 'maxmin':
    onedcnn_max_perturbation_scale_check(scaled_attack[0:step_size+window_size+2], eval_y[0:window_size], model, 3, mu_e, sigma_e)

if evaluate_mode == 'permutation':
    onedcnn_feature_permutation(model, scaled_attack, step_size, batch_size, mu_e, sigma_e, threshold, window_size, window_num_check, label_attack)

eval_y = eval_y.tolist()
y_pred_eval = model.predict(eval_generator)

chunk_length = 1040 # one quarter of the reading period
_, detected_result_all = z_score_anomaly_detection(y_pred_eval, eval_y, mu_e, sigma_e, threshold, window_size, window_num_check)


if evaluate_mode == 'stealthy':
    if attack_type not in ['delayed', 'expedited']:
        raise ValueError('Attack type is provided incorrectly.')
        
    start_index = find_consecutive_zeros(detected_result_all, chunk_length)

    selected_chunk = scaled_attack_alter[step_size+start_index:step_size+start_index+chunk_length, target_feature_index].copy()
    if attack_type == 'delayed':
        scaled_attack_alter[step_size+start_index:step_size+start_index+chunk_length, target_feature_index] = delay_attack_payload_generator(selected_chunk, 2, True)
    else:
        converted_data = expedited_attack_payload_generator(selected_chunk, 1, 1, 0.005)
        scaled_attack_alter[step_size+start_index:step_size+start_index+len(converted_data), target_feature_index] = converted_data

    augmented_data_attack_altered = augment_data(scaled_attack_alter.copy())
    eval_generator_altered = TimeSeriesGeneratorWithTargets(augmented_data_attack_altered, step_size, batch_size)

    y_pred_eval_altered = model.predict(eval_generator_altered)
    _, detected_result_all_altered = z_score_anomaly_detection(y_pred_eval_altered, eval_y, mu_e, sigma_e, threshold, window_size, window_num_check)

    chunk_data = detected_result_all[0:chunk_length]
    chunk_data_altered = detected_result_all_altered[0:chunk_length]
    counter = 0
    counter_normal = 0
    for i in range(len(chunk_data)):
        if chunk_data[i] != chunk_data_altered[i]:
            counter += 1 # number of data which the prediction result is changed
        else:
            counter_normal += 1 # number of data which the prediction result is not changed

    print(f'The percentage that the data prediction result is changed: {counter/chunk_length*100} %')
    print(f'Stealthy attack for {attack_type} is conducted, exiting program..')
    sys.exit()


min_length = min(len(label_attack), len(detected_result_all))
label_attack = label_attack[:min_length]
label_attack_various = label_attack_various[:min_length]
detected_result_all = detected_result_all[:min_length]

pre = precision_score(label_attack, detected_result_all)
rec = recall_score(label_attack, detected_result_all)
f1 = f1_score(label_attack, detected_result_all)

print('F1 score: ',f1)
print('Precision: ',pre)
print('Recall: ',rec)

print('The time spent is: {:.2f} s'.format(time.time()-start_time))



