import mylibrary
import sys
import numpy as np
import pandas as pd
import help_func
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.ensemble import RandomForestRegressor

'''Include the LSTM, KNN, RFR, LR's train, test, max-min perturbation test, stealthy attack, and feature permutation.'''

seed_num = 67999 # fix random seed for reproducibility
tf.random.set_seed(seed_num)
tf.compat.v1.set_random_seed(seed_num)
tf.experimental.numpy.random.seed(seed_num)
np.random.seed(seed_num)


model_name = sys.argv[1]
model_mode = sys.argv[2]
evaluate_mode = sys.argv[3]
attack_type = sys.argv[4] # default value: ignore

target_feature = 'LIT301'
look_beyond = 20

if model_mode == 'test':
	if model_name == 'lstm':
		batch_size = 256
		step_size = 100
		threshold = 0.305 # 0.305
		print(f"MSE Threshold: {threshold}")
		model = mylibrary.load_model('for_ppo_LSTM_70_neuron100_batch256_step100')

	elif model_name == 'knn':
		step_size = 12
		threshold = 0.3155
		print(f"MSE Threshold: {threshold}")
		model = mylibrary.load_variable('knn_2')

	elif model_name == 'rfr':
		step_size = 12
		threshold = 0.29
		print(f"MSE Threshold: {threshold}")
		model = mylibrary.load_variable('RFR_12_51_67999')

	elif model_name == 'lr':
		step_size = 12
		threshold = 0.1
		print(f"MSE Threshold: {threshold}")
		model = mylibrary.load_variable('LR')
else:
	if model_name == 'lstm':
		batch_size = 256
		epochs = 70
		neuron = 100
		step_size = 100
	elif model_name == 'knn':
		step_size = 12
		n_neighbors=2

	elif model_name == 'rfr':
		step_size = 12
		n_estimators = 51
		random_state = 67999
		n_jobs = -1

	elif model_name == 'lr':
		step_size = 12

train_test_ratio = 0.8

dataset_normal = pd.read_csv('./dataset/swat/train.csv', sep=',', index_col=0)
dataset_attack = pd.read_csv('./dataset/swat/test_all_label.csv', sep=',', index_col=0)

target_feature_index = dataset_attack.columns.get_loc(target_feature)

test_labels_list_various = dataset_attack['attack'].tolist()
label_attack_0_1 = [1 if isinstance(x, str) and x.startswith('A') else 0 for x in test_labels_list_various] # convert to 1s and 0s only

# values = dataset.values
values_normal = dataset_normal.drop('attack', axis=1).values
label_normal = np.asarray(dataset_normal['attack'])

values_attack = dataset_attack.drop('attack', axis=1).values
label_attack = label_attack_0_1
label_attack = label_attack[step_size:].copy()

look_back_window = step_size # specify the number of time steps
no_of_input_features = values_normal.shape[1]

training_size = round(train_test_ratio * values_normal.shape[0])

reframed_normal_dataset = mylibrary.series_to_supervised(values_normal, look_back_window, 1)
reframed_attack_dataset = mylibrary.series_to_supervised(values_attack, look_back_window, 1)

normal_values = reframed_normal_dataset.values
attack_values = reframed_attack_dataset.values

n_obs = look_back_window * no_of_input_features

train_x = normal_values[:training_size,:n_obs]
train_y = normal_values[:training_size,n_obs:] # define the target feature for prediction as label
test_x = normal_values[training_size:,:n_obs]
test_y = normal_values[training_size:,n_obs:]

attack_x = attack_values[:,:n_obs]
attack_y = attack_values[:,n_obs:]


if model_name == 'lstm':
	# reshape input to be 3D [samples, timesteps, features]
	train_x = train_x.reshape((train_x.shape[0], look_back_window, no_of_input_features))
	test_x = test_x.reshape((test_x.shape[0], look_back_window, no_of_input_features))
	attack_x = attack_x.reshape((attack_x.shape[0], look_back_window, no_of_input_features))


if model_mode == 'train':
	if model_name == 'lstm':
		model = Sequential()
		model.add(LSTM(neuron, input_shape=(train_x.shape[1], train_x.shape[2])))
		model.add(Dense(train_x.shape[2])) 
		model.compile(optimizer='adam', loss='mse')
		history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=1, shuffle=False)
		model.summary()

		pyplot.plot(history.history['loss'], label='train')
		pyplot.plot(history.history['val_loss'], label='test')
		pyplot.legend()

		model_name = 'for_ppo_LSTM_' + str(epochs) + '_' + 'neuron' + str(neuron) + '_' + 'batch' + str(batch_size) + '_' + 'step' + str(step_size)
		history_name = 'for_ppo_LSTM_history_' + str(epochs) + '_' + 'neuron' + str(neuron) + '_' + 'batch' + str(batch_size) + '_' + 'step' + str(step_size)
		
		# figname = '{}.png'.format(model_name)
		# pyplot.savefig(figname)

		mylibrary.save_model(model, model_name, True)
		mylibrary.save_variable(history.history, history_name, True)
		print(model_name)
		print(np.mean(history.history['val_loss']))

	elif model_name == 'knn':
		model = KNeighborsRegressor(n_neighbors=2)
		model.fit(train_x, train_y)

	elif model_name == 'rfr':
		model = RandomForestRegressor(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs)
		model.fit(train_x, train_y)

	elif model_name == 'lr':
		model = LinearRegression()
		model.fit(train_x, train_y)

y_pred_binary_refined = help_func.re_evaluate(model, attack_x, threshold, label_attack, attack_y, look_beyond)

if evaluate_mode == 'labelcheck':
	help_func.various_label_check(y_pred_binary_refined, test_labels_list_various[step_size:]) # check detection condition for each attack ID
elif evaluate_mode == 'permutation':
	help_func.feature_permutation(model, model_name, label_attack, values_attack, threshold, n_obs, look_back_window, no_of_input_features)
elif evaluate_mode == 'maxmin':
	help_func.max_perturbation_scale_check(train_x[0:1], train_y[0], model, model_name, 3) # Test the max-min limit for the action space
elif evaluate_mode == 'stealthy':
	if attack_type not in ['delayed', 'expedited']:
		raise ValueError('Attack type is provided incorrectly.')
	help_func.stealthy_attack(model, model_name, values_attack, y_pred_binary_refined, step_size, look_back_window, n_obs, attack_y, threshold, attack_type, target_feature_index, no_of_input_features, look_beyond)
