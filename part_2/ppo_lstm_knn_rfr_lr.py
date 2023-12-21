from mylibrary import *
import sys
import numpy as np
import pandas as pd
from pandas import concat
import math
import time
from pathlib import Path
from matplotlib import pyplot as plt
import random
from sklearn.metrics import mean_squared_error
import help_func
import numpy as np
import tensorflow as tf
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.networks import value_network


class CustomEnvironment(py_environment.PyEnvironment):

    def __init__(self, model_name, passivity_check, sub_check_total):
        self.action_boundry = 1.0
        self.total_num_of_feature = 51
        self.model_name = model_name
        if self.model_name == 'lstm':
            data_shape = 100
        else:
            data_shape = 12
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self.action_boundry, maximum=self.action_boundry, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(data_shape, 51), dtype=np.float64, name='observation')
        self._current_step = 0
        self.model_config = None
        self.anomaly_score_list = []
        self.one_episode_anomaly_score_list = []
        self.feature_importance_list = []
        self.action_sequence = []
        self.best_action_sequence = []
        self.current_state = np.zeros((data_shape, 51), dtype=np.float32)
        self.target_x = np.zeros((data_shape, 51), dtype=np.float32)
        self.first_step_score = 0
        self.overall_counter = -1
        self.attack_data_all = np.zeros((1,51), dtype=np.float32)
        self.attack_data_all_original = np.zeros((1,51), dtype=np.float32)
        self.attack_data_label = []
        self.score_change_across_episode = []
        self.episode_end = False
        self.sub_check_total = sub_check_total
        self.sub_check_counter = 0
        self.best_data_to_substitute = np.zeros((data_shape,51), dtype=np.float32) # this record the best action from the sub check loop
        self.best_score_in_sub_check_loop = np.inf
        self.avg_perturbation_per_data = []
        self.EFR_rate_summary = []
        self.sub_loop_end = True
        self.evasion_severity_factor = 0.5
        self.passivity_check = passivity_check
        self.exploration_end = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def EFR_rate_calculation(self):
        a = self.evasion_severity_factor
        ncf = len(self.feature_importance_list)
        ntf = self.total_num_of_feature
        pertub_sum_per_feature = sum(element / self.action_boundry for element in self.best_action_sequence)
        result = ((a * ncf) / ntf) + ((1 - a) * pertub_sum_per_feature / ncf)
        return result

    def _reset(self):
        self._current_step = 0

        if self.episode_end == False and self.sub_loop_end == True: 
            self.set_initial_state()
        if self.sub_loop_end == False:
            self.target_x = self.target_x_ori.copy()
            self.current_state = self.target_x_ori.copy()
        self.one_episode_anomaly_score_list = []
        self.action_sequence = []
        return ts.restart(self.current_state)

    def set_initial_state(self):
        ''' Used to set the initial state value for starting each new episode'''

        while True:
            self.overall_counter += 1
            if self.overall_counter + 1 == len(model_config["data_y"]):
                self.overall_counter = 0
                self.attack_data_all = self.attack_data_all_original.copy()
                self.exploration_end = True
            
            self.target_x = self.attack_data_all.copy()[self.overall_counter:self.overall_counter+self.model_config["step_size"]]
            if self.model_name == 'lstm':
                anomaly_score = help_func.lstm_anomaly_score_check(self.model_config, self.overall_counter, self.target_x.copy())
            else:
                anomaly_score = help_func.anomaly_score_check(self.model_config, self.overall_counter, self.target_x.copy())
            if  anomaly_score > self.model_config["threshold"]:
                self.target_x_ori = self.target_x.copy()
                self.current_state = self.target_x.copy()
                self.first_step_score = anomaly_score # this is the initial score without perturbation added
                self.sub_loop_end = False
                break

    def reset_sub_loop(self):
        self.sub_check_counter = 0
        self.best_score_in_sub_check_loop = np.inf
        self.best_data_to_substitute = np.zeros((100,51), dtype=np.float32) # this record the best action from the sub check loop
        self.sub_loop_end = True
        self.best_action_sequence = []


    def _step(self, action):

        if self._current_step >= len(self.feature_importance_list):
            return self.reset()

        action = tf.tanh(action)
        action_range = (self._action_spec.maximum - self._action_spec.minimum) / 2.0
        action_center = (self._action_spec.maximum + self._action_spec.minimum) / 2.0
        action = action_range * action + action_center

        self.action_sequence.append(np.abs(action.numpy()[0])/self.action_boundry)
        assessment_score = self.evaluate(action) # Compute the assessment score
        reward = -assessment_score
        self._current_step += 1
        if self._current_step == len(self.feature_importance_list):
            self.anomaly_score_list.append(self.one_episode_anomaly_score_list)

            if assessment_score < self.best_score_in_sub_check_loop:
                self.best_score_in_sub_check_loop = assessment_score
                self.best_data_to_substitute = self.target_x.copy()
                self.best_action_sequence = self.action_sequence.copy()

            self.sub_check_counter += 1

            if self.sub_check_counter == self.sub_check_total:
                if self.first_step_score - self.best_score_in_sub_check_loop > 0: # only we apply the perturbation if the score change > 0
                    self.attack_data_all[self.overall_counter:self.overall_counter + self.model_config["step_size"]] = self.best_data_to_substitute.copy() # save the changes on action back to the overall data for next episode 
                    
                if self.exploration_end == False:
                    self.avg_perturbation_per_data.append(np.sum(self.best_action_sequence)/len(self.best_action_sequence))
                    self.EFR_rate_summary.append(self.EFR_rate_calculation())

                # -------------------------Passivity calculation---------------------------
                if self.passivity_check == True:
                  
                    self.passivity_data_x_before = self.attack_data_all_original.copy()[self.overall_counter : self.overall_counter+2*self.model_config["step_size"]+round(0.2*self.model_config["step_size"])]
                    self.passivity_data_x_after = self.attack_data_all.copy()[self.overall_counter:self.overall_counter+2*self.model_config["step_size"]+round(0.2*self.model_config["step_size"])]
                    score_before_add_action = self.passivty_evaluate(self.passivity_data_x_before.copy())
                    score_after_add_action = self.passivty_evaluate(self.passivity_data_x_after.copy())
                    diff = score_before_add_action - score_after_add_action

                    plt.figure(figsize=(10, 6))
                    plt.plot(diff)
                    plt.title("Score diff before and after adding perturbation at 1th timestamp")
                    plt.xlabel("Anomaly score")
                    plt.ylabel("Anomaly score diff")
                    plt.grid(True)
                    plt.show()
                    
                    print('Passivity calculated, exiting the program..')
                    sys.exit()
                # ---------------------------End Passivity Check----------------------------
                
                self.score_change_across_episode.append(self.first_step_score - self.best_score_in_sub_check_loop)
                self.reset_sub_loop()

            self.episode_end = True
            return ts.termination(self.current_state, reward)
        else:
            return ts.transition(self.current_state, reward, discount=tf.constant(0.99))
        

    def evaluate(self, action):

        self.target_x[-1][self.feature_importance_list[self._current_step]] *= (1 + action) 
        self.current_state[-1][self.feature_importance_list[self._current_step]] *= (1 + action)
        if self.model_name == 'lstm':
            anomaly_score = help_func.lstm_anomaly_score_check(self.model_config, self.overall_counter, self.target_x.copy())
        else:
            anomaly_score = help_func.anomaly_score_check(self.model_config, self.overall_counter, self.target_x.copy())
        self.one_episode_anomaly_score_list.append(anomaly_score)

        return anomaly_score
    

    def passivty_evaluate(self, data_for_model):
        '''Return a list contains current and n future timestamp anomaly score'''

        attack_y = self.model_config["data_y"][self.overall_counter:self.overall_counter + self.model_config["step_size"]+round(0.2*self.model_config["step_size"])].copy()
        reframed_attack_dataset_passivity = series_to_supervised(data_for_model, self.model_config["step_size"], 1)
        attack_values = reframed_attack_dataset_passivity.values
        n_obs = self.model_config["step_size"] * self.total_num_of_feature
        attack_x_passivity = attack_values[:,:n_obs]
        if self.model_name == 'lstm':
            attack_x_passivity = attack_x_passivity.reshape((attack_x_passivity.shape[0], self.model_config["step_size"], self.total_num_of_feature))
        y_pred_attack = model.predict(attack_x_passivity)
        mse_values = np.array([mean_squared_error(attack_y[i, :], y_pred_attack[i, :]) for i in range(attack_y.shape[0])])

        return mse_values

'''Code usage:
py_file_name model_name attack_id_to_evaluate passivity_check, feature_portion_numerator feature_portion_denominator
'''

start_time = time.time()

seed_num = 67999
tf.random.set_seed(seed_num)
tf.compat.v1.set_random_seed(seed_num)
tf.experimental.numpy.random.seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)

model_name = sys.argv[1]
attack_id = sys.argv[2]
passivity_check = sys.argv[3]
feature_portion_numerator = int(sys.argv[4])
feature_portion_denominator = int(sys.argv[5])
feature_portion = feature_portion_numerator/feature_portion_denominator

num_feature = 51
iter_per_data = 25
passivity = passivity_check == 'true'

if model_name =='lstm':
    step_size = 100
    threshold = 0.305
    model = load_model('for_ppo_LSTM_70_neuron100_batch256_step100')
    print('Model loaded!')
    feature_importance_list = [5, 35, 6, 34, 37, 25, 45, 18, 28, 7, 16, 39, 1, 46, 47, 44, 17, 38, 36, 0, 40, 41, 27, 26, 8]
else:
    step_size = 12
    if model_name == 'knn':
        threshold = 0.3155
        model = load_variable('knn_2')
        print('Model loaded!')
        feature_importance_list = [8, 26, 1, 6, 7, 16, 17, 18, 25, 28, 34, 35, 36, 37, 47, 27, 38, 39, 40, 41, 44, 45, 46, 0, 5]
    if model_name == 'rfr':
        threshold = 0.29
        model = load_variable('RFR_12_51_67999')
        print('Model loaded!')
        feature_importance_list = [47, 35, 0, 18, 34, 1, 36, 8, 17, 45, 25, 40, 39, 7, 16, 6, 41, 27, 37, 28, 38, 26, 44, 46, 5]
    if model_name == 'lr':
        threshold = 0.1
        model = load_variable('LR')
        print('Model loaded!')
        feature_importance_list = [17, 8, 16, 38, 40, 44, 46, 6, 7, 25, 34, 36, 37, 39, 45, 47, 28, 0, 18, 41, 27, 1, 35, 26, 5]

new_data_to_test, new_data_label, new_data_label_various = help_func.get_attack_data(model_name, attack_id, step_size, passivity)
number_of_data = new_data_to_test.shape[0]
data_y = new_data_to_test[step_size:].copy()
new_data_label = new_data_label[step_size:].copy()

reframed_attack_dataset = series_to_supervised(new_data_to_test, step_size, 1)
attack_values = reframed_attack_dataset.values
n_obs = step_size * num_feature
attack_x = attack_values[:,:n_obs]
attack_y = attack_values[:,n_obs:]

if model_name == 'lstm':
    attack_x = attack_x.reshape((attack_x.shape[0], step_size, num_feature))

y_pred_attack = model.predict(attack_x)
mse_values = [mean_squared_error([true], [pred]) for true, pred in zip(attack_y, y_pred_attack)]
y_pred_binary = [1 if mse > threshold else 0 for mse in mse_values]

n = 20
y_pred_binary_refined = []
for i in range(len(y_pred_binary)):
    if y_pred_binary[i] == 1:
        if i + n < len(y_pred_binary):
            subsequent_anomalies = sum(y_pred_binary[i+1:i+n+1])
            if subsequent_anomalies >= n / 8:
                y_pred_binary_refined.append(1)  # Confirm as anomaly
            else:
                y_pred_binary_refined.append(0)  # Mark as normal
        else:
            y_pred_binary_refined.append(y_pred_binary[i])
    else:
        y_pred_binary_refined.append(0)

# -----------------------------PPO evasion attack start-----------------------------------

ppo_env = tf_py_environment.TFPyEnvironment(CustomEnvironment(model_name, passivity, iter_per_data))

actor_net = actor_distribution_network.ActorDistributionNetwork(
    ppo_env.observation_spec(),
    ppo_env.action_spec(),
    fc_layer_params=(100, 50))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)

value_net = value_network.ValueNetwork(
    ppo_env.observation_spec(),
    fc_layer_params=(100, 50)
)

train_step_counter = tf.Variable(0, dtype=tf.float32)

tf_agent = ppo_clip_agent.PPOClipAgent(
    ppo_env.time_step_spec(),
    ppo_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net,
    optimizer=optimizer,
    train_step_counter=train_step_counter,
    num_epochs=10
)
tf_agent.initialize()

num_iterations = number_of_data * iter_per_data
collect_episodes_per_iteration = 1 # update ppo per each episode
batch_size = 64
replay_buffer_max_length = 10000


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size=ppo_env.batch_size,
    max_length=replay_buffer_max_length
)

collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    ppo_env,
    tf_agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=collect_episodes_per_iteration
)

model_config = {
"model": model,
"step_size": step_size,
"threshold": threshold,
"data_y": data_y}

ppo_env.pyenv.envs[0].model_config = model_config
ppo_env.pyenv.envs[0].attack_data_all = new_data_to_test.copy()
ppo_env.pyenv.envs[0].attack_data_all_original = new_data_to_test.copy()
ppo_env.pyenv.envs[0].attack_data_label = new_data_label.copy()
marked_importance_list = help_func.mask_importance_list(attack_id, feature_importance_list)
new_len = math.ceil(feature_portion*len(marked_importance_list))
ppo_env.pyenv.envs[0].feature_importance_list = marked_importance_list[0:new_len]

print('Exploring..')
for k in range(num_iterations):
    ppo_env.pyenv.envs[0].episode_end = False
    collect_driver.run()
    ppo_env.pyenv.envs[0].episode_end = True # as the agent will call reset at the end and call again at the beginning, so this can prevent it from calling twice
    experience = replay_buffer.gather_all()
    train_loss = help_func.train_agent_with_clipping(tf_agent, experience)
    replay_buffer.clear()
    if ppo_env.pyenv.envs[0].exploration_end == True:
        break

print("AE exploration for this current attack ID is finished!")
print('The average score is ', np.mean(ppo_env.pyenv.envs[0].score_change_across_episode))
print('The time spent is: {:.2f} s'.format(time.time()-start_time))

# -----------------below is the re-evaluation on the perturbed data instances-----------------

perturbed_data = series_to_supervised(ppo_env.pyenv.envs[0].attack_data_all.copy(), step_size, 1).values
n_obs = step_size * num_feature
look_beyond = 20
attack_x_all = perturbed_data[:,:n_obs]

if model_name == 'lstm':
    attack_x_all = attack_x_all.reshape((attack_x_all.shape[0], step_size, num_feature))

y_pred_binary_refined_after = help_func.re_evaluate(model, attack_x_all, threshold, new_data_label, data_y, look_beyond)

count = 0
ori_detect_attack_count = 0
for i in range(len(y_pred_binary_refined)):
    if y_pred_binary_refined[i] == 1:
        ori_detect_attack_count += 1
        if y_pred_binary_refined_after[i] != 1:
            count += 1

print('-----------------------Evasion Attack Summary-----------------------')
print('Number_of_data', number_of_data)
print('Feature_portion', feature_portion)
print('The number of original attack instances: ', ori_detect_attack_count)
print('The number of flipped attack instances to normal: ', count)
print('The percentage of changing the attack to normal is: {:.3f}%'.format(count/ori_detect_attack_count*100))
print('The number of features used for adding the perturbation is: ', len(ppo_env.pyenv.envs[0].feature_importance_list))

if math.isnan(np.sum(ppo_env.pyenv.envs[0].avg_perturbation_per_data)/len(ppo_env.pyenv.envs[0].avg_perturbation_per_data)):
    print('The average perturbation per data is: ',0)
    print('The evasion feasibility rate is: ',0)
else:
    print('The average perturbation per data is: ', np.sum(ppo_env.pyenv.envs[0].avg_perturbation_per_data)/len(ppo_env.pyenv.envs[0].avg_perturbation_per_data))
    print('The evasion feasibility rate is: ', 1-np.sum(ppo_env.pyenv.envs[0].EFR_rate_summary)/len(ppo_env.pyenv.envs[0].EFR_rate_summary))


# data_to_save = [attack_id, ppo_env.pyenv.envs[0].action_boundry, number_of_data, feature_portion, ori_detect_attack_count, count, count/ori_detect_attack_count*100, len(ppo_env.pyenv.envs[0].feature_importance_list), np.sum(ppo_env.pyenv.envs[0].avg_perturbation_per_data)/len(ppo_env.pyenv.envs[0].avg_perturbation_per_data), np.sum(ppo_env.pyenv.envs[0].EFR_rate_summary)/len(ppo_env.pyenv.envs[0].EFR_rate_summary)]

# file_name = f'ppo_{model_name}_test_result.xlsx'
# df_new = pd.DataFrame([data_to_save])

# if Path(file_name).is_file():
#     df_existing = pd.read_excel(file_name)
#     df = pd.concat([df_existing, df_new], ignore_index=True)
# else:
#     df = df_new
    
# df.to_excel(file_name, index=False)