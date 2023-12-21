from mylibrary import *
import sys
import numpy as np
from pandas import concat
import math
import time
from matplotlib import pyplot as plt
import tensorflow as tf
from help_func import *
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

    def __init__(self, action_boundry_input, new_data_label_input, passivity, sub_check_total):
        self.action_boundry = action_boundry_input
        self.total_num_of_feature = 51
        self.step_size = 256
        self.new_data_label = new_data_label_input
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self.action_boundry, maximum=self.action_boundry, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(self.step_size, self.total_num_of_feature), dtype=np.float32, name='observation')
        self._original_state = np.zeros((self.total_num_of_feature,), dtype=np.float32)
        self._current_step = 0
        self.onedcnn_config = None
        self.anomaly_score_list = []
        self.one_episode_anomaly_score_list = []
        self.feature_importance_list = [27, 38, 44, 40, 46, 41, 37, 39, 26, 35, 5, 34, 17, 8, 16, 0, 36, 18, 1, 28, 45, 6, 25, 47, 7]
        self.action_sequence = []
        self.best_action_sequence = []
        self.current_state = np.zeros((self.step_size, self.total_num_of_feature), dtype=np.float32)
        self.first_step_score = 0
        self.overall_counter = 960 # start from -1, so that after inc at the first step, it is 0
        self.attack_data_all = np.zeros((1,self.total_num_of_feature), dtype=np.float32)
        self.attack_data_all_original = np.zeros((1,self.total_num_of_feature), dtype=np.float32)
        self.data_before_augmenting_ori = np.zeros((1,self.total_num_of_feature), dtype=np.float32)
        self.attack_data_label = []
        self.score_change_across_episode = []
        self.episode_end = False
        self.sub_check_total = sub_check_total
        self.sub_check_counter = 0
        self.best_data_to_substitute = np.zeros((1,self.total_num_of_feature), dtype=np.float32) # this record the best action from the sub check loop
        self.best_score_in_sub_check_loop = np.inf
        self.avg_perturbation_per_data = []
        self.EFR_rate_summary = []
        self.sub_loop_end = True
        self.evasion_severity_factor = 0.5
        self.passivity_check = passivity
        self.exploration_end = False
        self.explored_num_data = 1

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
            self.data_before_augmenting = self.data_before_augmenting_ori.copy()
            self.current_state = self.data_before_augmenting[1: self.onedcnn_config["step_size"]+1].copy()
        self.one_episode_anomaly_score_list = []
        self.action_sequence = []

        return ts.restart(self.current_state)

    def set_initial_state(self):
        ''' Used to set the initial state value for starting each new episode'''

        while True:
            self.overall_counter += 1
            if self.overall_counter >= len(self.attack_data_all) - self.onedcnn_config["window_size"] - self.onedcnn_config["step_size"]: 
                self.overall_counter = 0
                self.exploration_end = True
                self.attack_data_all = self.attack_data_all_original.copy() # reset the modified attack dataset to its original value

            data_before_augmenting = self.attack_data_all[self.overall_counter:self.overall_counter + self.onedcnn_config["step_size"] + self.onedcnn_config["window_size"]+2].copy()
            _, zscore = single_check_z_score_result(self.onedcnn_config["model"], self.overall_counter, data_before_augmenting.copy(), self.onedcnn_config["step_size"], self.onedcnn_config["window_size"], eval_y, mu_e, sigma_e, threshold, window_num_check)

            if  zscore > self.onedcnn_config["threshold"]:
                self.data_before_augmenting = data_before_augmenting.copy()
                self.data_before_augmenting_ori = data_before_augmenting.copy()
                self.current_state = self.data_before_augmenting[1: self.onedcnn_config["step_size"]+1].copy()
                self.first_step_score = zscore # this is the initial score without perturbation added
                self.sub_loop_end = False
                break

    def reset_sub_loop(self):
        
        self.sub_check_counter = 0
        self.best_score_in_sub_check_loop = np.inf
        self.best_data_to_substitute = np.zeros((1,51), dtype=np.float32) # this record the best action from the sub check loop
        self.sub_loop_end = True
        self.best_action_sequence = []
        if self.success_label_flip_check() == True:
            self.exploration_end = True
        else:
            self.explored_num_data += 1

    def _step(self, action):

        if self._current_step >= len(self.feature_importance_list):
            return self.reset()

        action = tf.tanh(action) * self.action_boundry
        action_range = (self._action_spec.maximum - self._action_spec.minimum) / 2.0
        action_center = (self._action_spec.maximum + self._action_spec.minimum) / 2.0
        action = action_range * action + action_center

        self.action_sequence.append(np.abs(action.numpy()[0])/self.action_boundry)
        assessment_score = self.evaluate(action) # Compute the anomaly score for this data
        reward = -assessment_score

        self._current_step += 1
        if self._current_step == len(self.feature_importance_list):
            self.anomaly_score_list.append(self.one_episode_anomaly_score_list)

            if assessment_score < self.best_score_in_sub_check_loop:
                self.best_score_in_sub_check_loop = assessment_score
                self.best_data_to_substitute = self.data_before_augmenting.copy()
                self.best_action_sequence = self.action_sequence.copy()
            self.action_sequence = []
            self.sub_check_counter += 1

            if self.sub_check_counter == self.sub_check_total:
                if self.first_step_score - self.best_score_in_sub_check_loop > 0: # only we apply the perturbation if the score change > 0
                    self.attack_data_all[self.overall_counter:self.overall_counter + self.onedcnn_config["step_size"] + self.onedcnn_config["window_size"]+2] = self.best_data_to_substitute.copy() # save the changes on action back to the overall data for next episode

                if self.exploration_end == False:
                    self.avg_perturbation_per_data.append(np.sum(self.best_action_sequence)/len(self.best_action_sequence))
                    self.EFR_rate_summary.append(self.EFR_rate_calculation())

                # -------------------------Passivity calculation---------------------------
                if self.passivity_check == True:
                    score_before_add_action = self.passivty_evaluate(self.data_before_augmenting_ori.copy())
                    score_after_add_action = self.passivty_evaluate(self.best_data_to_substitute.copy())
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
                # ---------------------------The end of passivity calculation----------------

                self.score_change_across_episode.append(self.first_step_score - self.best_score_in_sub_check_loop)
                self.reset_sub_loop()

            self.episode_end = True
            return ts.termination(self.current_state, reward)
        else:
            return ts.transition(self.current_state, reward, discount=tf.constant(0.99))

    def evaluate(self, action):
        '''Main evaluate for the PPO exploration'''
        self.data_before_augmenting[self.onedcnn_config["step_size"]][self.feature_importance_list[self._current_step]] *= (1 + action)
        self.current_state[-1, self.feature_importance_list[self._current_step]] *= (1 + action)
        _, zscore = single_check_z_score_result(
        self.onedcnn_config["model"], 
        self.overall_counter, 
        self.data_before_augmenting.copy(),
        self.onedcnn_config["step_size"], 
        self.onedcnn_config["window_size"], 
        self.onedcnn_config["eval_y"], 
        self.onedcnn_config["mu_e"], 
        self.onedcnn_config["sigma_e"], 
        self.onedcnn_config["threshold"], 
        self.onedcnn_config["window_num_check"])
        self.one_episode_anomaly_score_list.append(zscore)

        return zscore
    
    def passivty_evaluate(self, data_for_model):
        '''Passivity calculation preparation'''
        true_y = eval_y[self.overall_counter:self.overall_counter+window_size]
        augmented_data = augment_data(data_for_model.copy())
        eval_generator = TimeSeriesGeneratorWithTargets(augmented_data, step_size, 256)
        pred_y = self.onedcnn_config["model"].predict(eval_generator, verbose=0)
        sum_z_scores = z_score_anomaly_detection_passivity_check(pred_y, true_y, mu_e, sigma_e, self.onedcnn_config['step_size'])

        return sum_z_scores

    def success_label_flip_check(self):
        '''Act as the continuous check for the 1dcnn model'''
        augmented_data_attack_single_check = augment_data(self.attack_data_all.copy())
        eval_generator = TimeSeriesGeneratorWithTargets(augmented_data_attack_single_check, self.onedcnn_config["step_size"], 256)
        y_pred_eval = self.onedcnn_config["model"].predict(eval_generator)
        eval_y = []
        for batch_num in range(len(eval_generator)):
            _, y_batch = eval_generator[batch_num]
            eval_y.append(y_batch)
        eval_y = np.concatenate(eval_y, axis=0)[:len(y_pred_eval)]
        _, detected_result_all = z_score_anomaly_detection(y_pred_eval, eval_y, self.onedcnn_config["mu_e"], self.onedcnn_config["sigma_e"], self.onedcnn_config["threshold"], self.onedcnn_config["window_size"], onedcnn_config["window_num_check"])
        detected_result_all = detected_result_all[:len(self.new_data_label)]
        count = 0
        ori_detect_attack_count = 0
        for i in range(len(before_perturbation_predicted_list)):
            if before_perturbation_predicted_list[i] == 1:
                ori_detect_attack_count += 1
                if detected_result_all[i] != 1:
                    count += 1
        # print('Current flipped percent is: {:.3f}%'.format(count/ori_detect_attack_count*100))
        # print('The time spent is: {:.2f} s'.format(time.time()-start_time))
        if count == ori_detect_attack_count:
            return True
        else:
            return False
            

def single_check_z_score_result(target_model, i, step_data_to_test, step_size, window_size, eval_y, mu_e, sigma_e, threshold, window_num_check):
    '''PPO exclusive function. 
    Check the detection result of the data individually'''
    true_y = eval_y[i:i+window_size]
    augmented_data = augment_data(step_data_to_test)
    eval_generator = TimeSeriesGeneratorWithTargets(augmented_data, step_size, 256)
    pred_y = target_model.predict(eval_generator, verbose=0)

    min_length = min(len(pred_y), len(true_y))
    pred_y = pred_y[:min_length].copy()
    true_y = true_y[:min_length].copy()
    errors = np.abs(pred_y - true_y)
    epsilon = 1e-10
    z_scores = np.abs(errors - mu_e) / (sigma_e + epsilon) # (2000, 51)
    anomalies = []
    if np.sum(z_scores[0]) > threshold:
        window_anomalies = np.where(np.sum(z_scores[0:window_size], axis=1) > threshold, 1, 0)
        if np.sum(window_anomalies) >= window_size / window_num_check:
            anomalies.append(1)
        else:
            anomalies.append(0)
    else:
        anomalies.append(0)
                
    return anomalies[0], np.sum(z_scores[0])

def z_score_anomaly_detection_passivity_check(realtime_predictions, realtime_actuals, mu_e, sigma_e, step_size):
    """
    Detect anomalies in real-time data using z-scores and apply a time window refinement
    This is used for passivity check, so record 1 + step size anomaly score.
    """
    min_length = min(len(realtime_predictions), len(realtime_actuals))
    realtime_predictions = realtime_predictions[:min_length].copy()
    realtime_actuals = realtime_actuals[:min_length].copy()
    errors = np.abs(realtime_predictions - realtime_actuals)
    epsilon = 1e-10
    z_scores = np.abs(errors - mu_e) / (sigma_e + epsilon) # (2000, 51)
    plot_len = round(step_size * 1.2)
    sum_z_scores = np.sum(z_scores[:plot_len, :], axis=1)

    return sum_z_scores


# ---------------------------------start------------------------------------

start_time = time.time()


seed_num = 67999
tf.random.set_seed(seed_num)
tf.compat.v1.set_random_seed(seed_num)
tf.experimental.numpy.random.seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)

action_boundry_input = 1.0

model_name = sys.argv[1]
attack_id = sys.argv[2]
passivity_check = sys.argv[3] # default: false
feature_portion_numerator = int(sys.argv[4])
feature_portion_denominator = int(sys.argv[5])

feature_portion = feature_portion_numerator/feature_portion_denominator
num_feature = 51
iter_per_data = 25
passivity = passivity_check == 'true'


step_size = 256
batch_size = 256
window_size = 2000
window_num_check = 2
num_feature = 51
augmentation_lag = 1

mu_e = load_variable('1dcnn_mu')
sigma_e = load_variable('1dcnn_sigma')
threshold = load_variable('1dcnn_threshold')

new_data_to_test, new_data_label, new_data_label_various = get_attack_data(model_name, attack_id, step_size, passivity)
number_of_data = new_data_to_test.shape[0]
new_data_label = new_data_label[step_size+1:-window_size+1]
new_data_label_various = new_data_label_various[step_size+1:-window_size+1]

model = load_model('1dcnn_zscore_epo_100_batch256_step256_patience15_fullepoch')
augmented_data_attack = augment_data(new_data_to_test.copy())
eval_generator = TimeSeriesGeneratorWithTargets(augmented_data_attack, step_size, 256)
y_pred_eval = model.predict(eval_generator)

eval_y = []
for batch_num in range(len(eval_generator)):
    x_batch, y_batch = eval_generator[batch_num]
    eval_y.append(y_batch)
eval_y = np.concatenate(eval_y, axis=0)[:len(y_pred_eval)]

detected_zscores, detected_result_all = z_score_anomaly_detection(y_pred_eval, eval_y, mu_e, sigma_e, threshold, window_size, window_num_check)
detected_result_all = detected_result_all[:len(new_data_label)]
before_perturbation_predicted_list = detected_result_all.copy()

# -----------------------------PPO evasion attack start-----------------------------------

ppo_env = tf_py_environment.TFPyEnvironment(CustomEnvironment(action_boundry_input, new_data_label, passivity, iter_per_data))
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
collect_episodes_per_iteration = 1
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

onedcnn_config = {
"model": model,
"step_size": step_size, 
"window_size": window_size,  
"eval_y": eval_y,  
"mu_e": mu_e, 
"sigma_e": sigma_e, 
"threshold": threshold, 
"window_num_check": window_num_check}

ppo_env.pyenv.envs[0].onedcnn_config = onedcnn_config
ppo_env.pyenv.envs[0].attack_data_all = new_data_to_test.copy()
ppo_env.pyenv.envs[0].attack_data_all_original = new_data_to_test.copy()
ppo_env.pyenv.envs[0].attack_data_label = new_data_label.copy()

marked_importance_list = mask_importance_list(attack_id, ppo_env.pyenv.envs[0].feature_importance_list.copy())
new_len = math.ceil(feature_portion*len(marked_importance_list))
ppo_env.pyenv.envs[0].feature_importance_list = marked_importance_list[0:new_len]

print('PPO starts exploring..')
for k in range(num_iterations):

    ppo_env.pyenv.envs[0].episode_end = False
    collect_driver.run()
    ppo_env.pyenv.envs[0].episode_end = True
    if ppo_env.pyenv.envs[0].exploration_end == True:
        print('Exploration finished, or not found appropriate data to perturb after one whole iteration.')
        print('The number of instances explored so far is: ', ppo_env.pyenv.envs[0].explored_num_data)
        break
    experience = replay_buffer.gather_all()
    train_loss = train_agent_with_clipping(tf_agent, experience)
    replay_buffer.clear()

print("AE exploration for this current attack ID is finished!")
print('The average score is ', np.mean(ppo_env.pyenv.envs[0].score_change_across_episode))
print('The time spent is: {:.2f} s'.format(time.time()-start_time))

if math.isnan(np.sum(ppo_env.pyenv.envs[0].avg_perturbation_per_data)/len(ppo_env.pyenv.envs[0].avg_perturbation_per_data)):
    print('The average perturbation per data is: ', 0)
    print('The evasion feasibility rate is: ', 0)
else:
    print('The average perturbation per data is: ', np.sum(ppo_env.pyenv.envs[0].avg_perturbation_per_data)/len(ppo_env.pyenv.envs[0].avg_perturbation_per_data))
    print('The evasion feasibility rate is: ', 1-np.sum(ppo_env.pyenv.envs[0].EFR_rate_summary)/len(ppo_env.pyenv.envs[0].EFR_rate_summary))

# data_to_save = [action_boundry_input, number_of_data, feature_portion, len(ppo_env.pyenv.envs[0].feature_importance_list), np.sum(ppo_env.pyenv.envs[0].avg_perturbation_per_data)/len(ppo_env.pyenv.envs[0].avg_perturbation_per_data), np.sum(ppo_env.pyenv.envs[0].EFR_rate_summary)/len(ppo_env.pyenv.envs[0].EFR_rate_summary)]


# file_name = 'ppo_1dcnn_test_result.xlsx'
# df_new = pd.DataFrame([data_to_save])

# if Path(file_name).is_file():
#     # Read the existing file
#     df_existing = pd.read_excel(file_name)
#     # Append new data
#     df = pd.concat([df_existing, df_new], ignore_index=True)
# else:
#     # Use the new DataFrame as the starting point
#     df = df_new
    
# df.to_excel(file_name, index=False)