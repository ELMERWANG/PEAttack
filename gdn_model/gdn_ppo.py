import gym
import numpy as np
import torch
from helper_func import *
import math
import sys
import matplotlib.pyplot as plt


class PerturbPPO(gym.Env):
    def __init__(self, target_model, edge_index, sepcified_attack_data_only, initialize_evasion_instance, attack_data_label, marked_importance_list, device_type, passivity_check):
        super(PerturbPPO, self).__init__()
        self.action_boundry = 1.0
        self.total_num_of_feature = 51
        self.gdn_instance = initialize_evasion_instance
        self.attack_data_all = sepcified_attack_data_only.copy()
        self.attack_data_all_true_ori = sepcified_attack_data_only.copy()
        self.attack_data_label = attack_data_label
        self.device = device_type
        self.action_space = gym.spaces.Box(low=-self.action_boundry, high=self.action_boundry, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, 51), dtype=np.float32) # (51,5,)
        self._current_step = 0 # indicate the index in the current episode
        self.anomaly_score_list = []
        self.one_episode_anomaly_score_list = []
        self.feature_importance_list = marked_importance_list
        self.threshold = 35.458921639159946
        self.action_sequence = []
        self.best_action_sequence = []
        self.current_state = np.zeros((5, 51), dtype=np.float32)
        self.first_step_score = 0
        self.overall_counter = -1 # start from -1, so that after inc at the first step, it is 0
        self.score_change_across_episode = []
        self.sub_check_total = 25
        self.sub_check_counter = 0
        self.timestep = 5
        self.best_score_in_sub_check_loop = np.inf
        self.avg_perturbation_per_data = []
        self.EFR_rate_summary = []
        self.sub_loop_end = True
        self.evasion_severity_factor = 0.5
        self.passivity_check = passivity_check
        self.target_model = target_model
        self.edge_index = edge_index
        self.last_anomaly_score = 0
        self.t_test_predicted_list = []
        self.t_test_ground_list = []
        self.t_test_labels_list = []
        self.best_t_test_predicted_last = 0
        self.ori_t_test_predicted_last = 0
        self.end_exploration = False


    def step(self, action):

        if self._current_step >= len(self.feature_importance_list):
            return self.reset()
        action = math.tanh(action)
        action_range = self.action_boundry
        action_center = 0
        action = action_range * action + action_center
        self.action_sequence.append(action)
        assessment_score = self.evaluate()
        reward = -assessment_score
        self._current_step += 1

        if self._current_step == len(self.feature_importance_list):
            self.anomaly_score_list.append(self.one_episode_anomaly_score_list)
            if assessment_score < self.best_score_in_sub_check_loop:
                self.best_score_in_sub_check_loop = assessment_score
                self.best_action_sequence = self.action_sequence.copy()
                self.best_t_test_predicted_last = self.t_test_predicted_list[-1].clone()

            self.sub_check_counter += 1

            if self.sub_check_counter == self.sub_check_total:
                if self.end_exploration == False:
                    if self.first_step_score - self.best_score_in_sub_check_loop > 0: # only we apply the perturbation if the score change > 0
                        self.t_test_predicted_list[-1] = self.best_t_test_predicted_last.clone()
                        for i in range(len(self.best_action_sequence)):
                            self.attack_data_all.iloc[self.overall_counter+4, self.feature_importance_list[i]] *= (1 + self.best_action_sequence[i])
                            self.avg_perturbation_per_data.append(np.sum(np.abs(self.best_action_sequence)) / len(self.best_action_sequence)) # only used for calculate the avg perturbation, so used abs
                            self.EFR_rate_summary.append(self.EFR_rate_calculation())

                    # ==========================================start passivity calculation=============================================
                    if self.passivity_check == True:
                        print('Passivity check started..')
                        before_anomaly_score_list = self.passivity_evaluate(self.attack_data_all_true_ori)
                        after_anomaly_score_list = self.passivity_evaluate(self.attack_data_all)
                        array1 = np.array(before_anomaly_score_list)
                        array2 = np.array(after_anomaly_score_list)
                        diff = array1 - array2

                        # mylibrary.save_variable(diff, 'gdn_passivity', True)
                        plt.figure(figsize=(10, 6))
                        plt.plot(diff)
                        plt.title("Score diff before and after adding perturbation at 1th timestamp")
                        plt.xlabel("Anomaly score")
                        plt.ylabel("Anomaly score diff")
                        plt.grid(True)
                        plt.savefig('gdn_passivity')

                        print('The passivity rate is calculated, exiting program..')
                        sys.exit()
                    # ==========================================end passivity calculation================================================

                self.score_change_across_episode.append(self.first_step_score - self.best_score_in_sub_check_loop)
                self.reset_sub_loop()
            done = True
        else:
            done = False

        return self.current_state, reward, done, {}
    
    
    def reset(self):

        self._current_step = 0
        if self.sub_loop_end == True:
            self.set_initial_state()
        else:
            self.current_state = self.attack_data_all.copy().iloc[self.overall_counter:self.overall_counter+5, 0:51]

        self.one_episode_anomaly_score_list = []
        self.action_sequence = []

        return self.current_state
    
    def set_initial_state(self):
        ''' Used to set the initial state value for starting each new episode'''

        while True:
            self.overall_counter += 1
            input_loader = self.gdn_instance.get_dataloader(self.attack_data_all.copy())

            # reach to the max of the given attack dataset, reset counter to 0 and cont training
            if self.overall_counter >= len(input_loader.dataset):
                self.overall_counter = 0
                self.attack_data_all = self.attack_data_all_true_ori.copy() # reset the modified attack dataset to its original value
                self.end_exploration = True

            x_data = input_loader.dataset[self.overall_counter][0].to(self.device).float() # size([1,51,5])
            x_data = x_data.reshape(1, x_data.shape[0], x_data.shape[1])
            y_data = input_loader.dataset[self.overall_counter][1].to(self.device).float()
            y_data = y_data.reshape(1, y_data.shape[0])
            labels = input_loader.dataset[self.overall_counter][2].to(self.device).float()
            labels = labels.view(torch.Size([1]))
            edge_index = input_loader.dataset[self.overall_counter][3].to(self.device).float()
            edge_index = edge_index.reshape(1, edge_index.shape[0], edge_index.shape[1])

            self.target_model.eval()
            with torch.no_grad(): # turn off gradient computation during evaluation

                predicted = self.target_model(x_data, edge_index).float().to(self.device)
                labels_single = labels.clone()
                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

                if len(self.t_test_predicted_list) <= 0:
                    self.t_test_predicted_list = predicted.clone()
                    self.t_test_ground_list = y_data
                    self.t_test_labels_list = labels
                else:
                    self.t_test_predicted_list = torch.cat((self.t_test_predicted_list, predicted.clone()), dim=0)[-4:]
                    self.t_test_ground_list = torch.cat((self.t_test_ground_list, y_data), dim=0)[-4:]
                    self.t_test_labels_list = torch.cat((self.t_test_labels_list, labels), dim=0)[-4:]

            if self.overall_counter >= 3 and labels_single.item() == 1:
                _, anomaly_score_value = altered_result_check_ppo([self.t_test_predicted_list.tolist(), self.t_test_ground_list.tolist(), self.t_test_labels_list.tolist()], True)

                if anomaly_score_value >= self.threshold:
                    self.current_state = self.attack_data_all.copy().iloc[self.overall_counter:self.overall_counter+5, 0:51] # target timestamp is actually overall counter + 4
                    self.ori_t_test_predicted_last = self.t_test_predicted_list[-1].clone()
                    self.first_step_score = anomaly_score_value.copy() # this is the initial score without perturbation added
                    self.sub_loop_end = False
                    break

    def evaluate(self):
        
        temp_attack_data = self.attack_data_all.copy()
        to_predict = self.gdn_instance.get_dataloader(temp_attack_data)

        x_data_eval = to_predict.dataset[self.overall_counter][0].to(self.device).float()# size([1,51,5])
        x_data_eval = x_data_eval.reshape(1, x_data_eval.shape[0], x_data_eval.shape[1])

        for i in range(len(self.action_sequence)):
            x_data_eval[0, self.feature_importance_list[i], -1] *= (1 + self.action_sequence[i])

        y_data = to_predict.dataset[self.overall_counter][1].to(self.device).float()
        y_data = y_data.reshape(1, y_data.shape[0])
        labels = to_predict.dataset[self.overall_counter][2].to(self.device).float()
        labels = labels.view(torch.Size([1]))
        edge_index = to_predict.dataset[self.overall_counter][3].to(self.device).float()
        edge_index = edge_index.reshape(1, edge_index.shape[0], edge_index.shape[1])
        squeezed_array = x_data_eval.squeeze(0)
        self.current_state = np.transpose(squeezed_array.numpy()).copy()
        self.target_model.eval()

        with torch.no_grad():

            predicted_y = self.target_model(x_data_eval, edge_index).float().to(self.device)
            self.t_test_predicted_list[-1] = predicted_y.clone()
            _, tested_anomaly_score = altered_result_check_ppo([self.t_test_predicted_list.tolist(), self.t_test_ground_list.tolist(), self.t_test_labels_list.tolist()], True) # single_check enabled

        self.one_episode_anomaly_score_list.append(tested_anomaly_score)
    
        return tested_anomaly_score

    def reset_sub_loop(self):

        self.sub_check_counter = 0
        self.best_score_in_sub_check_loop = np.inf
        self.sub_loop_end = True
        self.best_action_sequence = []

    
    def EFR_rate_calculation(self):

        a = self.evasion_severity_factor
        ncf = len(self.feature_importance_list)
        ntf = self.total_num_of_feature
        pertub_sum_per_feature = sum(element / self.action_boundry for element in self.best_action_sequence)
        result = ((a * ncf) / ntf) + ((1 - a) * pertub_sum_per_feature / ncf)

        return result
    

    def passivity_evaluate(self, target_data):
        
        temp_attack_data = target_data.copy()
        to_predict = self.gdn_instance.get_dataloader(temp_attack_data)
        anomaly_score_list = []
        loop_counter = self.overall_counter
        for i in range(0, 8):
            loop_counter += i
            x_data_eval = to_predict.dataset[loop_counter][0].to(self.device).float()# size([1,51,5])
            x_data_eval = x_data_eval.reshape(1, x_data_eval.shape[0], x_data_eval.shape[1])
            y_data = to_predict.dataset[loop_counter][1].to(self.device).float()
            y_data = y_data.reshape(1, y_data.shape[0])
            labels = to_predict.dataset[loop_counter][2].to(self.device).float()
            labels = labels.view(torch.Size([1]))
            edge_index = to_predict.dataset[loop_counter][3].to(self.device).float()
            edge_index = edge_index.reshape(1, edge_index.shape[0], edge_index.shape[1])
            
            squeezed_array = x_data_eval.squeeze(0)
            self.current_state = np.transpose(squeezed_array.numpy()).copy()
            self.target_model.eval()

            with torch.no_grad():

                predicted_y = self.target_model(x_data_eval, edge_index).float().to(self.device)
                self.t_test_predicted_list[-1] = predicted_y.clone()
                _, tested_anomaly_score = altered_result_check_ppo([self.t_test_predicted_list.tolist(), self.t_test_ground_list.tolist(), self.t_test_labels_list.tolist()], True) # single_check enabled

            anomaly_score_list.append(tested_anomaly_score)
    
        return anomaly_score_list