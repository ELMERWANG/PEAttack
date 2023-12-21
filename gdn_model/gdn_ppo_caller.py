import torch
from util.data import *
import numpy as np
import helper_func
import os
import argparse
import random
import mylibrary
from stable_baselines3 import PPO
from gdn_model_main import *
import math
from gdn_ppo import PerturbPPO
import time

'''The caller for the GDN's PPO evaluation'''


step_size = 5
model_name = 'GDN'
num_feature = 51

# =========================To be adjusted==============================
attack_id = 'A39'
feature_portion = 3/3 # max % of sensor features to be perturbed
passivity_check = False
# ======================================================================

new_data_to_test, new_data_label, new_data_label_various = helper_func.get_attack_data(model_name, attack_id, step_size, passivity_check)
number_of_data = new_data_to_test.shape[0]

start_time = time.time()

parser = argparse.ArgumentParser()

parser.add_argument('-batch', help='batch size', type = int, default=128)
parser.add_argument('-epoch', help='train epoch', type = int, default=100)
parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
parser.add_argument('-dim', help='dimension', type = int, default=64)
parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
parser.add_argument('-random_seed', help='random seed', type = int, default=0)
parser.add_argument('-comment', help='experiment comment', type = str, default='')
parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
parser.add_argument('-decay', help='decay', type = float, default=0)
parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
parser.add_argument('-topk', help='topk num', type = int, default=20)
parser.add_argument('-report', help='best / val', type = str, default='best')
parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

args = parser.parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.random_seed)


train_config = {
    'batch': args.batch,
    'epoch': args.epoch,
    'slide_win': args.slide_win,
    'dim': args.dim,
    'slide_stride': args.slide_stride,
    'comment': args.comment,
    'seed': args.random_seed,
    'out_layer_num': args.out_layer_num,
    'out_layer_inter_dim': args.out_layer_inter_dim,
    'decay': args.decay,
    'val_ratio': args.val_ratio,
    'topk': args.topk,
}

env_config={
    'save_path': args.save_path_pattern,
    'dataset': args.dataset,
    'report': args.report,
    'device': args.device,
    'load_model_path': args.load_model_path
}

initialize_evasion_instance = GDNMain(train_config, env_config, debug=False)
target_model = initialize_evasion_instance.initialize_GDN()
target_model.load_state_dict(torch.load(env_config['load_model_path']))
target_model.eval()

test_predicted_list = []
test_ground_list = []
test_labels_list = []
t_test_predicted_list = []
t_test_ground_list = []
t_test_labels_list = []

single_check = True
marked_importance_list = helper_func.mask_importance_list(attack_id)
ppo_n_steps = len(marked_importance_list)
new_len = math.ceil(feature_portion*len(marked_importance_list))
timesteps_base = 25 * number_of_data * ppo_n_steps
edge_index = mylibrary.load_variable('edge_index')
env = PerturbPPO(target_model, edge_index, new_data_to_test.copy(), initialize_evasion_instance, new_data_label, marked_importance_list[0:new_len], env_config['device'], passivity_check)
ppo_model = PPO('MlpPolicy', env, verbose=0, n_steps=ppo_n_steps)
print('PPO start exploring..')
ppo_model.learn(total_timesteps=timesteps_base)
before_loader = initialize_evasion_instance.get_dataloader(new_data_to_test.copy())
after_loader = initialize_evasion_instance.get_dataloader(env.attack_data_all)


original_predict_list = helper_func.do_prediction(target_model, before_loader, env_config)
predict_list_alter = helper_func.do_prediction(target_model, after_loader, env_config)

counter = 0
counter_same = 0
for i in range(len(original_predict_list)):
    if original_predict_list[i] != predict_list_alter[i]:
        counter += 1
    else:
        counter_same += 1

print('----------------------FINAL RESULT------------------------')
print('The percentage of changing the attack to normal is: {:.3f}%'.format(counter/(counter+counter_same)*100))
print('The number of features used for adding the perturbation is: ', len(env.feature_importance_list))
print('The average perturbation per data is: ', np.sum(env.avg_perturbation_per_data)/len(env.avg_perturbation_per_data))
print('The evasion feasibility rate is: ', 1-np.sum(env.EFR_rate_summary)/len(env.EFR_rate_summary))

print('The time spent is: {:.2f} s'.format(time.time()-start_time))
