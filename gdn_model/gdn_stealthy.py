import torch.nn.functional as F
import torch
from util.data import *
import numpy as np
import os
import argparse
import random
import time
from gdn_model_main import *
import helper_func

'''GDN stealthy attack for delayed and expedited attack'''


start_time = time.time()


# ==============================Adjustable parameters===============================
target_feature_index = 18
attack_type = 'delayed'
# ==================================================================================


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
input_samples = initialize_evasion_instance.get_test_data()
input_loader = initialize_evasion_instance.get_dataloader(input_samples)
original_predict_list = helper_func.do_prediction(target_model, input_loader, env_config)

# =================================Stealthy attack start=====================================
step_size = 5
chunk_length = 104
start_index = helper_func.find_consecutive_zeros(original_predict_list, chunk_length)
input_samples_alter = initialize_evasion_instance.get_test_data()
chunk_data_to_process = input_samples_alter.iloc[step_size+start_index:step_size+start_index+chunk_length, target_feature_index].copy()

if attack_type == 'delay':
    input_samples_alter.iloc[step_size+start_index:step_size+start_index+chunk_length, target_feature_index] = helper_func.delay_attack_payload_generator(chunk_data_to_process.to_numpy(), 2, True)
else:
    converted_data = helper_func.expedited_attack_payload_generator(chunk_data_to_process.to_numpy(), 1, 1, 0.005)
    input_samples_alter.iloc[step_size+start_index:step_size+start_index+len(converted_data), target_feature_index] = converted_data

input_loader_alter = initialize_evasion_instance.get_dataloader(input_samples_alter)
predict_list_alter = helper_func.do_prediction(target_model, input_loader_alter, env_config)

counter = 0
counter_normal = 0
for k in range(len(original_predict_list)):
    if original_predict_list[k] != predict_list_alter[k]:
        counter += 1
    else:
        counter_normal += 1

print(f'The stealthy attack success rate is:{counter/chunk_length*100}%')
