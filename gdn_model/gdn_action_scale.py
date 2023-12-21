import torch.nn.functional as F
import torch
from util.data import *
import numpy as np
import sys
import os
import argparse
import random
import time
from gdn_model_main import *
import helper_func


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
input_samples = initialize_evasion_instance.get_test_data()
input_loader = initialize_evasion_instance.get_dataloader(input_samples)

test_predicted_list = []
test_ground_list = []
test_labels_list = []
t_test_predicted_list = []
t_test_ground_list = []
t_test_labels_list = []
target_model.eval()

threshold = 35.458921639159946
counter = 0
action_scale = 1
feature_list = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]

while counter < len(input_loader.dataset):

    x_data = input_loader.dataset[counter][0].to(env_config['device']).float()
    x_data = x_data.reshape(1, x_data.shape[0], x_data.shape[1])
    y_data = input_loader.dataset[counter][1].to(env_config['device']).float()
    y_data = y_data.reshape(1, y_data.shape[0])
    labels = input_loader.dataset[counter][2].to(env_config['device']).float()
    labels = labels.view(torch.Size([1]))
    edge_index = input_loader.dataset[counter][3].to(env_config['device']).float()
    edge_index = edge_index.reshape(1, edge_index.shape[0], edge_index.shape[1])

    x_data_max = x_data.clone()
    x_data_min = x_data.clone()

    for k in feature_list:
        
        x_data_max[0,k,-1] *= (1+action_scale)
        x_data_min[0,k,-1] *= (1-action_scale)

    with torch.no_grad():

        predicted = target_model(x_data, edge_index).float().to(env_config['device'])
        if counter == 3:
            predicted_max = target_model(x_data_max, edge_index).float().to(env_config['device'])
            predicted_min = target_model(x_data_min, edge_index).float().to(env_config['device'])

        labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])
        if len(t_test_predicted_list) <= 0:
            t_test_predicted_list = predicted
            t_test_ground_list = y_data
            t_test_labels_list = labels
        else:
            t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)[-4:]
            t_test_ground_list = torch.cat((t_test_ground_list, y_data), dim=0)[-4:]
            t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)[-4:]

        if counter >= 3:
            predict_list_max = t_test_predicted_list.clone()
            predict_list_min = t_test_predicted_list.clone()
            predict_list_max[-1] = predicted_max
            predict_list_min[-1] = predicted_min

            anomaly_score = helper_func.altered_result_check_action_scale([t_test_predicted_list.tolist(), t_test_ground_list.tolist(), t_test_labels_list.tolist()], True)
            anomaly_score_max = helper_func.altered_result_check_action_scale([predict_list_max.tolist(), t_test_ground_list.tolist(), t_test_labels_list.tolist()], True)
            anomaly_score_min = helper_func.altered_result_check_action_scale([predict_list_min.tolist(), t_test_ground_list.tolist(), t_test_labels_list.tolist()], True)

            print(f'The anomaly score diff for max is {anomaly_score-anomaly_score_max}')
            print(f'The anomaly score diff for min is {anomaly_score-anomaly_score_min}')
            print('The max-min scale test is finished, exiting program..')
            sys.exit()

    counter += 1
