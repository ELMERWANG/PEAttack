import torch.nn.functional as F
import torch
from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import argparse
import random
import time
from gdn_model_main import *


start_time = time.time()

def altered_result_check(test_result, single_check):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)
    test_labels = np_test_result[2, :, 0].tolist()
    all_scores =  None
    feature_num = np_test_result.shape[-1]
            
    for i in range(feature_num):
        if single_check:
            test_re_list = np_test_result[:2,-4:,i]
            test_labels = np_test_result[2,-4:, i].tolist()
        else:
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
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1
    test_labels = np.array(test_labels)

    pre = precision_score(test_labels, pred_labels)
    rec = recall_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels)

    print('F1 score: ',f1)
    print('Precision: ',pre)
    print('Recall: ',rec)


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
counter = 0

while counter < len(input_loader.dataset):

    x_data = input_loader.dataset[counter][0].to(env_config['device']).float()
    x_data = x_data.reshape(1, x_data.shape[0], x_data.shape[1])
    y_data = input_loader.dataset[counter][1].to(env_config['device']).float()
    y_data = y_data.reshape(1, y_data.shape[0])
    labels = input_loader.dataset[counter][2].to(env_config['device']).float()
    labels = labels.view(torch.Size([1]))
    edge_index = input_loader.dataset[counter][3].to(env_config['device']).float()
    edge_index = edge_index.reshape(1, edge_index.shape[0], edge_index.shape[1])

    with torch.no_grad(): # turn off gradient computation during evaluation

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
    
print('----------------------FINAL RESULT------------------------')
test_predicted_list = t_test_predicted_list.tolist()
test_ground_list = t_test_ground_list.tolist()    
test_labels_list = t_test_labels_list.tolist()  
single_check = False
altered_result_check([test_predicted_list, test_ground_list, test_labels_list], single_check)
print('The time spent is: {:.2f} s'.format(time.time()-start_time))

