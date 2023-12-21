import torch
from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import argparse
import random
from gdn_model_main import *
import time

def altered_result_check_permutation(test_result):

    feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)
    test_labels = np_test_result[2, :, 0].tolist()
    all_scores =  None
    feature_num = np_test_result.shape[-1]
    
    para_list_all_test = [0.0340476930141449, 0.030281655490398407, 0.017242267727851868, 0.03566721826791763, 0.008933395147323608, 0.019947320222854614, 0.01802733540534973, 0.004210375249385834, 0.0018786638975143433, 0.0009308904409408569, 2.305459499359131, 2.696294814348221, 0.30325765907764435, 0.3488807901740074, 0.037966009229421616, 0.06902330042794347, 0.0349506139755249, 0.03651968389749527, 0.019668519496917725, 0.01806042343378067, 0.016895443201065063, 0.013939663767814636, 0.009604275226593018, 0.00045733898878097534, 0.016164630651474, 0.025830477476119995, 0.014166727662086487, 0.0045279860496521, 0.01459529995918274, 0.016994863748550415, 0.005035564303398132, 0.0025365129113197327, 0.011150360107421875, 0.01804201304912567, 0.006132751703262329, 0.03951841592788696, 0.029582351446151733, 0.054810985922813416, 0.0049295127391815186, 0.01884138584136963, 0.010411947965621948, 0.010396033525466919, 0.0063134729862213135, 0.0013577714562416077, 0.010552048683166504, 0.00142601877450943, 0.006651431322097778, 0.0014042407274246216, 0.011830419301986694, 0.010372772812843323, 0.021661758422851562, 0.024722829461097717, 0.15903190337121487, 0.10623387157102115, 0.01512196660041809, 0.00535312294960022, 0.0380932092666626, 0.03692331910133362, 0.005101799964904785, 0.0005407184362411499, 0.012979954481124878, 0.0010611414909362793, 0.009819209575653076, 0.0007431879639625549, 0.011462032794952393, 0.0003506690263748169, 0.010029464960098267, 0.00783132016658783, 0.0338405966758728, 0.046367138624191284, 0.10952365421690047, 0.07713164482265711, 0.06852343678474426, 0.15918677300214767, 0.004874696023762226, 0.004463938530534506, 0.01105421781539917, 0.007791385054588318, 0.01532965898513794, 0.012232720851898193, 0.014491647481918335, 0.01453825831413269, 0.046957194805145264, 0.014056220650672913, 0.014759957790374756, 0.013457223773002625, 0.013816595077514648, 0.0011324584484100342, 0.00956752896308899, 0.009589701890945435, 0.043571799993515015, 0.052660245448350906, 0.012064218521118164, 0.0054275840520858765, 0.011920935685338918, 0.0021867416508030146, 0.006061851978302002, 0.00042732805013656616, 0.010233476758003235, 0.0046311840415000916, 0.009731411933898926, 8.412450551986694e-05]

    for i in range(feature_num):
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
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1
    test_labels = np.array(test_labels)

    pre = precision_score(test_labels, pred_labels)
    rec = recall_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels)

    return f1, pre, rec


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
input_samples = initialize_evasion_instance.get_test_data() # shape(44991, 52)
sensor_feature_index = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
per_result = []
print('Start shuffling and testing, please wait..')
for i in sensor_feature_index:
    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []
    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    target_model.eval()
    X_test_permuted = input_samples.copy()
    
    col = X_test_permuted.columns[i]  # Get the column name at index i
    shuffled_values = X_test_permuted[col].values  # Extract the column values as an array
    np.random.shuffle(shuffled_values)  # Shuffle the array
    X_test_permuted[col] = shuffled_values
    input_loader = initialize_evasion_instance.get_dataloader(X_test_permuted)

    for x, y, labels, edge_index in input_loader:
        x, y, labels, edge_index = [item.to(env_config['device']).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad(): # turn off gradient computation during evaluation
            predicted = target_model(x, edge_index).float().to(env_config['device'])
            labels_single = labels.clone()
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()    
    test_labels_list = t_test_labels_list.tolist()  
    f1, pre, rec = altered_result_check_permutation([test_predicted_list, test_ground_list, test_labels_list])
    per_result.append([i, f1, pre, rec])

df = pd.DataFrame(per_result, columns=["feature ID", "f1", "pre", "rec"])
df.to_excel("output_permutation.xlsx", index=False, engine='openpyxl') # Write the DataFrame to an Excel file
print('Permutation result is saved, exiting..')
print('The time spent is: {:.2f} s'.format(time.time()-start_time))
