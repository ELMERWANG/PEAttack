gpu_n=$1
DATASET=$2

seed=6799
BATCH_SIZE=1
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=15
out_layer_inter_dim=128
val_ratio=0.1
decay=0

path_pattern="${DATASET}"
COMMENT="${DATASET}"

EPOCH=50
report='val'

if [[ "$gpu_n" == "cpu" ]]; then
    python main_program_to_run.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu' \
        -load_model_path 'path_to_your_folder/pretrained/swat/replace_with_your_pretrained_model.pt'

else
    CUDA_VISIBLE_DEVICES=$gpu_n  python none.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -load_model_path 'path_to_your_folder/pretrained/swat/replace_with_your_pretrained_model.pt'
fi
