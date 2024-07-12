if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=Fredformer

root_path_name=./dataset/
data_path_name=Solar.csv
model_id_name=Solar
data_name=Solar

random_seed=2021
for pred_len in 96 192 336 720
do
    case $pred_len in
        96)  cf_dim=164 cf_depth=3 cf_heads=8 cf_mlp=256 cf_head_dim=32 d_model=64;;
        192) cf_dim=164 cf_depth=3 cf_heads=8 cf_mlp=256 cf_head_dim=32 d_model=64;;
        336) cf_dim=256 cf_depth=3 cf_heads=8 cf_mlp=256 cf_head_dim=32 d_model=64;;
        720) cf_dim=256 cf_depth=3 cf_heads=8 cf_mlp=256 cf_head_dim=32 d_model=64;;
    esac
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --d_model $d_model \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 16\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log \
      --gpu 0 \
      --cf_dim $cf_dim \
      --cf_depth $cf_depth \
      --cf_heads $cf_heads \
      --cf_mlp $cf_mlp \
      --cf_head_dim $cf_head_dim \
      --use_nys 0 \
      --individual 0
done