export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
data=custom
root_path=E:/Pdata/iTransformer_datasets/ETT-small/
seq_len=96
#pred_len=96
order=0.5

##for pred_len in 96 192 336 720
for order in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#for diff in  0.9 1.0
do

python -u run.py \
  --is_training 1 \
  --diff 0.4 \
  --order $order \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.0 \
  --order $order \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.2 \
  --order $order \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.1 \
  --order $order \
  --root_path $root_path \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

done
