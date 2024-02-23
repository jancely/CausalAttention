export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

#for pred_len in 96 192 336 720
#for diff in 0.0
for order in 0.1 0.2 0.3 0.4 0.6 0.5 0.6 0.7 0.8 0.9
#do
python -u run.py \
  --is_training 1 \
  --diff 0.0 \
  --order $order \
  --root_path /home/liyh/data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0002 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.5 \
  --order $order \
  --root_path /home/liyh/data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0002 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.6 \
  --order $order \
  --root_path /home/liyh/data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0002 \
  --itr 1

#python -u run.py \
#  --is_training 1 \
#  --diff 0.4 \
#  --order $order \
#  --root_path /home/liyh/data/ETT/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_96_720 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len 96 \
#  --pred_len 720 \
#  --e_layers 2 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --d_model 512 \
#  --d_ff 512 \
#  --learning_rate 0.0002 \
#  --itr 1
##done
