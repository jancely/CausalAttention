export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
order=0.5

#for order in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#for diff in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
for e_layers in 1 2 3 4 5
#for d_model in 32 64 256 512
#for order in 0.5
#for lr in 0.002
do
 
python -u run.py \
  --is_training 1 \
  --diff 0.0 \
  --order $order \
  --root_path /home/liyh/data/economy/shanghaiComposite/ \
  --data_path dshanghai.csv \
  --model_id shanghai_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers $e_layers \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.002 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.0 \
  --order $order \
  --root_path /home/liyh/data/economy/shanghaiComposite/ \
  --data_path dshanghai.csv \
  --model_id shanghai_96_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers $e_layers \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.002 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.0 \
  --order $order \
  --root_path /home/liyh/data/economy/shanghaiComposite/ \
  --data_path dshanghai.csv \
  --model_id shanghai_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers $e_layers \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.002
 

python -u run.py \
  --is_training 1 \
  --diff 0.0 \
  --order $order \
  --root_path /home/liyh/data/economy/shanghaiComposite/ \
  --data_path dshanghai.csv \
  --model_id shanghai_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --learning_rate 0.002

done
