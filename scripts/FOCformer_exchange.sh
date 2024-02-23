export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

#for order in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#for diff in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
for diff in 0.9
do
  
python -u run.py \
  --is_training 1 \
  --diff 0.9 \
  --order 0.8 \
  --root_path /home/liyh/data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --factor 3 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --diff 0.7 \
  --order 0.8 \
  --root_path /home/liyh/data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --factor 3 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --diff 0.1 \
  --order 0.8 \
  --root_path /home/liyh/data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model 512 \
  --d_ff 512 \
  --factor 3 \

python -u run.py \
  --is_training 1 \
  --diff 1.0 \
  --order 0.8 \
  --root_path /home/liyh/data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --factor 3 \
  --itr 1

done
