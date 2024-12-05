export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

experiment_name=Benchmark
experiment_id=exp_001
model_name=Transformer
root_path=./dataset/m4
task_name=short_term_forecast
data=m4
features=M
e_layers=2
d_layers=1
factor=3
enc_in=1
dec_in=1
c_out=1
batch_size=16
d_model=512
description=Exp
iterations=1
learning_rate=0.001
loss=SMAPE

for seasonal_pattern in Monthly Yearly Quarterly Weekly Daily Hourly; do
  model_id="m4_${seasonal_pattern}"
  python -u run.py \
    --experiment_name $experiment_name \
    --experiment_id $experiment_id \
    --task_name $task_name \
    --is_training 1 \
    --root_path $root_path \
    --seasonal_patterns $seasonal_pattern \
    --model_id $model_id \
    --model $model_name \
    --data $data \
    --features $features \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor $factor \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --batch_size $batch_size \
    --d_model $d_model \
    --des $description \
    --itr $iterations \
    --learning_rate $learning_rate \
    --loss $loss \
    --use_gpu True \
    --use_multi_gpu --devices '0,1,2,3,4,5,6'
done