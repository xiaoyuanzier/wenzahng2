model_name=AutoTimes_Llama
gpu=2

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --data_path m3_yearly_dataset.tsf \
  --test_data_path m4_yearly_dataset.tsf \
  --seasonal_patterns 'Yearly' \
  --model_id m3_Yearly \
  --model $model_name \
  --data tsf \
  --test_pred_len 6 \
  --gpu $gpu \
  --use_amp \
  --llm_ckp_dir ./model_down/Llama \
  --visualize \
  --test_dir zero_shot_forecast_m3_Yearly_AutoTimes_Llama_tsf_sl6_ll0_tl6_lr0.0001_bt16_wd0_hd256_hl3_cosTrue_mixFalse_Exp_0

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --data_path m3_quarterly_dataset.tsf \
  --test_data_path m4_quarterly_dataset.tsf \
  --seasonal_patterns 'Quarterly' \
  --model_id m3_Quarterly \
  --model $model_name \
  --data tsf \
  --test_pred_len 8 \
  --gpu $gpu \
  --use_amp \
  --llm_ckp_dir ./model_down/Llama \
  --visualize \
  --test_dir zero_shot_forecast_m3_Quarterly_AutoTimes_Llama_tsf_sl8_ll0_tl8_lr5e-06_bt16_wd0_hd1024_hl2_cosTrue_mixFalse_Exp_0

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_monthly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly \
  --model $model_name \
  --data tsf \
  --test_pred_len 24 \
  --gpu $gpu \
  --use_amp \
  --llm_ckp_dir ./model_down/Llama \
  --visualize \
  --test_dir zero_shot_forecast_m3_Monthly_AutoTimes_Llama_tsf_sl24_ll0_tl24_lr1e-05_bt16_wd0_hd512_hl2_cosTrue_mixFalse_Exp_0
python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_weekly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly \
  --model $model_name \
  --data tsf \
  --test_pred_len 13 \
  --use_amp \
  --llm_ckp_dir ./model_down/Llama \
  --gpu $gpu \
  --visualize \
  --test_dir zero_shot_forecast_m3_Monthly_AutoTimes_Llama_tsf_sl26_ll13_tl13_lr0.001_bt16_wd0_hd256_hl2_cosTrue_mixFalse_Exp_0

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_daily_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly \
  --model $model_name \
  --data tsf \
  --test_pred_len 14 \
  --gpu $gpu \
  --use_amp \
  --llm_ckp_dir ./model_down/Llama \
  --visualize \
  --test_dir zero_shot_forecast_m3_Monthly_AutoTimes_Llama_tsf_sl28_ll14_tl14_lr0.0001_bt16_wd0_hd256_hl2_cosTrue_mixFalse_Exp_0

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_hourly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly \
  --model $model_name \
  --data tsf \
  --test_pred_len 24 \
  --gpu $gpu \
  --use_amp \
  --llm_ckp_dir ./model_down/Llama \
  --visualize \
  --test_dir zero_shot_forecast_m3_Monthly_AutoTimes_Llama_tsf_sl48_ll24_tl24_lr0.001_bt16_wd0_hd128_hl3_cosTrue_mixFalse_Exp_0
