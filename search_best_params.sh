#!/usr/bin/env bash

set -e

BASE_DIR="/home/jupyter-yyf-yyy/Program/Timesseriespre/wenzahng2"
LOG_DIR="$BASE_DIR/logs_search"
SEARCH_HISTORY_FILE="$BASE_DIR/search_history_autotimes.txt"

mkdir -p "$LOG_DIR"
touch "$SEARCH_HISTORY_FILE"

MODEL_NAME="AutoTimes_Llama"
DATASET="ETTh1"
GPU=1
NUM_TRIALS=20

# 提取 MSE 的辅助函数
metric_from_log() {
  local log_file="$1"
  [ ! -f "$log_file" ] && echo "" && return
  grep "mse:" "$log_file" | tail -n 1 | sed -E 's/.*mse:([0-9.eE+-]+), mae:.*/\1/'
}

# 提取 MAE 的辅助函数
metric_mae_from_log() {
  local log_file="$1"
  [ ! -f "$log_file" ] && echo "" && return
  grep "mse:" "$log_file" | tail -n 1 | sed -E 's/.*mae:([0-9.eE+-]+).*/\1/'
}

for pred_len in 96 192 336 720; do
  echo "Processing $DATASET, pred_len=$pred_len..."
  
  for ((i=1; i<=NUM_TRIALS; i++)); do
    # 调用自适应建议脚本获取下一组参数
    combo=$(python "$BASE_DIR/adaptive_suggest_autotimes.py" \
            --history_file "$SEARCH_HISTORY_FILE" \
            --dataset "$DATASET" \
            --pred_len "$pred_len")
    
    read sl tl lr hl wd <<< "$combo"
    
    tag="ada_trial${i}_sl${sl}_tl${tl}_lr${lr}_hl${hl}_wd${wd}"
    log_file="$LOG_DIR/${DATASET}_pl${pred_len}_${tag}.log"

    # 检查是否已跑过
    mse=$(metric_from_log "$log_file")
    if [ -n "$mse" ]; then
      echo "Skip finished: $tag"
    else
      echo "Running Trial $i/$NUM_TRIALS: $tag"
      
      # 训练
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id "${DATASET}_sl${sl}_pl${pred_len}" \
        --model $MODEL_NAME \
        --data $DATASET \
        --seq_len "$sl" \
        --label_len 576 \
        --token_len "$tl" \
        --batch_size 128 \
        --learning_rate "$lr" \
        --mlp_hidden_layers "$hl" \
        --weight_decay "$wd" \
        --train_epochs 5 \
        --patience 3 \
        --use_amp \
        --gpu $GPU \
        --cosine \
        --mix_embeds \
        --llm_ckp_dir /home/jupyter-yyf-yyy/Program/Timesseriespre/AutoTimes1/model_down/Llama \
        --des "$tag" > "$log_file" 2>&1

      # 记录结果到历史文件
      mse=$(metric_from_log "$log_file")
      mae=$(metric_mae_from_log "$log_file")
      if [ -n "$mse" ]; then
        echo "data=$DATASET, pred_len=$pred_len, seq_len=$sl, token_len=$tl, learning_rate=$lr, mlp_hidden_layers=$hl, weight_decay=$wd, mse=$mse, mae=$mae, log=$log_file" >> "$SEARCH_HISTORY_FILE"
      fi
    fi
  done
done

echo "Search completed. History saved in $SEARCH_HISTORY_FILE"
