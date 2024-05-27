

python python mem_benchmark.py \
  --num_features=1000 \
  --seq_len=1000 \
  --time_steps=100000 \
  --mode='regular' \
  --n_epoch=10 \
  --device='cuda' \
  --batch_size=500