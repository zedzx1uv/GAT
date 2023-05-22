for k in 3 10 20 30
do
CUDA_VISIBLE_DEVICES=0 python train_kl.py \
                               --do_train \
                               --sst \
                               --adv_K $k \
                               --num_labels 2 \
                               --epochs 10 \
                               --batch_size 64 \
                               --base_model bert \
                               --model_type bert-base-uncased \
                               --save_path path-to-yours/ > log-path/log.out 2>&1 
done
