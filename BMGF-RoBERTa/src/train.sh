python /home/CE/musaeed/BMGF-RoBERTa/src/train.py \
    --train_dataset_path /home/CE/musaeed/BMGF-RoBERTa/processed_data/train_implicit_roberta.pt \
    --valid_dataset_path /home/CE/musaeed/BMGF-RoBERTa/processed_data/valid_implicit_roberta.pt \
    --test_dataset_path /home/CE/musaeed/BMGF-RoBERTa/processed_data/test_implicit_roberta.pt \
    --save_model_dir /home/CE/musaeed/BMGF-RoBERTa/trained_model/dumps \
    --num_rels 4 \
    --gpu_ids 3,4,5,1 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --max_grad_norm 2.0 \
    --loss ce \
    --encoder roberta \
    --finetune type \
    --hidden_dim 128 \
    --num_perspectives 16 \
    --num_filters 64 \
    --activation leaky_relu \
    --dropout 0.2