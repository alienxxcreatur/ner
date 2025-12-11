export MODEL=bert-base-cased  # برای شروع، همین مدل عمومی را استفاده کن
epoch=5  # برای تست اولیه 5 اپوک کافی است
lr=5e-5
wis=1qq3qq5qq7
data_type=ncbi
connect_type=dot-att

CUDA_VISIBLE_DEVICES=0 python run_hgn.py \
  --train_data_dir=/content/hgn_project/data/$data_type/train_merge.txt \
  --dev_data_dir=/content/hgn_project/data/$data_type/dev.txt \
  --test_data_dir=/content/hgn_project/data/$data_type/test.txt \
  --bert_model=${MODEL} \
  --task_name=ner \
  --output_dir=/content/hgn_project/output/ncbi_hgner \
  --max_seq_length=128 \
  --num_train_epochs ${epoch} \
  --do_train \
  --gpu_id 0 \
  --learning_rate ${lr} \
  --warmup_proportion=0.1 \
  --train_batch_size=16 \
  --use_bilstm True \
  --use_multiple_window True \
  --windows_list="${wis}" \
  --connect_type=${connect_type} \
  --use_dconv \
  --use_gate
