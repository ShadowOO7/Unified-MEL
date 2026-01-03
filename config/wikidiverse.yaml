run_name: "Wikidiverse"
seed: 42

ent:
  model_dir: "/home/kmpooja/UniMEL/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct"

  train_data_dir: "/home/kmpooja/UniMEL/wikidiverse/entities/entity2brief_train.json"
  train_output_dir: "/home/kmpooja/UniMEL/results/entity2sum_train.json"

  val_data_dir: "/home/kmpooja/UniMEL/wikidiverse/entities/entity2brief_valid.json"
  val_output_dir: "/home/kmpooja/UniMEL/results/entity2sum_valid.json"

  test_data_dir: "/home/kmpooja/UniMEL/wikidiverse/entities/entity2brief_test.json"
  test_output_dir: "/home/kmpooja/UniMEL/results/entity2sum_test.json"

mention:
  # IMAGE DESCRIPTION MODEL
  model_dir_img: "/home/kmpooja/UniMEL/models/llava-v1.6-mistral-7b-hf/llava-v1.6-mistral-7b-hf"

  train_mentions_dir: "/home/kmpooja/UniMEL/wikidiverse/candidates/train_w_10cands_pos_source.json"
  train_save_dir: "/home/kmpooja/UniMEL/results/train_w_10cands_pos_source_des_llava.json"

  val_mentions_dir: "/home/kmpooja/UniMEL/wikidiverse/candidates/valid_w_10cands_pos_source.json"
  val_save_dir: "/home/kmpooja/UniMEL/results/valid_w_10cands_pos_source_des_llava.json"

  test_mentions_dir: "/home/kmpooja/UniMEL/wikidiverse/candidates/test_w_10cands_pos_source.json"
  test_save_dir: "/home/kmpooja/UniMEL/results/test_w_10cands_pos_source_des_llava.json"

  train_kb_img_folder: "/home/kmpooja/UniMEL/wikidiverse/mention_images/train_image/train_"
  val_kb_img_folder: "/home/kmpooja/UniMEL/wikidiverse/mention_images/valid_image/valid_"
  test_kb_img_folder: "/home/kmpooja/UniMEL/wikidiverse/mention_images/test_image/test_"

  # TEXT DESCRIPTION MODEL
  model_dir_text: "/home/kmpooja/UniMEL/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct"

  train_data_dir: "/home/kmpooja/UniMEL/results/train_w_10cands_pos_source_des_llava.json"
  train_output_dir: "/home/kmpooja/UniMEL/results/train_w_10cands_pos_source_des_llava_llama.json"

  val_data_dir: "/home/kmpooja/UniMEL/results/valid_w_10cands_pos_source_des_llava.json"
  val_output_dir: "/home/kmpooja/UniMEL/results/valid_w_10cands_pos_source_des_llava_llama.json"

  test_data_dir: "/home/kmpooja/UniMEL/results/test_w_10cands_pos_source_des_llava.json"
  test_output_dir: "/home/kmpooja/UniMEL/results/test_w_10cands_pos_source_des_llava_llama.json"

embed:
  max_length: 4096
  emb_model_dir: "/home/kmpooja/UniMEL/models/SFR-Embedding-Mistral/SFR-Embedding-Mistral"

  train_data_dir: "/home/kmpooja/UniMEL/results/entity2sum_train.json"
  train_embed_dir: "/home/kmpooja/UniMEL/results/embedding_SFR_train.json"

  val_data_dir: "/home/kmpooja/UniMEL/results/entity2sum_valid.json"
  val_embed_dir: "/home/kmpooja/UniMEL/results/embedding_SFR_valid.json"

  test_data_dir: "/home/kmpooja/UniMEL/results/entity2sum_test.json"
  test_embed_dir: "/home/kmpooja/UniMEL/results/embedding_SFR_test.json"

top:
  K: 5
  max_length: 4096
  model_dir: "/home/kmpooja/UniMEL/models/SFR-Embedding-Mistral/SFR-Embedding-Mistral"
  
  # 🔁 USE MERGED EMBEDDINGS FOR ALL SPLITS
  train_database_emb: "/home/kmpooja/UniMEL/results/embedding_SFR_train.json"
  val_database_emb: "/home/kmpooja/UniMEL/results/embedding_SFR_train.json"
  test_database_emb: "/home/kmpooja/UniMEL/results/embedding_SFR_train.json"

  train_database_sum: "/home/kmpooja/UniMEL/results/entity2sum_train.json"
  train_mention_dir: "/home/kmpooja/UniMEL/results/train_w_10cands_pos_source_des_llava_llama.json"
  train_mention_topK_dir: "/home/kmpooja/UniMEL/results/train_w_10cands_pos_source_des_llava_llama_topK.json"

  val_database_sum: "/home/kmpooja/UniMEL/results/entity2sum_train.json"
  val_mention_dir: "/home/kmpooja/UniMEL/results/valid_w_10cands_pos_source_des_llava_llama.json"
  val_mention_topK_dir: "/home/kmpooja/UniMEL/results/valid_w_10cands_pos_source_des_llava_llama_topK.json"

  test_database_sum: "/home/kmpooja/UniMEL/results/entity2sum_train.json"
  test_mention_dir: "/home/kmpooja/UniMEL/results/test_w_10cands_pos_source_des_llava_llama.json"
  test_mention_topK_dir: "/home/kmpooja/UniMEL/results/test_w_10cands_pos_source_des_llava_llama_topK.json"

infer:
  max_length: 2048
  model_id: "qwen/Qwen2-7B-Instruct"
  ckpt_id: null

  train_database_sum: "/home/kmpooja/UniMEL/results/entity2sum_train.json"
  train_mention_topK_dir: "/home/kmpooja/UniMEL/results/train_w_10cands_pos_source_des_llava_llama_topK.json"
  train_res_topK_dir: "/home/kmpooja/UniMEL/results/res_train_top5_SFR.json"

  val_database_sum: "/home/kmpooja/UniMEL/results/entity2sum_valid.json"
  val_mention_topK_dir: "/home/kmpooja/UniMEL/results/valid_w_10cands_pos_source_des_llava_llama_topK.json"
  val_res_topK_dir: "/home/kmpooja/UniMEL/results/res_valid_top5_SFR.json"

  test_database_sum: "/home/kmpooja/UniMEL/results/entity2sum_test.json"
  test_mention_topK_dir: "/home/kmpooja/UniMEL/results/test_w_10cands_pos_source_des_llava_llama_topK.json"
  test_res_topK_dir: "/home/kmpooja/UniMEL/results/res_test_top5_SFR.json"
