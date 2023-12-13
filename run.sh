CUDA_VISIBLE_DEVICES=0 python main.py \
      --input_file 'data/testa/annos.jsonl' \
      --detector_file 'data/dets_dict.json' \
      --image_root 'data/testa/images' \
      --clip_model 'ViT-B/32,RN50x16' \
      --results_path 'result/result_testa.json' \
      --cache_path 'cache/test' \
      --box_representation_method 'blur,circle,crop,gray,edge,blur-circle,gray-circle' \
      --method 'baseline' \
      --device 0 \
      --dynamic_color_width True \
      --fine_grained True \
      --sam_cache_path 'cache/sam' \
      --sam_temp_path 'cache/sam_tmp' \
      --resize_square True \
      --enlarge_boxes 0 \
      --size_Q 300 \
      --box_area_threshold 0.06 \
      --img_cache_path 'cache/img'

