# Exploring your data with CLIP

## Pre-requisite


Also use the img2dataset package from https://github.com/rom1504/img2dataset.

### The different strategies

python explore_dataset_utils.py \
    --im_dir "/home/u.tanielian/US/img_emb" \
    --txt_dir "/home/u.tanielian/US/text_emb" \
    --metadata_dir "/home/u.tanielian/US/metadata" \
    --strategies "image_vs_text_similarity , text_categories , text_constraints" \
    --categories_prompt "this_text_is_in_english, este_texto_es_en_espanol"\
    --positive_constraints_prompt "fashion, clothe" \
    --negative_constraints_prompt "human face" \
    --ratio 0.35 \
    --output_folder "/home/u.tanielian/US/training_productGen_clip"
    
img2dataset \
    --url_list=/home/u.tanielian/US/training_productGen_clip \
    --output_folder=/home/u.tanielian/US/training_productGen_images \
    --thread_count=64 \
    --image_size=256 \
    --input_format parquet \
    --output_format webdataset \
    --url_col image_link \
    --caption_col caption \
    --processes_count 15