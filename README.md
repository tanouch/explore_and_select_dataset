# Exploring your data with CLIP

## Pre-requisite
1) In your venv:
    pip install --upgrade pip
    pip install -e .
    register-kernel --venv venv --display-name 'explore_dataset'

2) You also need a dataset of :
    - clip image embeddings
    - clip text embeddings
    - metadata
    To get these, use https://github.com/rom1504/clip-retrieval

## Approach
1) Explore your dataset with the notebook. 

2) Then run the following command to select a reduced dataset.
    
python explore_dataset_utils.py \
    --im_dir "/home/img_emb" \
    --txt_dir "/home/text_emb" \
    --metadata_dir "/home/metadata" \
    --strategies "image_vs_text_similarity , text_categories , text_constraints" \
    --categories_prompt "this_text_is_in_english, este_texto_es_en_espanol"\
    --positive_constraints_prompt "fashion, clothe" \
    --negative_constraints_prompt "human face" \
    --ratio 0.35 \
    --output_folder "/home/training_productGen_clip"


4) Download your reduced dataset the img2dataset package from https://github.com/rom1504/img2dataset, as follows:
    
img2dataset \
    --url_list=/home/training_productGen_clip \
    --output_folder=/home/training_productGen_images \
    --thread_count=64 \
    --image_size=256 \
    --input_format parquet \
    --output_format webdataset \
    --url_col image_link \
    --caption_col caption \
    --processes_count 15