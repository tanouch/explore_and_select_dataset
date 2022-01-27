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
First, create your index using autofaiss (https://github.com/criteo/autofaiss). Run the following command:
autofaiss build_index --embeddings="/home/u.tanielian/US/img_emb" --index_path="/home/u.tanielian/US/img_clip_knn_test.index" --index_infos_path="/home/u.tanielian/US/index_infos_test.json" --metric_type="ip" --save_on_disk True

1) Explore your dataset with the notebook. 

2) Then run the following command to select a reduced dataset.
    
python explore_dataset_utils.py \
    --im_dir "/home/u.tanielian/US/img_emb" \
    --txt_dir "/home/u.tanielian/US/text_emb" \
    --metadata_dir "/home/u.tanielian/US/metadata" \
    --strategies "image_vs_text_similarity , text_categories , text_constraints" \
    --categories_prompt "this_text_is_in_english, este_texto_es_en_espanol"\
    --positive_constraints_prompt "shoes, trousers, fashion, clothe" \
    --negative_constraints_prompt "human_being, human_face, eyes, head" \
    --ratio 0.45 \
    --ratio_constraints 0.17 \
    --ratio_constraints_neg 0.45 \
    --save_parquet_files true \
    --output_folder "/home/u.tanielian/training_productGen_clip"


3) Download your reduced dataset the img2dataset package from https://github.com/rom1504/img2dataset, as follows:
    
img2dataset \
    --url_list=/home/u.tanielian/training_productGen_clip \
    --output_folder=/home/u.tanielian/training_productGen_images \
    --thread_count=64 \
    --image_size=256 \
    --input_format parquet \
    --output_format webdataset \
    --url_col image_link \
    --caption_col caption \
    --processes_count 15