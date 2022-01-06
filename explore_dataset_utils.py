import os, fire, fsspec, subprocess
import torch, torchvision, clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn.functional as F
import pyarrow.parquet as pq
from tqdm import tqdm
from itertools import product
from scipy.special import softmax

import webdataset as wds
from torchvision import transforms as T
from PIL import Image, ImageFont, ImageDraw 
from pathlib import Path
from dalle_pytorch.tokenizer import tokenizer
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import ipywidgets as widgets
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_histograms(distances, txt):
    fig, ax1 = plt.subplots()
    ax1.hist(distances, bins=20)
    ax2 = ax1.twinx()
    ax1.hist(distances, alpha=0.25, cumulative=True, color='red', bins=20)
    plt.savefig(txt + '.jpg')

    
def measuring_similarity(im_embs, txt_embs, txt_inputs, plot=False):
    distances = list()
    batch_size = 25000
    num_batches = int(len(im_embs)/batch_size)
    if len(im_embs)==len(txt_embs):
        for txt_batch, im_batch in zip(np.array_split(txt_embs[:num_batches*batch_size], num_batches), \
            np.array_split(im_embs[:num_batches*batch_size], num_batches)):
            txt_batch, im_batch = torch.from_numpy(txt_batch).to(device), torch.from_numpy(im_batch).to(device)
            distances.append(torch.sum(torch.multiply(im_batch, txt_batch), dim=1).detach().cpu().numpy())
    else:
        for im_batch in np.array_split(im_embs[:num_batches*batch_size], num_batches):
            txt_batch, im_batch = torch.transpose(torch.from_numpy(txt_embs).to(device), 0, 1), torch.from_numpy(im_batch).to(device)
            distances.append(torch.matmul(im_batch, txt_batch).detach().cpu().numpy())
    distances = np.concatenate(distances, axis=0)
    distances = np.reshape(distances, (-1, len(txt_inputs)))
    if plot:
        for i, txt_input in enumerate(txt_inputs):
            plot_histograms(distances[:,i], txt_input)
    return distances


def splitting_based_on_im_and_txt_similarity(im_embs, text_embs, ratio):
    distances = measuring_similarity(im_embs, text_embs, ["Image and Text CLIP similarity"])
    threshold_index = int(ratio*len(distances))
    distances = np.reshape(distances, (-1,))
    indexes = np.argpartition(-distances, threshold_index)[:threshold_index]
    return indexes


def splitting_based_on_zero_shot_classification(model, embs, text_inputs, softmax_bool, ratio, \
                                                positivity_constraints=None, plot=False):
    text_features = torch.cat([clip.tokenize(f"{c}") for c in text_inputs]).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.detach().cpu().numpy()
    distances = measuring_similarity(embs, text_features, text_inputs)
    threshold_index = int(ratio*len(distances))
        
    if softmax_bool:
        distances = softmax(distances, axis=1)
        if plot:
            plot_histograms(distances[:,0], text_inputs[0])
        return np.argpartition(-distances[:,0], threshold_index)[:threshold_index]
    else:
        indexes = np.arange(len(embs))
        for i in range(len(text_inputs)):
            if positivity_constraints[i]:
                indexes = np.intersect1d(indexes, np.argpartition(-distances[:,i], threshold_index)[:threshold_index])
            else:
                indexes = np.setdiff1d(indexes, np.argpartition(-distances[:,i], threshold_index)[:threshold_index])
        return indexes


def splitting_based_on_clustering(index, centroids, threshold_index, num_clusters):
    indexes = {str(i): [] for i in range(num_clusters)}
    distances, neighbors = index.search(centroids.astype(np.float32), threshold_index)
    for i in range(num_clusters):
        indexes[str(i)] = neighbors[i]
    return indexes


def load_embeddings(directory):
    list_files = os.listdir(directory)[1:5]
    clip_files = sorted([directory+'/'+file for file in list_files if 'npy' in file])
    embs = [np.load(file) for file in clip_files]
    embs = np.concatenate(embs, axis=0)
    return embs


def load_metadata(directory):
    list_files = os.listdir(directory)
    list_files = sorted([directory+'/'+file for file in list_files if 'parquet' in file])
    metadata = [pq.read_table(file).to_pandas().to_numpy() for file in list_files]
    metadata = np.concatenate(metadata, axis=0)
    column_names = list(pq.read_table(list_files[0]).to_pandas().columns) 
    return metadata, column_names


def save_parquet(fs, data, data_columns, output_path_metadata):
    df = pd.DataFrame(data=data, columns=data_columns)
    with fs.open(output_path_metadata, "wb") as f:
        df.to_parquet(f)

def imagetransform(b):
    return Image.open(BytesIO(b))

def tokenize(s):
    return s.decode('utf-8')#.replace("\n", "")
    #tokenizer.tokenize(s.decode('utf-8'), TEXT_SEQ_LEN, truncate_text=True).squeeze(0)

def custom_to_pil(x):
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def stack_reconstructions(images, texts, title, num_rows, num_columns):
    w, h = images.shape[2], images.shape[3]
    img = Image.new("RGB", (num_columns*w, num_rows*h))
    for i in range(num_rows):
        for j in range(num_columns):
            im = images[i*num_columns+j] 
            txt = texts[i*num_columns+j]
            im = custom_to_pil(im)
            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype("arial.ttf", 16)
            draw.text((20,225), txt[:50], (0,0,0), font=font)
            img.paste(im, (j*w, i*h))
    plt.figure(figsize=(3*num_columns, 3*num_rows))
    plt.imshow(img, aspect='auto')
    plt.axis('off')
    plt.show()

def create_your_dataloader(
    folder_images=None,
    dataset_size=None,
    batch_size=50,
    IMAGE_SIZE = 256,
    TEXT_SEQ_LEN = 77,
    RESIZE_RATIO = 0.75,
    ):
    
    assert folder_images is not None
    assert dataset_size is not None
    imagepreproc = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')
        if img.mode != 'RGB' else img),
        T.RandomResizedCrop(IMAGE_SIZE,
                            scale=(RESIZE_RATIO, 1.),
                            ratio=(1., 1.)),
        T.ToTensor(),
    ])    

    dataset = [str(p) for p in Path(folder_images).glob("**/*") if ".tar" in str(p).lower()][-1:]
    myimg, mycap = "jpg", "txt"
    image_text_mapping = {
        myimg: imagetransform,
        mycap: tokenize
    }
    image_mapping = {
        myimg: imagepreproc
    }

    ds = (
            wds.WebDataset(dataset)
            .map_dict(**image_text_mapping)     
            .map_dict(**image_mapping)
            .to_tuple(mycap, myimg)
            .batched(batch_size, partial=True) #avoid partial batches when using Distributed training
        )
    dl = wds.WebLoader(ds, batch_size=None, shuffle=False, num_workers=4) # optionally add num_workers=2 (n) argument
    number_of_batches = dataset_size // (batch_size)
    dl = dl.slice(number_of_batches)
    dl.length = number_of_batches
    return dl


def get_sub_dataset(
    im_dir,
    txt_dir, 
    metadata_dir,
    strategies,
    output_folder,
    ratio,
    categories_prompt="",
    positive_constraints_prompt="",
    negative_constraints_prompt="",
    max_num_files_analysis = None,
    num_images_to_plot=None,
    save_parquet_files=True
    ):
    
    assert im_dir is not None
    assert txt_dir is not None
    assert metadata_dir is not None
    assert strategies is not None
    assert output_folder is not None
    os.makedirs(output_folder, exist_ok=True)

    im_files, txt_files, metadata_files = os.listdir(im_dir), os.listdir(txt_dir), os.listdir(metadata_dir)
    im_files = sorted([im_dir+'/'+file for file in im_files if 'npy' in file])
    txt_files = sorted([txt_dir+'/'+file for file in txt_files if 'npy' in file])
    metadata_files = sorted([metadata_dir+'/'+file for file in metadata_files if 'parquet' in file])
    if max_num_files_analysis==None:
        max_num_files_analysis = len(im_files)
    
    if "text_categories" in strategies:
        if categories_prompt != "":
            categories_prompt = [s.strip().replace("_", " ") for s in categories_prompt]
            print("Number of categories", len(categories_prompt))
            print(categories_prompt)
            print("")
                
    if "text_constraints" in strategies:
        pos_list, neg_list = list(), list()
        if positive_constraints_prompt != "":
            pos_list = [s.strip().replace("_", " ") for s in positive_constraints_prompt]
            print("Number of positive constraints", len(pos_list))
            print(pos_list)
            print("")
        if negative_constraints_prompt != "":
            neg_list = [s.strip().replace("_", " ") for s in negative_constraints_prompt]
            print("Number of negative constraints", len(neg_list))
            print(neg_list)
            print("")
        constraint_prompts = pos_list + neg_list
        positivity_constraints = [True]*len(pos_list) + [False]*len(neg_list)
        
    if "text_categories" in strategies or "text_constraints" in strategies:
        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    
    i = 0
    for im_file, txt_file, metadata_file in tqdm(zip(im_files, txt_files, metadata_files)):
        im_embs = np.load(im_file)
        txt_embs = np.load(txt_file)
        metadata = pq.read_table(metadata_file).to_pandas()
        assert len(im_embs) == len(txt_embs) == len(metadata)
        column_names = list(metadata.columns)
        metadata = metadata.to_numpy()
        indexes1, indexes2, indexes3 = np.arange(len(im_embs)), np.arange(len(im_embs)), np.arange(len(im_embs))

        if "image_vs_text_similarity" in strategies:
            indexes1 = splitting_based_on_im_and_txt_similarity(im_embs, txt_embs, ratio)

        if "text_categories" in strategies:
            indexes2 = splitting_based_on_zero_shot_classification(clip_model, txt_embs, categories_prompt, \
                                                                  softmax_bool=True, ratio=ratio)

        if "text_constraints" in strategies:
            indexes3 = splitting_based_on_zero_shot_classification(clip_model, im_embs, constraint_prompts, \
                                                                   softmax_bool=False, ratio=ratio, \
                                                                   positivity_constraints=positivity_constraints)

        indexes = np.array(list(set(indexes1).intersection(indexes2, indexes3)))
        print("Ratio kept", len(indexes)/len(im_embs))
        if save_parquet_files:
            fs, _ = fsspec.core.url_to_fs(".")
            save_parquet(fs, metadata[indexes], column_names, output_folder + "/" + "metadata_" + str(i) + ".parquet")
        i += 1
        if i>=max_num_files_analysis:
            break
    
    if num_images_to_plot is not None:
        sub_indexes = np.random.choice(len(indexes), size=num_images_to_plot, replace=False)
        sub_metadata = metadata[indexes[sub_indexes]]
        return sub_metadata, column_names
    
    
def show_a_subset_of_selected_dataset(
    im_dir, \
    txt_dir, \
    metadata_dir, \
    strategies, \
    output_folder, \
    output_folder_images, \
    ratio, \
    categories_prompt, \
    positive_constraints_prompt, \
    negative_constraints_prompt, \
    max_num_files_analysis, \
    num_images_to_plot, \
    batch_size, \
    num_rows, \
    num_columns, \
    num_batches_to_show
    ):
    
    sub_metadata, column_names = get_sub_dataset(im_dir, txt_dir, metadata_dir, strategies, \
                                                 output_folder, ratio, categories_prompt, \
                                                 positive_constraints_prompt, negative_constraints_prompt, \
                                                 max_num_files_analysis, num_images_to_plot)
    fs, _ = fsspec.core.url_to_fs(".")
    save_parquet(fs, sub_metadata, column_names, output_folder + "/" + "metadata_0.parquet")
    
    subprocess.run(['img2dataset', '--url_list', output_folder, \
                     '--output_folder', output_folder_images, \
                     '--thread_count', '64', '--image_size', '256', '--input_format', 'parquet', \
                     '--output_format', 'webdataset', '--url_col', 'image_link', 
                     '--caption_col', 'caption', '--processes_count', '15'])
    dl = create_your_dataloader(folder_images=output_folder_images, dataset_size=num_images_to_plot, \
                               batch_size=batch_size)
    for i, batch in enumerate(dl):
        (texts, images) = batch
        stack_reconstructions(images, texts, title="test", num_rows=5, num_columns=7)
        if i>=num_batches_to_show:
            break


def create_interface_ipywidget():
    
    categorie_constraints_with_prompt = widgets.Text(
        value=None,
        description='Categories',
        indent=False
    )

    positive_constraints_with_prompt = widgets.Text(
        value=None,
        description='Pos. constraints',
        disabled=False,
        indent=False
    )

    negative_constraints_with_prompt = widgets.Text(
        value=None,
        description='Neg. constraints',
        disabled=False,
        indent=False
    )

    ratio_slider = widgets.FloatSlider(
             value=0.5,
             description='Ratio:',
             min=0.1,
             max=1,
             step=0.05)

    query_with_text_im_similarity = widgets.Checkbox(
        value=False,
        description='Query with text-im similarity',
        disabled=False,
        indent=False
    )
    query_with_categories = widgets.Checkbox(
        value=False,
        description='Query with categories',
        disabled=False,
        indent=False
    )
    query_with_text_constraints = widgets.Checkbox(
        value=False,
        description='Query with text constraints',
        disabled=False,
        indent=False
    )
    button = widgets.Button(description='Show subset!')
    out = widgets.Output()
    return query_with_text_im_similarity, \
          query_with_categories, \
          query_with_text_constraints, \
          categorie_constraints_with_prompt, \
          positive_constraints_with_prompt, \
          negative_constraints_with_prompt, \
          ratio_slider, \
          button, \
          out            
            
if __name__ == "__main__":
    fire.Fire(get_sub_dataset)