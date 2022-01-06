import numpy as np
import torch
import torchvision
from torchvision import models, transforms
import clip
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor
from torch.nn import Parameter
from omegaconf import OmegaConf
from PIL import Image
import timm

from tqdm.notebook import tqdm
import gzip
import html
import os
import sys
from functools import lru_cache
import ftfy
import regex as re
import time
torch.manual_seed(0)

from taming_transformers.taming.models import cond_transformer, vqgan

def load_clip_model():
    MODELS = {
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    }
    clip.available_models()
    model, preprocess = clip.load("ViT-B/32")
    model.to('cuda')
    
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()
    vocab_size = model.vocab_size.item()
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    return model, input_resolution, context_length, vocab_size

def show(img):
    npimg = img.cpu().numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def stack_reconstructions(images, text, num_rows, num_columns):
    #assert input.size == x1.size == x2.size == x3.size
    w, h = images[0].shape[1], images[0].shape[2]
    img = Image.new("RGB", (num_columns*w, num_rows*h))
    for i in range(num_rows):
        for j in range(num_columns):
            im = images[i*num_columns+j] 
            #im = preprocess_vqgan(im) if j==0 else im                
            im = custom_to_pil(im)
            img.paste(im, (j*w, i*h))
    #ImageDraw.Draw(img).text(((i%5)*w, int(i/5)), f'{title}', (255, 255, 255), font=font)
    img.save(text+".png")
    return img

def plot_images_reconstruction(images, text, num_rows, num_columns):
    torchvision.utils.save_image(images, text+'.jpg', nrow=num_columns)
    stack_reconstructions(images, text, num_rows, num_columns)
    #plt.clf()
    #fig=plt.figure(figsize=(10, 90), dpi=384)
    #for i in range(len(images)):
    #    ax_i = fig.add_subplot(num_rows, num_columns, i+1)
    #    show(images[i])
    #text = "_".join(text.split())
    #plt.savefig(text[:50], bbox_inches="tight")
           
def plot_images(images, text, num_rows, num_columns):
    plt.clf()
    fig=plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        ax_i = fig.add_subplot(num_rows, num_columns, i+1)
        show(images[i])
    text = "_".join(text.split())
    plt.savefig(text[:50], bbox_inches="tight")
    
def get_additional_layers(num_features):
    hidden_layer = 32
    if hidden_layer:
        return nn.Sequential(nn.Linear(num_features, hidden_layer),
                             nn.ReLU(),
                             nn.Linear(hidden_layer, 1))
    return nn.Linear(num_features, 1)

def get_model_pclick(path):
    model= models.resnet50(pretrained=True)
    model.fc = get_additional_layers(model.fc.in_features)
    model.to('cuda')
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def get_additional_layers(num_features):
    return nn.Sequential(nn.Dropout(p=0.2),
                    nn.Linear(num_features, 1))

def get_model_efficientnet2(filename):
    m = timm.create_model('tf_efficientnetv2_s_in21k', pretrained=True, num_classes=0)
    o = m(torch.randn(2, 3, 224, 224))
    model = nn.Sequential(m, get_additional_layers(o.shape[1]))
    return model

def get_pclick_model(model, transform):
    model.eval()
    def pclik(image):
        print(image.shape)
        image = transforms.ToPILImage(mode="RGB")(image)
        return model(transform(image))
    return pclik


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming_transformers.taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming_transformers.taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming_transformers.taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def reconstruct_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    #x = preprocess_vqgan(x)
    z, _, [_, _, indices] = model.encode(x)
    #print(f"VQGAN: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
