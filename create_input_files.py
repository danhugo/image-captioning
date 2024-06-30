# image binary files
# caption, caption length in a file
# image_path, captions, word to index in a file

import os
import random
import h5py
from tqdm import tqdm
import random
from collections import Counter
from config import DATA_DIR
from argparse import ArgumentParser
from utils import read_json_file, write_json_file, init_seed
from PIL import Image
import numpy as np
from pathlib import Path

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--min_word_freq", type=int, default=5, help="vocab min word-frequency")
    parser.add_argument("--captions_per_image", type=int, default=3, help="num of captions per image")
    parser.add_argument("--max_caption_len", type=int, default=100, 
                        help="max caption length in data. Captions whose length are larger will be filtered.")
    args = parser.parse_args()
    return args


class CreateInput():
    def __init__(self, args):
        self.min_word_freq = args.min_word_freq
        self.captions_per_image = args.captions_per_image
        self.max_caption_len = args.max_caption_len
        self.annotation_dir = os.path.join(DATA_DIR, "annotations")
        self.processed_path = os.path.join(self.annotation_dir, f"processed_vie_captions_2017.json")
        self.wordmap_path = os.path.join(self.annotation_dir, f"word_map_freq{self.min_word_freq}.json")
        self.h5py_dir = os.path.join(DATA_DIR, "h5py")
        self.caps_len_dir = os.path.join(DATA_DIR, "captions")
        Path(self.h5py_dir).mkdir(parents=True, exist_ok=True)
        Path(self.caps_len_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def split_train_test(data, test_size=0.1):
        random.shuffle(data)
        split_index = int(len(data) * (1-test_size))
        train_data = data[:split_index]
        test_data = data[split_index:]
        return train_data, test_data
    
    @staticmethod
    def associate_captions_with_images(images, captions_dict, split, subset):
        captions_list = []
        for item in tqdm(images, desc=f"{subset}: Associate captions with image paths"):
            file_name = item["file_name"]
            id = str(item["id"])
            captions = captions_dict[id]
            relative_path = f"{split}2017/{file_name}"
            captions_list.append({
                "file_path": relative_path,
                "captions": captions
                })
        return captions_list


    def process_annotations(self):
        """Collect list of captions corresponding to an image id."""
        pro_data = {}
        for split in ['train', 'val']:
            # print(split)
            annotation_path = os.path.join(self.annotation_dir, f"vie_captions_{split}2017.json")
            data = read_json_file(annotation_path)

            annotations = data["annotations"]
            captions_dict = {}

            for ann in tqdm(annotations, desc="collecting captions"):
                image_id = str(ann["image_id"])
                caption = ann["caption"]
                tokens = ann["tokens"]
                d = {"caption": caption, "tokens": tokens}
                if image_id in captions_dict:
                    captions_dict[image_id].append(d)
                else:
                    captions_dict[image_id] = [d]

            if split == "train":
                train_images, test_images = self.split_train_test(data["images"], test_size=0.1)
                for images, dataset in [(train_images, 'train'), (test_images, 'test')]:
                    captions_list = self.associate_captions_with_images(images, captions_dict, split, dataset)
                    pro_data[dataset] = captions_list
            else:
                images = data["images"]
                dataset = 'val'
                captions_list = self.associate_captions_with_images(images, captions_dict, split, dataset)
                pro_data[dataset] = captions_list

        print("writing processed data to file")
        write_json_file(self.processed_path, pro_data)
        return pro_data

    def create_word_map(self, data):
        """Create a dict mapping word to id"""
        word_counter = Counter()
        for split, split_data in data.items():
            print(split)
            for d in tqdm(split_data, desc="Updating word counter"):
                captions = d["captions"]
                for caption in captions:
                    word_counter.update(caption["tokens"])

        words = [w for w in word_counter.keys() if word_counter[w] > self.min_word_freq]
        word_map = {word: i + 1 for i, word in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0
        write_json_file(self.wordmap_path, word_map)
        return word_map

    @staticmethod
    def _process_image(path):
        np_img = np.array(Image.open(path))
        if len(np_img.shape) == 2:
            np_img = np_img[:, :, np.newaxis]
            np_img = np.concatenate([np_img, np_img, np_img], axis=2)
        
        pil_img = Image.fromarray(np_img)
        pil_img = pil_img.resize((256, 256), Image.LANCZOS)
        np_img = np.array(pil_img)
        np_img = np_img.transpose(2, 0, 1)
        assert np_img.shape == (3, 256, 256)
        assert np.max(np_img) <= 255
        return np_img

    def process_images_and_captions(self, data, word_map):
        """
        Save images in np.darray to hdf5 for compression and efficient data loading.
        """
        for split in ['train', 'val', 'test']:
            basename = f"{split}_caps{self.captions_per_image}_freq{self.min_word_freq}_maxlen{self.max_caption_len}"
            h5py_path = os.path.join(self.h5py_dir, f"{basename}.hdf5")
            caps_len_path = os.path.join(self.caps_len_dir, f"{basename}.json")

            with h5py.File(h5py_path, "w") as f:
                f.attrs['captions_per_image'] = self.captions_per_image
                data_type = np.dtype([('img_path', h5py.string_dtype(encoding='utf-8')), ('value', 'uint8', (3, 256, 256))])
                images = f.create_dataset('images', (len(data[split]), ), dtype=data_type)

                enc_data = []
                for i, d in enumerate(tqdm(data[split], desc=f"{split}: saving images to h5py")):
                    enc_captions = []
                    caplens = []
                    relative_path = d["file_path"]
                    img_path = os.path.join(DATA_DIR, relative_path)
                    img_captions = [c["tokens"] for c in d["captions"] if len(c["tokens"]) <= self.max_caption_len]

                    if len(img_captions) < self.captions_per_image:
                        img_captions = img_captions + [random.choice(img_captions) for _ in range(self.captions_per_image - len(img_captions))]
                    else:
                        img_captions = random.sample(img_captions, k=self.captions_per_image)

                    assert len(img_captions) == self.captions_per_image 

                    images[i] = (relative_path, self._process_image(img_path))

                    for c in img_captions:
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                            word_map['<end>']] + [word_map['<pad>']] * (self.max_caption_len - len(c))

                        c_len = len(c) + 2

                        enc_captions.append(enc_c)
                        caplens.append(c_len)
                    
                    enc_data.append({
                        "img_path": relative_path,
                        "enc_captions": enc_captions,
                        "caplens": caplens
                    })

            write_json_file(caps_len_path, enc_data)

    def create(self):
        if os.path.isfile(self.processed_path):
            data = read_json_file(self.processed_path)
        else:
            data = self.process_annotations()

        if os.path.isfile(self.wordmap_path):
            word_map = read_json_file(self.wordmap_path)
        else:
            word_map = self.create_word_map(data)

        self.process_images_and_captions(data, word_map)

        
if __name__ == "__main__":
    args = parser_args()
    init_seed(0)
    creator = CreateInput(args)
    creator.create()