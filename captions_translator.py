from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

from argparse import ArgumentParser
import os
from config import DATA_DIR
from utils import read_json_file, write_json_file, init_seed 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int,  default=512, help="batch of texts being translated")
    parser.add_argument("--max_length", type=int, default=100, help="model generated max length")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="source language")
    parser.add_argument("--tgt_lang", type=str, default="vie_Latn", help="target language")
    return parser.parse_args()

class TranslateCaption():
    """
    Create translation annotations with additional list of tokens for each caption.
    """
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang
        self.annotation_dir = os.path.join(DATA_DIR, "annotations")
        self.splits = ["train", "val"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model.to(self.device)
        
    def translate_batch(self, batch):
        captions = [item["caption"] for item in batch]
        inputs = self.tokenizer(text=captions, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)
        translated_tokens = self.model.generate(**inputs, max_length=self.max_length, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang])
        translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        tokenized_captions = [self.tokenizer.tokenize(c) for c in translations]
        translations_batch = []
        for item, translation, tokens in zip(batch, translations, tokenized_captions):
            t = {
                "image_id": item["image_id"],
                "id": item["id"],
                "caption": translation,
                "tokens": tokens,
            }
            translations_batch.append(t)
        return translations_batch
        
    def convert(self):
        for split in self.splits:
            annotation_path = os.path.join(self.annotation_dir, f"captions_{split}2017.json")
            data = read_json_file(annotation_path)
            annotations = data["annotations"]
            convert_annotations = []
            for i in tqdm(range(0, len(annotations), self.batch_size)):
                batch = annotations[i: i + self.batch_size]
                batch_annotations = self.translate_batch(batch)
                convert_annotations.extend(batch_annotations)

            data["annotations"] = convert_annotations
            write_json_file(os.path.join(self.annotation_dir, f"vie_captions_{split}2017.json"), data)

if __name__ == "__main__":
    args = parse_args()
    init_seed(seed=0)
    caption_translator = TranslateCaption(args)
    caption_translator.convert()

