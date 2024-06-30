from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from dataset import CaptionDataset, caption_transforms
from rnn import LstmTranslator
from utils import read_json_file, init_seed, write_json_file
import os
from config import DATA_DIR
from tqdm import tqdm
from torchmetrics.text import BLEUScore
import json


def parser_args():
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for dataloader")
    parser.add_argument("--checkpoint", type=str, default='exp/caps3_freq5_maxlen100_20240628_031101.pt',
                        help="path to checkpoint")
    parser.add_argument("--epochs", type=int, default=120,
                        help="number of training epochs")
    parser.add_argument("--beam_size", type=int, default=3,
                        help="beam size")
    parser.add_argument("--data_name", type=str, default="caps3_freq5_maxlen100", 
                        help="Dataset file name according to setting of create_input_files.py")
    parser.add_argument("--wordmap", type=str, default="word_map_freq5",
                        help="word map file name according to setting of create_input_files.py")
    parser.add_argument("--max_caption_len", type=int, default=100, 
                        help="max generated caption length")
    
    args = parser.parse_args()
    return args

def get_dataset(args):
    return CaptionDataset(data_name=args.data_name, split='test', transform=caption_transforms)

def get_dataloader(args, dataset):
    testloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return testloader

def get_wordmap(args):
    word_map = read_json_file(os.path.join(DATA_DIR, "annotations", f"{args.wordmap}.json"))
    id_map = {v: k for k, v in word_map.items()}
    return word_map, id_map

def get_model(vocab_size, args):
    checkpoint = torch.load(args.checkpoint)
    best_bleu4 = checkpoint["bleu4"]
    model_state_dict = checkpoint["model"]

    model = LstmTranslator(
        vocab_size=vocab_size,
    )
    model.load_state_dict(model_state_dict)
    print(f"load model from {args.checkpoint} with bleu4 val {best_bleu4:.6f}")
    return model

class Evaluator():
    def __init__(
        self,
        testloader,
        model,
        word_map,
        id_map,
        args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.testloader = testloader
        self.model = model.to(self.device)
        self.word_map = word_map
        self.id_map = id_map
        self.beam_size = self.args.beam_size
        self.bleu4_fn = BLEUScore(n_gram=4).to(self.device)

    def decode_caption(self, enc_cap):
        tokens = []
        for t in enc_cap:
            token = self.id_map[t]
            if token not in ('<start>', '<pad>', '<end>'):
                if token != '_':
                    tokens.append(token.replace('▁', ' '))
                else:
                    tokens.append(token)
        return ''.join(tokens).replace(" ", "", 1)

    def run(self):
        vocab_size = len(self.word_map)
        self.model.eval()
        eval_results = []

        with torch.no_grad():
            for imgs, caps, caplens, allcaps, img_paths in tqdm(self.testloader, ncols=100, desc=f"evaluating..."):
                references = list()
                hypotheses = list()
                k = self.beam_size
                imgs = imgs.to(self.device)
                encoder_out = self.model.encoder(imgs)
                encoder_dim = encoder_out.size(-1)
                encoder_out = encoder_out.view(1, -1, encoder_dim)
                num_pixels = encoder_out.size(1)

                encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
                k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * k).to(self.device)  # (k, 1)
                seqs = k_prev_words  # (k, 1)
                top_k_scores = torch.zeros(k, 1).to(self.device)

                complete_seqs = list()
                complete_seqs_scores = list()

                step = 1
                h, c = self.model.decoder.init_hidden_state(encoder_out)

                while True:
                    embeddings = self.model.decoder.embedding(k_prev_words).squeeze(1)
                    attention_weighted_encoding, _ = self.model.decoder.attention(encoder_out, h)  
                    gate = self.model.decoder.sigmoid(self.model.decoder.f_beta(h))
                    attention_weighted_encoding = gate * attention_weighted_encoding
                    h, c = self.model.decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))  # (k, decoder_dim)
                    preds = self.model.decoder.fc(h)  # (k, vocab_size)
                    preds = torch.nn.functional.log_softmax(preds, dim=1)

                    # Add
                    preds = top_k_scores.expand_as(preds) + preds  # (k, vocab_size)

                    # For the first step, all k points will have the same scores (since same k previous words, h, c)
                    if step == 1:
                        top_k_scores, top_k_words = preds[0].topk(k, dim=0, largest=True, sorted=True)  # (k)
                    else:
                        # Unroll and find top scores, and their unrolled indices
                        top_k_scores, top_k_words = preds.view(-1).topk(k, dim=0, largest=True, sorted=True)  # (k)

                    # Convert unrolled indices to actual indices of scores
                    prev_word_inds = top_k_words // vocab_size  # (k)
                    next_word_inds = top_k_words % vocab_size  # (k)

                    # Add new words to sequences
                    seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                    # Which sequences are incomplete (didn't reach <end>)?
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                    next_word != self.word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)  # reduce beam length accordingly

                    # Proceed with incomplete sequences
                    if k == 0:
                        break
                    seqs = seqs[incomplete_inds]
                    h = h[prev_word_inds[incomplete_inds]]
                    c = c[prev_word_inds[incomplete_inds]]
                    encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                    # Break if things have been going on too long
                    if step > self.args.max_caption_len:
                        break
                    step += 1
                
                if complete_seqs != []:
                    i = complete_seqs_scores.index(max(complete_seqs_scores))
                    seq = complete_seqs[i]
                else:
                    i = torch.randint(0, seqs.size(0), (1,)).item()
                    seq = seqs[i].tolist()

                # References
                img_caps = allcaps[0].tolist()
                img_captions = [self.decode_caption(c) for c in img_caps]
                references.append(img_captions)

                # Hypotheses
                hypotheses.append(self.decode_caption(seq))

                assert len(references) == len(hypotheses)
                self.bleu4_fn.update(hypotheses, references)

            
                eval_results.append({
                    "img_path": img_paths[0],
                    "hypotheses": hypotheses[0],
                    "references": references[0],
                })
        
        bleu4 = self.bleu4_fn.compute()

        print(f"BLEU-4 score with beam k = {self.beam_size}: {bleu4:.6f}")
        eval_data =  {"beam_size": self.args.beam_size, "bleu4" : bleu4.cpu().item(), "captions": eval_results}
        with open(f"results/eval_{self.args.beam_size}.json", "w", encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=4)


def main(args):
    testset = get_dataset(args)
    testloader = get_dataloader(args, testset)
    word_map, id_map = get_wordmap(args)
    model = get_model(len(word_map), args)
    evaluator = Evaluator(
        testloader=testloader,
        model=model,
        word_map=word_map,
        id_map=id_map,
        args=args)
    evaluator.run()

if __name__ == "__main__":
    init_seed(0)
    args = parser_args()
    main(args)
    