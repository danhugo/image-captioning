import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import BLEUScore
import numpy as np
import wandb
from argparse import ArgumentParser
from datetime import datetime
import os

from config import DATA_DIR
from dataset import CaptionDataset, caption_transforms
from utils import read_json_file, init_seed
from rnn import LstmTranslator

torch.backends.cudnn.deterministic = True

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for dataloader")
    parser.add_argument("--checkpoint", type=str, default='exp/caps3_freq5_maxlen100_20240627_154825.pt',
                        help="path to checkpoint")
    parser.add_argument("--epochs", type=int, default=120,
                        help="number of training epochs")
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--data_name", type=str, default="caps3_freq5_maxlen100", 
                        help="Dataset file name according to setting of create_input_files.py")
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--wordmap", type=str, default="word_map_freq5",
                        help="word map file name according to setting of create_input_files.py")
    
    parser.add_argument("--wandb_name", type=str, default='hicehehe',
                        help="Wandb account name")
    
    parser.add_argument("--encoder_finetune", action="store_true", help="Flag it to enable finetuning encoder")
    
    return parser.parse_args()

def get_dataset(args):
    trainset = CaptionDataset(data_name=args.data_name, split='train', transform=caption_transforms)
    valset = CaptionDataset(data_name=args.data_name, split='val', transform=caption_transforms)
    return trainset, valset

def get_wordmap(args):
    word_map = read_json_file(os.path.join(DATA_DIR, "annotations", f"{args.wordmap}.json"))
    id_map = {v: k for k, v in word_map.items()}
    return word_map, id_map
 
def get_dataloader(trainset, valset, args):
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return trainloader, valloader

def get_model(vocab_size, args):
    model = LstmTranslator(
        vocab_size=vocab_size,
        encoder_finetune=args.encoder_finetune
    )
    return model

class TrainerSingle():
    def __init__(
        self,
        trainloader,
        valloader,
        model,
        word_map,
        id_map,
        args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        # self.device = torch.device("cpu")
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.model.to(self.device)
        self.word_map = word_map
        self.id_map = id_map
        self.epochs = args.epochs
        self.alpha_c = 1.0
        self.start_epoch = 0

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": args.encoder_lr},
                {"params": self.model.decoder.parameters(), "lr": args.decoder_lr},
            ],)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.8, patience=10, verbose=True)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.train_accuracy_fn = MulticlassAccuracy(num_classes=self.model.vocab_size, top_k=5).to(self.device)
        self.val_accuracy_fn = MulticlassAccuracy(num_classes=self.model.vocab_size, top_k=5).to(self.device)
        self.bleu4_fn = BLEUScore(n_gram=4).to(self.device)
        self.average_train_loss = MeanMetric().to(self.device)
        self.average_val_loss = MeanMetric().to(self.device)

        if args.checkpoint is not None:
            self.load_checkpoint(args.checkpoint)
        
    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_bleu4 = checkpoint["bleu4"]
        model_state_dict = checkpoint["model"]
        optimizer_state_dict = checkpoint["optimizer"]
        scheduler_state_dict = checkpoint["scheduler"]
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.optimizer.param_groups[0]['lr'] = self.args.encoder_lr
        self.optimizer.param_groups[1]['lr'] = self.args.decoder_lr
        self.scheduler.load_state_dict(scheduler_state_dict)

    def save_checkpoint(self, epoch):
        save_dict = {
            "epoch": epoch,
            "bleu4": self.best_bleu4,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        torch.save(save_dict, f"exp/{self.args.exp}.pt")

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
    
    @staticmethod
    def _visuallization(tensor_imgs, pred_captions, tgt_captions, num=3):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        imgs = ((tensor_imgs * std  + mean).permute(0,2,3,1) * 255.).numpy().astype(np.uint8)
        imgs = imgs[:num]
        pred_captions = pred_captions[:num]
        tgt_captions = tgt_captions[:num]
        for i, (img, pred_c, tgt_c) in enumerate(zip(imgs, pred_captions, tgt_captions)):
            wandb.log({f"Image {i}": wandb.Image(img, caption=f"pred: {pred_c} \n tgt: {tgt_c}")})
            

    def train(self, epoch):
        self.model.train()

        for imgs, caps, caplens in tqdm(self.trainloader, ncols=100, desc=f"train epoch {epoch+1}/{self.args.epochs}"):
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            preds, caps_sorted, decode_lengths, alphas, sort_ind = self.model(imgs, caps, caplens)
            targets = caps_sorted[:, 1:] # get all tokens after <start>

            preds = torch.nn.utils.rnn.pack_padded_sequence(preds, decode_lengths, batch_first=True)[0]
            targets = torch.nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = self.criterion(preds, targets)
            regularize_loss = self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean() # doubly stochastic attention regularization
            loss += regularize_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step(self.best_bleu4)
            self.average_train_loss.update(loss)
            self.train_accuracy_fn.update(preds, targets)
            # break
        
        epoch_loss = self.average_train_loss.compute()
        acc = self.train_accuracy_fn.compute()
        self.average_train_loss.reset()
        self.train_accuracy_fn.reset()
        print("loss {:.4f} - acc5 {:.6f}".format(epoch_loss, acc))
        wandb.log({"train_loss": loss, "train_acc5": acc})
        wandb.log({"encoder_lr": self.optimizer.param_groups[0]['lr'], "decoder_lr": self.optimizer.param_groups[1]['lr']})

    def val(self, epoch):
        self.model.eval()
       
        with torch.no_grad():
            for imgs, caps, caplens, allcaps, _ in tqdm(self.valloader, ncols=100, desc=f"val epoch: {epoch+1}/{self.args.epochs}"):
                references = list()
                hypotheses = list()
                imgs = imgs.to(self.device)
                caps = caps.to(self.device)
                caplens = caplens.to(self.device)

                preds, caps_sorted, decode_lengths, alphas, sort_ind = self.model(imgs, caps, caplens)
                targets = caps_sorted[:, 1:] # get all tokens after <start>
                preds_copy = preds.clone()
                preds = torch.nn.utils.rnn.pack_padded_sequence(preds, decode_lengths, batch_first=True)[0]
                targets = torch.nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                loss = self.criterion(preds, targets)
                regularize_loss = self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean() # doubly stochastic attention regularization
                loss += regularize_loss
                self.average_val_loss.update(loss)
                self.val_accuracy_fn.update(preds, targets)

                allcaps = allcaps[sort_ind.cpu()]
                for img_caps in allcaps:
                    img_caps = img_caps.tolist()
                    de_caps = [self.decode_caption(c) for c in img_caps]
                    references.append(de_caps)
                
                pred_caps = torch.max(preds_copy, dim=2)[1].tolist()
                for pred_cap in pred_caps:
                    de_cap = self.decode_caption(pred_cap)
                    hypotheses.append(de_cap)
                
                self.bleu4_fn.update(hypotheses, references)
                # break
            self._visuallization(imgs[sort_ind].cpu(), hypotheses, references, num=10)
            bleu4_score = self.bleu4_fn.compute()
            loss = self.average_val_loss.compute()
            acc = self.val_accuracy_fn.compute()
            print("loss {:.4f} - acc5 {:.6f} - bleu4: {:.6f} - best bleu4: {:6f}".format(loss, acc, bleu4_score, self.best_bleu4))
            wandb.log({"val_loss": loss, "val_acc5": acc, "bleu4": bleu4_score})
            self.bleu4_fn.reset()
            self.average_val_loss.reset()
            self.val_accuracy_fn.reset()
            return bleu4_score

    def run(self):
        self.best_bleu4 = 0.0
        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            print("-" * 100)
            self.train(epoch)
            blue4_score = self.val(epoch)

            if blue4_score > self.best_bleu4:
                self.best_bleu4 = blue4_score
                self.save_checkpoint(epoch)


def main(args):
    args.exp = f"{args.data_name}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        mode="online", # online, disabled
        project="image-caption",
        entity= args.wandb_name, # change with your wandb account
        name=args.exp
    )
    trainset, valset = get_dataset(args)
    trainloader, valloader = get_dataloader(trainset, valset, args)
    word_map, id_map = get_wordmap(args)
    model = get_model(len(word_map), args)
    trainer = TrainerSingle(
        trainloader=trainloader,
        valloader=valloader,
        model=model,
        word_map=word_map,
        id_map=id_map,
        args=args
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    init_seed(0)
    args = parser_args()
    main(args)