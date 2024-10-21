import torch
from torch import nn
import torchvision

class Encoder(nn.Module):
    """Encode batch of images into latent space. Return output with size 
    (batch_size, encoded_image_size, encoded_image_size, encoder_dim)"""

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.encoder_dim = resnet.fc.in_features

    def forward(self, images):
        out = self.resnet(images) 
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, encoder_dim)
        return out

    def fine_tune(self, fine_tune=True):
        """Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder."""
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """Attention Network compute attention of each hidden state over the encoder output (encoded image)"""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim) 
        self.decoder_att = nn.Linear(decoder_dim, attention_dim) 
        self.full_att = nn.Linear(attention_dim, 1)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """return: attention encodings, attention scores (alpha)"""
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """Decoder"""

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # LSTM Unit
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to initialize hidden state
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to initialize cell state
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings)."""
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    @staticmethod
    def sort_descending_caption_lengths(caption_lengths, encoder_out, encoded_captions):
        """Sort input data by decreasing lengths for effective batch computation"""
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        return caption_lengths, encoder_out, encoded_captions, sort_ind

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, encoder_out, encoded_captions, sort_ind = self.sort_descending_caption_lengths(caption_lengths, encoder_out, encoded_captions)
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # decoding lengths are actual lengths - 1 since we don't compute with <end> token.
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths]) # effective batch size for all sentence length > t
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  #  (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    

class LstmTranslator(nn.Module):
    def __init__(
            self,
            vocab_size,
            encoder_finetune=False,
            encoded_image_size=14,
            attention_dim=512,
            embed_dim=512,
            decoder_dim=512,
            dropout=0.5,
            ):
        super().__init__()
        self.encoder = Encoder(encoded_image_size)
        self.encoder.fine_tune(fine_tune=encoder_finetune)
        encoder_dim = self.encoder.encoder_dim
        self.vocab_size = vocab_size
        self.decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, self.vocab_size, encoder_dim, dropout)
    
    def forward(self, images, encoded_captions, caption_lengths):
        enc_imgs = self.encoder(images)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(enc_imgs, encoded_captions, caption_lengths)
        return scores, caps_sorted, decode_lengths, alphas, sort_ind
