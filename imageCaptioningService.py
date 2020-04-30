import json

from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.handlers import ImageHandler

import models

# Global values
transform = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
   ])
beam_size = 3
word_map_file = 'word_map.json'
word_map = json.load(open(word_map_file, 'r'))
vocab_size = len(word_map)
rev_word_map = {v:key for key, v in word_map.items()}

@bentoml.env(pip_dependencies=['torch', 'torchvision'])
@bentoml.artifacts([PytorchModelArtifact('encoder'), PytorchModelArtifact('decoder')])
class ImageCaptioner(bentoml.BentoService):

    @bentoml.api(ImageHandler)
    def gen_caption(self, img):
        k = beam_size

        # Get the image
        img = Image.fromarray(img)
        img = transform(img)
        # (3, 224, 224)
        img = img.unsqueeze(0)

        # Pass to encoder
        self.artifacts.encoder.eval()
        # (1, enc_img_size, enc_img_size, encoder_dim)
        encoder_out = self.artifacts.encoder(img)
        enc_img_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # (1, enc_img_size*enc_img_size, encoder_dim)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # we are treating the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        # Tensor to store top k previous word at each step
        # (k, 1)
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k)

        # Tensor to store top k sequences; now they're just <start>
        # (k, 1)
        seqs = k_prev_words

        # Tensor to keep track of the scores of each seq
        top_k_scores = torch.zeros(k, 1)

        # list to store completed seqs and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        decoder = self.artifacts.decoder.eval()
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed
        # from this process once they hit <end>
        while True:

            # (s, embed_dim)
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            # (s, encoder_dim) (s, num_pixels)
            awe, _ = decoder.attention(encoder_out, h)


            # Gating scalar
            # (s, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate*awe

            # (s, decoder_dim)
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h,c))

            # (s, vocab_size)
            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                # if first step then all the words are same
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                # else unroll and get the top k scores and word indexes
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            # convert the unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size
            next_word_inds = top_k_words % vocab_size

            # add words to sequence
            # (s, step+1)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # the seq that did not reach <end>
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # break if running for too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        print(seq)
        words = [rev_word_map[ind] for ind in seq]
        return ' '.join(words)
