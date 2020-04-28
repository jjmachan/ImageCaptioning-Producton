import argparse
import json
import shutil

import torch

from imageCaptioningService import ImageCaptioner
from models import Encoder, DecoderWithAttention

def saveToBento(checkpoint_path, word_map_file):

    # Define model
    # IMP: make sure the hyperparameters for the model is the same as
    # that in training.
    emb_dim = 512
    attention_dim = 512
    decoder_dim = 512
    dropout = 0.5
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    vocab_size = len(word_map)

    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=vocab_size,
                                   dropout=dropout,)
    encoder = Encoder()

    # Initialize model
    checkpoint = torch.load(checkpoint_path)
    decoder_state = checkpoint['decoder_state']
    encoder_state = checkpoint['encoder_state']
    decoder.load_state_dict(decoder_state)
    encoder.load_state_dict(encoder_state)

    # Add model to BentoML
    bento_svc = ImageCaptioner()
    bento_svc.pack('encoder', encoder)
    bento_svc.pack('decoder', decoder)

    # save bento service
    saved_path = bento_svc.save()

    # save word_map_file to bento container
    dest_file = saved_path + '/ImageCaptioner/artifacts/word_map.json'
    shutil.copy(word_map_file, dest_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint to load the file from')
    parser.add_argument('word_map', help='Path of the word map json file created')
    args = parser.parse_args()

    saveToBento(args.checkpoint, args.word_map)
