import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
# from unicodedata import normalize
# import sentencepiece as spm
import torch
import trax
from trax import layers as tl
import trax.fastmath as ts


print(trax.fastmath.ops.backend_name())
from trax.supervised import decoding
import textwrap
from jax import config
# import jax
# print(jax.devices())
# config.update('jax_platform_name', 'gpu')


wrapper = textwrap.TextWrapper(width=70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

text_pairs = pd.read_csv('./lenta-ru-news.csv')
print(text_pairs.columns)
text_pairs = [(x, y) for x, y in zip(text_pairs.text, text_pairs.title)]
print(wrapper.fill(text_pairs[0][0]))
print(wrapper.fill(text_pairs[0][1]))

margin = int(len(text_pairs)*0.95)
train_text_pairs = text_pairs[:margin]
print('train cases: ', len(train_text_pairs))
eval_text_pairs = text_pairs[margin:]
print('eval cases: ', len(eval_text_pairs))

print(wrapper.fill(train_text_pairs[0][0]))


def data_generator(data, shuffle=True):
    '''
      Input:
        data - list containing tuples (article, summary)
        shuffle - If True: shuffle the data order
      Output:
        a tuple containing 2 elements:
        article
        summary
    '''

    data_lng = len(data)  # len(data)
    index_list = [*range(data_lng)]  # Create a list with the ordered indexes of sample data

    if shuffle:
        random.shuffle(index_list)  # re-shuffle the order

    index = 0  # Start with the first element
    while True:
        # Wrap the index each time that we reach the end of the list
        if index >= data_lng:
            index = 0
            if shuffle:
                random.shuffle(index_list)  # re-shuffle the order

        sample = data[index_list[index]]
        index += 1
        yield (sample)


# create data streams
def train_data_stream():
    return data_generator(train_text_pairs, shuffle=True)


def eval_data_stream():
    return data_generator(eval_text_pairs, shuffle=True)


PAD, EOS, UNK = 0, 1, 2


def detokenize(integers):
    s = trax.data.detokenize(
        integers,
        vocab_type='sentencepiece',
        vocab_file='bpe.model',
        vocab_dir='./1')  # loading pre-prepared model to save time
    return wrapper.fill(s)


def tokenize(s):
    inputs = next(trax.data.tokenize(
        iter([s]),
        vocab_type='sentencepiece',
        vocab_file='bpe.model',
        vocab_dir='./1'))

    return list(inputs) + [EOS]


vocab_size = trax.data.vocab_size(
    vocab_type='sentencepiece',
    vocab_file='bpe.model',
    vocab_dir='./1')

print('vocab size: ', vocab_size)


def preprocess(stream):
    for (article, summary) in stream:
        joint = np.array(list(article) + [EOS, PAD] + list(summary) + [EOS])
        mask = [0] * (len(list(article)) + 2) + [1] * (len(list(summary)) + 1)
        yield joint, joint, np.array(mask)

# You can combine a few data preprocessing steps into a pipeline like this.
input_pipeline = trax.data.Serial(
    # Tokenizes
    trax.data.Tokenize(vocab_type='sentencepiece',
                       vocab_dir='./1',
                       vocab_file='bpe.model'),
    # Uses function defined above
    preprocess,
    trax.data.FilterByLength(1024)
)

# Apply preprocessing to data streams.
train_stream = input_pipeline(train_data_stream())
eval_stream = input_pipeline(eval_data_stream())

train_input, train_target, train_mask = next(train_stream)
# assert sum((train_input - train_target)**2) == 0  # They are the same in Language Model (LM).
# check pad (id:0) and sep/eos (id:1)
print(train_input[-20:])

# batch of 8 sentences of length < 256 , 4 of length < 512....
boundaries =  [256, 512, 1024]
batch_sizes = [16, 8, 4]

# Create the streams.
train_batch_stream = trax.data.BucketByLength(
    boundaries, batch_sizes)(train_stream)

eval_batch_stream = trax.data.BucketByLength(
    boundaries, batch_sizes)(eval_stream)

input_batch, _, mask_batch = next(train_batch_stream)

# Shape of the input_batch
print(input_batch.shape)

##transformer


def PositionalEncoder(vocab_size, d_model, dropout, max_len, mode):
    """Returns a list of layers that:
    1. takes a block of text as input,
    2. embeds the words in that text, and
    3. adds positional encoding,
       i.e. associates a number in range(max_len) with
       each word in each sentence of embedded input text

    The input is a list of tokenized blocks of text

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train' or 'eval'.
    """
    # Embedding inputs and positional encoder
    return [
        # Add embedding layer of dimension (vocab_size, d_model)
        tl.Embedding(vocab_size, d_model),
        # Use dropout with rate and mode specified
        tl.Dropout(rate=dropout, mode=mode),
        # Add positional encoding layer with maximum input length and mode specified
        tl.PositionalEncoding(max_len=max_len, mode=mode)]


def FeedForward(d_model, d_ff, dropout, mode, ff_activation):
    """Returns a list of layers that implements a feed-forward block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """

    # Feed-forward block (list) with two dense layers with dropout and input normalized
    return [
        # Normalize layer inputs
        tl.LayerNorm(),
        # Add first feed forward (dense) layer
        tl.Dense(d_ff),
        # Add activation function passed in as a parameter
        ff_activation(),  # ReLU
        # Add dropout with rate and mode specified (don't use dropout during evaluation)
        tl.Dropout(rate=dropout, mode=mode),
        # Add second feed forward layer
        tl.Dense(d_model),
        # Add dropout with rate and mode specified
        tl.Dropout(rate=dropout, mode=mode)
    ]


def DecoderBlock(d_model, d_ff, n_heads,
                 dropout, mode, ff_activation):
    """Returns a list of layers that implements a Transformer decoder block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """

    # List of two Residual blocks: the attention with normalization and dropout and feed-forward blocks
    return [
        tl.Residual(
            # Normalize layer input
            tl.LayerNorm(),
            # Add causal attention
            tl.CausalAttention(d_model, n_heads=n_heads, dropout=dropout, mode=mode)
        ),
        tl.Residual(
            # Add feed-forward block
            # The feed-forward block takes care of normalization
            FeedForward(d_model, d_ff, dropout, mode, ff_activation)
        ),
    ]


def SumTransformer(vocab_size=vocab_size,
                   d_model=512,
                   d_ff=2048,
                   n_layers=6,
                   n_heads=8,
                   dropout=0.1,
                   max_len=4096,
                   mode='train',
                   ff_activation=tl.Relu):
    """Returns a Transformer language model.

    The input to the model is a tensor of tokens. (This model uses only the
    decoder part of the overall Transformer.)

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_layers (int): number of decoder layers.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train', 'eval' or 'predict', predict mode is for fast inference.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        trax.layers.combinators.Serial: A Transformer language model as a layer that maps from a tensor of tokens
        to activations over a vocab set.
    """

    # Stack of decoder blocks with n_layers with necessary parameters
    decoder_blocks = [
        DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation) for _ in range(n_layers)]

    # The complete model
    return tl.Serial(
        # Use teacher forcing (feed output of previous step to current step)
        tl.ShiftRight(mode=mode),
        # Add embedding inputs and positional encoder
        PositionalEncoder(vocab_size, d_model, dropout, max_len, mode),
        # Add decoder blocks
        decoder_blocks,
        # Normalize layer
        tl.LayerNorm(),

        # Add dense layer of vocab_size (since need to select a word to translate to)
        # (a.k.a., logits layer. Note: activation already set by ff_activation)
        tl.Dense(vocab_size),
        # Get probabilities with Logsoftmax
        tl.LogSoftmax()
    )


print(SumTransformer(n_layers=1))

from trax.supervised import training


def training_loop(SumTransformer, train_gen, eval_gen, output_dir="./model"):
    '''
    Input:
        SumTransformer (trax.layers.combinators.Serial): The transformer model.
        train_gen (generator): Training stream of data.
        eval_gen (generator): Evaluation stream of data.
        output_dir (str): folder to save your file.

    Returns:
        trax.supervised.training.Loop: Training loop.
    '''
    output_dir = os.path.expanduser(output_dir)  # trainer is an object

    # for initial train
    # lr_schedule = trax.lr.warmup(n_warmup_steps=4000, max_value=0.00015)
    # lr_schedule = trax.lr.warmup_and_rsqrt_decay(n_warmup_steps=8000, max_value=0.00015)

    # for re-train
    # lr_schedule = trax.supervised.lr_schedules.constant(0.0001)

    train_task = training.TrainTask(
        labeled_data=train_gen,  # The training generator
        loss_layer=tl.CrossEntropyLoss(),  # Loss function
        optimizer=trax.optimizers.Adam(0.00005),  # Optimizer
        # lr_schedule=lr_schedule,
        n_steps_per_checkpoint=100
    )

    eval_task = training.EvalTask(
        labeled_data=eval_gen,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )

    loop = training.Loop(SumTransformer(),
                         train_task,
                         eval_tasks=[eval_task],
                         output_dir=output_dir)

    return loop

# model = SumTransformer()
# model.to(device)
loop = training_loop(SumTransformer, train_batch_stream, eval_batch_stream)
loop.run(20000)