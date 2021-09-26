###Import Modules
import tensorflow as tf
import numpy as np
import math
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer

# Download the text data
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip', 
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

#create function to clean up data to remove punctuation and spaces
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

#create tokenisers to assign vector to each word
tokeniser_eng = tf.keras.preprocessing.text.Tokenizer()
tokeniser_fra = tf.keras.preprocessing.text.Tokenizer()

#CREATE THE NEURAL NETWORK

encoder_input = Input(shape=(None, ), name='encoder_input')
embedding_size = 128
num_words = 1000
encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')

state_size = 512

encoder_gru1 = GRU(state_size, name='encoder_gru1',
                   return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',
                   return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',
                   return_sequences=False)

def connect_encoder():
    # Start the neural network with its input-layer.
    net = encoder_input
    
    # Connect the embedding-layer.
    net = encoder_embedding(net)

    # Connect all the GRU-layers.
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)

    # This is the output of the encoder.
    encoder_output = net
    
    return encoder_output

encoder_output = connect_encoder()

decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')

decoder_input = Input(shape=(None, ), name='decoder_input')

decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)

decoder_dense = Dense(num_words,
                      activation='softmax',
                      name='decoder_output')

def connect_decoder(initial_state):
    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])

model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])

decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])

model_train.compile(optimizer=RMSprop(lr=1e-3),
                    loss='sparse_categorical_crossentropy')

