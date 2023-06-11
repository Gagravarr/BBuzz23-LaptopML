#!/usr/bin/python3

#   Licensed to the Apache Software Foundation (ASF) under one
#   or more contributor license agreements.  See the NOTICE file
#   distributed with this work for additional information
#   regarding copyright ownership.  The ASF licenses this file
#   to you under the Apache License, Version 2.0 (the
#   "License"); you may not use this file except in compliance
#   with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing,
#   software distributed under the License is distributed on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied.  See the License for the
#   specific language governing permissions and limitations
#   under the License.


# ----------------------------------------------------------------------------

# Setup step - load all our libraries
from mxnet import nd
from mxnet.contrib.text import embedding

import numpy as np

from random import randrange
import json

# Build a GloVe word embedding for our text
# This will take some time on the first run, as it downloads the
#  pre-trained embedding
# We use a smaller pre-trained model for speed, you may want to use 
#  the larger default by skipping the pretrained_file_name option
print("Loading GloVe embeddings")
glove = embedding.GloVe(pretrained_file_name='glove.6B.50d.txt')
print("GloVe loaded, contains %d terms" % len(glove))
print("")

# ----------------------------------------------------------------------------

# For finding cosine-similar embeddings
def find_nearest(vectors, wanted, num):
    # 1e-9 factor is to avoid zero/negative numbers
    cos = nd.dot(vectors, wanted.reshape((-1,))) / (
            (nd.sum(vectors * vectors, axis=1) + 1e-9).sqrt() *
            nd.sum(wanted * wanted).sqrt())
    top_n = nd.topk(cos, k=num, ret_typ='indices').asnumpy().astype('int32')
    return top_n, [cos[i].asscalar() for i in top_n]

# Looking up some similar words
def print_similar_tokens(query_token, num, embed):
    top_n, cos = find_nearest(embed.idx_to_vec,
                         embed.get_vecs_by_tokens([query_token]), num+1)
    print("Similar tokens to: %s" % query_token)
    for i, c in zip(top_n[1:], cos[1:]):  # Skip the word itself
        print(' - Cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))

# How "close" are two words, in costine terms of their embeddings?
def find_similarity_score(word_a, word_b, embed):
   vec_a, vec_b = embed.get_vecs_by_tokens([word_a, word_b])
   return (nd.dot(vec_a, vec_b) / (
            nd.sum(vec_a*vec_a).sqrt() * nd.sum(vec_b*vec_b).sqrt()
          )).asnumpy()[0] * 100

def print_similarity_score(word_a, word_b, embed):
   print("Difference between %s and %s is %d" % 
         (word_a, word_b, find_similarity_score(word_a, word_b, embed)))

# ----------------------------------------------------------------------------

# Test the embeddings
print_similar_tokens("linux", 3, glove)
print_similar_tokens("raise", 3, glove)
print("")

# Test the embedding similarity
print_similarity_score("raise", "risen", glove)
print_similarity_score("raise", "above", glove)
print_similarity_score("raise", "below", glove)
print_similarity_score("raise", "shine", glove)
print_similarity_score("raise", "linux", glove)
print("")

# ----------------------------------------------------------------------------

# Looking up word relationships, to verify the embeddings are working
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = find_nearest(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]  # Remove unknown words
def print_analogy(token_a, token_b, token_c, embed):
    anal = get_analogy(token_a, token_b, token_c, embed)
    print("The analogy of   %s -> %s   for   %s is %s" %
                                (token_a, token_b, token_c, anal))

print_analogy('berlin','germany','paris', glove)
print_analogy('madrid','spain','lisbon', glove)
print_analogy('man','boy','woman', glove)
print("")

print("GloVe can get it wrong...")
print_analogy('spain','madrid','portugal', glove)
print("GloVe can be sexist...")
print_analogy('doctor','man','nurse', glove)
