# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from tqdm import tqdm
from collections import Counter
import reader
import numpy as np

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""


def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    stop_words = ['the', 'and', 'a', 'of', 'to']
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir, testdir, stemming, lowercase,
                                                                       silently)
    return [[word for word in review if word not in stop_words] for review in train_set], \
           train_labels, [[word for word in review if word not in stop_words] for review in dev_set], dev_labels


def print_paramter_vals(laplace, pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""


def naiveBayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.8, silently=False):
    print_paramter_vals(laplace, pos_prior)

    # training
    word_counts = {}
    word_totals = [0, 0]
    for review, label in zip(train_set, train_labels):
        word_totals[label] += len(review)
        for word in review:
            if word not in word_counts:
                word_counts[word] = [0, 0]
            word_counts[word][label] += 1
    n_types = len(word_counts)

    word_counts[None] = [
        laplace / (word_totals[0] + laplace * (n_types + 1)),
        laplace / (word_totals[1] + laplace * (n_types + 1))
    ]
    for word, (count_0, count_1) in word_counts.items():
        word_counts[word] = [
            (count_0 + laplace) / (word_totals[0] + laplace * (n_types + 1)),
            (count_1 + laplace) / (word_totals[1] + laplace * (n_types + 1))
        ]

    for pair in sorted([(count_0 + count_1, word) for word, (count_0, count_1) in word_counts.items()])[::-1][:30]:
        print(pair)

    # development
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        p_neg = np.log(1 - pos_prior) + np.sum([
            np.log(word_counts[word if word in word_counts else None][0]) for word in doc
        ])
        p_pos = np.log(pos_prior) + np.sum([
            np.log(word_counts[word if word in word_counts else None][1]) for word in doc
        ])
        yhats.append(int(p_pos > p_neg))
    return yhats


def print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.3,
                pos_prior=0.8, silently=False):
    print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    # unigram training
    word_counts = {}
    word_totals = [0, 0]
    for review, label in zip(train_set, train_labels):
        word_totals[label] += len(review)
        for word in review:
            if word not in word_counts:
                word_counts[word] = [0, 0]
            word_counts[word][label] += 1
    n_types = len(word_counts)

    word_counts[None] = [
        unigram_laplace / (word_totals[0] + unigram_laplace * (n_types + 1)),
        unigram_laplace / (word_totals[1] + unigram_laplace * (n_types + 1))
    ]
    for word, (count_0, count_1) in word_counts.items():
        word_counts[word] = [
            (count_0 + unigram_laplace) / (word_totals[0] + unigram_laplace * (n_types + 1)),
            (count_1 + unigram_laplace) / (word_totals[1] + unigram_laplace * (n_types + 1))
        ]

    # bigram training
    bigram_counts = {}
    bigram_totals = [0, 0]
    for review, label in zip(train_set, train_labels):
        if len(review) < 2:
            continue
        bigram_totals[label] += len(review) - 1
        for prev_word, next_word in zip(review[:-1], review[1:]):
            bigram = prev_word, next_word
            if bigram not in bigram_counts:
                bigram_counts[bigram] = [0, 0]
            bigram_counts[bigram][label] += 1
    n_types = len(bigram_counts)

    bigram_counts[None] = [
        bigram_laplace / (bigram_totals[0] + bigram_laplace * (n_types + 1)),
        bigram_laplace / (bigram_totals[1] + bigram_laplace * (n_types + 1))
    ]
    for bigram, (count_0, count_1) in bigram_counts.items():
        bigram_counts[bigram] = [
            (count_0 + bigram_laplace) / (bigram_totals[0] + bigram_laplace * (n_types + 1)),
            (count_1 + bigram_laplace) / (bigram_totals[1] + bigram_laplace * (n_types + 1))
        ]

    # development
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        p_neg_1 = np.log(1 - pos_prior) + np.sum([
            np.log(word_counts[word if word in word_counts else None][0]) for word in doc
        ])
        p_pos_1 = np.log(pos_prior) + np.sum([
            np.log(word_counts[word if word in word_counts else None][1]) for word in doc
        ])
        p_neg_2 = np.log(1 - pos_prior) + np.sum([
            np.log(bigram_counts[bigram if bigram in bigram_counts else None][0]) for bigram in zip(doc[:-1], doc[1:])
        ])
        p_pos_2 = np.log(pos_prior) + np.sum([
            np.log(bigram_counts[bigram if bigram in bigram_counts else None][1]) for bigram in zip(doc[:-1], doc[1:])
        ])
        yhats.append(int(
            p_pos_1 * (1 - bigram_lambda) + p_pos_2 * bigram_lambda >
            p_neg_1 * (1 - bigram_lambda) + p_neg_2 * bigram_lambda
        ))
    return yhats
