import numpy as np

"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    """
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    tags = set()
    for sentence in train:
        tags = tags.union(set([tag for _, tag in sentence]))
    tags = list(tags)
    tag_to_index = {tag: i for i, tag in enumerate(tags)}
    tag_counts = [0 for _ in range(len(tags))]
    word_counts = {}
    for sentence in train:
        for word, tag in sentence:
            if word not in word_counts:
                word_counts[word] = [0 for _ in range(len(tags))]
            word_counts[word][tag_to_index[tag]] += 1
            tag_counts[tag_to_index[tag]] += 1
    default_tag = tags[np.argmax(tag_counts)]

    return [
        [
            ((word, tags[np.argmax(word_counts[word])]) if word in word_counts else (word, default_tag))
            for word in sentence
        ] for sentence in test
    ]
