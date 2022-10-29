import numpy as np

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""


def viterbi_1(train, test, alpha=1e-5):
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
    ps = np.array([0 for _ in range(len(tags))], dtype=float)
    for sentence in train:
        for word, tag in sentence:
            ps[tag_to_index[tag]] += 1
    v = np.sum(ps > 0)
    n = np.sum(ps)
    ps[ps != 0] = (ps[ps != 0] + alpha) / (n + alpha * (v + 1))
    ps[ps == 0] = alpha / (n + alpha * (v + 1)) / (len(tags) - v)
    # words = set()
    # for sentence in train:
    #     words = words.union(set([word for word, _ in sentence]))
    # words = list(words)

    pt, pe = {}, {}
    for sentence in train:
        for i, ((word_1, tag_1), (word_2, tag_2)) in enumerate(zip(sentence[:-1], sentence[1:])):
            if tag_1 not in pt:
                pt[tag_1] = {}
            if tag_2 not in pt[tag_1]:
                pt[tag_1][tag_2] = 0
            pt[tag_1][tag_2] += 1

            if i == 0:
                if tag_1 not in pe:
                    pe[tag_1] = {}
                if word_1 not in pe[tag_1]:
                    pe[tag_1][word_1] = 0
                pe[tag_1][word_1] += 1

            if tag_2 not in pe:
                pe[tag_2] = {}
            if word_2 not in pe[tag_2]:
                pe[tag_2][word_2] = 0
            pe[tag_2][word_2] += 1

    for tag_1 in pt:
        v = len(pt[tag_1])
        n = sum(pt[tag_1].values())
        pt[tag_1][None] = alpha / (n + alpha * (v + 1))
        for tag_2 in pt[tag_1]:
            pt[tag_1][tag_2] = (pt[tag_1][tag_2] + alpha) / (n + alpha * (v + 1))
    pt[None] = {tag: 1 / len(tags) for tag in tags}

    for tag in pe:
        v = len(pe[tag])
        n = sum(pe[tag].values())
        pe[tag][None] = alpha / (n + alpha * (v + 1))
        for word in pe[tag]:
            pe[tag][word] = (pe[tag][word] + alpha) / (n + alpha * (v + 1))

    output = []
    for sentence in test:
        v, b = np.full((len(sentence), len(tags)), 0, dtype=float), np.full((len(sentence), len(tags)), 0, dtype=int)
        for i, tag in enumerate(tags):
            v[0, i] = np.log(ps[i]) + np.log(pe[tag][sentence[0] if sentence[0] in pe[tag] else None])
        for i, word in enumerate(sentence[1:], start=1):
            for i_tag_2, tag_2 in enumerate(tags):
                vs = np.array([
                    v[i - 1, i_tag_1] +
                    np.log(pt[tag_1 if tag_1 in pt else None]
                           [tag_2 if tag_2 in pt[tag_1 if tag_1 in pt else None] else None]) +
                    np.log(pe[tag_2][word if word in pe[tag_2] else None])
                    for i_tag_1, tag_1 in enumerate(tags)
                ])
                b[i, i_tag_2] = np.argmax(vs)
                v[i, i_tag_2] = vs[b[i, i_tag_2]]

        # print(v[-1, :])
        indices = [np.argmax(v[-1, :])]
        for i in range(len(sentence) - 1, 0, -1):
            indices.append(b[i, indices[-1]])
        output.append([(word, tags[i]) for word, i in zip(sentence, indices[::-1])])
        # for tup in output[-1]:
        #     print(tup)
        # print()

    return output
