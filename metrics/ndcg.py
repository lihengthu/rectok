import numpy as np


def dcg_score(y_true, y_score, k=10):
    index = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, index[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    idcg = dcg_score(y_true, y_true, k)
    dcg = dcg_score(y_true, y_score, k)
    return dcg / idcg


def dcg_from_ranking(y_true, ranking):
    y_true = np.asarray(y_true)
    ranking = np.asarray(ranking)
    rel = y_true[ranking]
    gains = 2 ** rel - 1
    discounts = np.log2(np.arange(len(ranking)) + 2)
    return np.sum(gains / discounts)


def ndcg_from_ranking(y_true, ranking):
    k = len(ranking)
    best_ranking = np.argsort(y_true)[::-1]
    dcg = dcg_from_ranking(y_true, ranking)
    idcg = dcg_from_ranking(y_true, best_ranking[:k])
    return dcg / idcg


if __name__ == '__main__':
    l1 = [5, 3, 2, 1, 2]
    l2 = [5, 4, 3, 2, 2]

    print(dcg_score(l2, l1))
    print(ndcg_score(l2, l1))

    assert dcg_score([5, 3, 2], [2, 1, 0]) > dcg_score([4, 3, 2], [2, 1, 0])
    assert ndcg_score([5, 3, 2], [2, 1, 0]) == 1.0
    assert dcg_score([5, 3, 2], [2, 1, 0]) == dcg_from_ranking([5, 3, 2], [0, 1, 2])
