import numpy as np
from sklearn.metrics import dcg_score
from sklearn.metrics import ndcg_score


true_relevance = np.asarray([[5, 4, 3, 2, 2]])
scores = np.asarray([[5, 3, 2, 1, 2]])

print(dcg_score(true_relevance, scores))
print(ndcg_score(true_relevance, scores))
