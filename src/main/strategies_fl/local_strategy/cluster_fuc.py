import datetime

from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
)
from sklearn.metrics import silhouette_score


def calculate_proto(data):
    pass


def apply_clustering(data, method_name="KMeans", n_clusters=10):
    if method_name == "KMeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=2024, batch_size=100
        )
    elif method_name == "AffinityPropagation":
        model = AffinityPropagation(random_state=2024, damping=0.9)
    elif method_name == "MeanShift":
        model = MeanShift()
    elif method_name == "DBSCAN":
        model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Clustering method {method_name} not supported!")
    labels = model.fit_predict(data)
    return labels


def do_silhouette_score(k_range, data_samples):
    scores = []
    deltas = []

    start_k = 10

    now = datetime.datetime.now()

    for k in range(start_k, k_range):
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"this k={k} is calculating!")
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data_samples)

        if len(set(model.labels_)) > 1:
            score = silhouette_score(data_samples, model.labels_)
            print(f"Print the score = {score}")
            scores.append(score)

            if len(scores) > 1:
                delta = score - scores[-2]
                deltas.append(delta)
            else:
                deltas.append(0)
        # else:
        #     scores.append(float('-inf'))
        #     deltas.append(float('-inf'))  # Nếu không có clustering tốt, delta sẽ là -inf

    max_delta = max(deltas)
    best_k = deltas.index(max_delta) + start_k  # +2

    print(f"Print the max delta = {max_delta}")
    print(f"Print the index of max delta = {deltas.index(max_delta)}")
    print(f"the optimal_cluster is {best_k}")

    return best_k
