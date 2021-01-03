import argparse
import numpy as np
from sklearn import datasets

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

NSEEDS = 5

def main():
    args = parse_args()
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    num_classes = len(np.unique(y))




    if args.dim_reduce == "PCA":
        X = PCA(n_components=args.dim_size).fit_transform(iris.data)

    elif args.dim_reduce == "TSNE":
        X = TSNE(n_components=args.dim_size).fit_transform(iris.data)

    elif args.dim_reduce == "UMAP":
        X = umap.UMAP(n_components=args.dim_size).fit_transform(X)




    y_pred = None
    if args.clustering_algo == "KMeans":
        y_pred = KMeans(n_clusters=num_classes, random_state=20).fit_predict(X)

    if args.clustering_algo == "GMM":
        y_pred = GaussianMixture(n_components=num_classes, random_state=20).fit_predict(X)



    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, y_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(y, y_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(y, y_pred))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, y_pred, sample_size=1000))





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--clustering_algo", type=str, required=True, choices=["KMeans", "SPKMeans", "GMM", "KMedoids","Agglo","DBSCAN","Spectral","VMFM"])
    parser.add_argument('--dim_reduce', type=str)
    parser.add_argument('--dim_size', type=int)
    parser.add_argument('--num_clusters',  nargs='+', type=int, default=[20])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
