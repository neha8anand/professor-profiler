"""
Module containing model comparison code for KMeans clustering algorithm
with different number of clusters i.e. it helps to choose optimum number of clusters.
When run as a module, this will load a json dataset, pass the data and range of clusters to test
the algorithms on.
It returns plots showing the results of the elbow method and silhouette analysis using
the original data and the reduced data (to help with visualization of clusters).
"""

from cleaning import database_cleaner
from model import get_data, MyModel

from time import time

import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('ggplot')

from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale

np.random.seed(10)

def elbow_method(range_n_clusters, data):
    """
    Plots the within-cluster sum of squares or the inertia of a clustering model
    v/s the number of clusters
    """
    sses = [KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=10).fit(data).inertia_ for n_clusters in range_n_clusters]
    plt.plot(range_n_clusters,sses)
    plt.xlabel('Number of clusters(k) for kmeans++')
    plt.ylabel('Within-Cluster Sum of Squares/Inertia')
    plt.show()

def silhouette_analysis(range_n_clusters, data):
    """
    Plots the silhouette plot for different number of clusters
    and the 2-D visualization plot of clusters of TruncatedSVD
    version of the data.
    """
    silhouette_avg_scores = []
    ch_scores = [] # Calinski Harabaz score

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 0.5]
        ax1.set_xlim([-0.1, 0.5])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, data.shape[0] + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=10)
        cluster_labels = clusterer.fit_predict(data)

        # PCA using TruncatedSVD for visualization of clusters in 2-D space
        svd = TruncatedSVD(n_components=2, n_iter=7, random_state=10)
        reduced_data = svd.fit_transform(data)
        clusterer_reduced = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=10)

        clusterer_reduced.fit(reduced_data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        silhouette_avg_scores.append(silhouette_avg)

        # Calinski Harabaz score
        ch_score = calinski_harabaz_score(data.toarray(), cluster_labels)
        ch_scores.append(ch_score)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])

        # Visualization Code for reduced data
        #Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = clusterer_reduced.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        #plt.clf()
        ax2.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        ax2.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = clusterer_reduced.cluster_centers_
        ax2.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)


        ax2.set_title('K-means clustering on the abstracts dataset (SVD-reduced data)\n'
                  'Centroids are marked with white cross')
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_xticks(())
        ax2.set_yticks(())

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()

    fig, ax3 = plt.subplots()
    ax4 = ax3.twinx()

    ax3.plot(range_n_clusters, silhouette_avg_scores, color='green')
    ax4.plot(range_n_clusters, ch_scores)
    ax3.set_title('Choosing optimal number of clusters')
    ax3.set_xlabel('Number of clusters')
    ax3.legend(['Silhouette Score'])
    ax4.legend(['Calinski-Harabaz Index'])
    plt.show()

    return pd.DataFrame({'range_n_clusters': range_n_clusters,
                            'silhouette_scores': silhouette_avg_scores,
                            'calinski_harabaz_scores': ch_scores})

if __name__ == '__main__':
    range_n_clusters = range(5, 20)
    vectorizer, data = get_data('../data/pge_database.json')
    elbow_method(range_n_clusters, data)
    score_df = silhouette_analysis(range_n_clusters, data)
    print(score_df)
