import numpy as np
import pandas as pd
import xarray as xr
import sklearn.preprocessing as skpp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('darkgrid')
import cartopy.crs as ccrs
import sklearn.cluster as skcls
import sklearn.metrics as skmtr
import scipy.cluster.hierarchy as scihy
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.neighbors import NearestCentroid

# ERA5 Precipitation
ERA  = xr.open_dataset('precip_monthly_ID.nc')
ERA['tp'] = ERA['tp']*1000
ERA  = ERA.assign_coords(longitude=(((ERA.longitude + 180) % 360) - 180)).sortby('longitude')
ERA  = ERA.rename({'latitude':'lat', 'longitude':'lon'})

# Average time dimension
ERAmean = ERA.tp.groupby('time.month').mean(dim='time')

# Merge into Dataset, stack lat,lon --> latlon
ds = ERAmean
ds = ds.stack(latlon=('lat','lon'))

# Convert into Pandas DataFrame
df = ds.to_pandas()

input = df.copy(True).T
scaler = skpp.PowerTransformer().fit(input)

input_scaled = pd.DataFrame(scaler.transform(input),
                            index=input.index, columns=input.columns
                            )

def plot_dendrogram(model, **kwargs):
    from scipy.cluster.hierarchy import dendrogram
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

h_clust = skcls.AgglomerativeClustering(n_clusters=3,
                                        distance_threshold=None,
                                        metric='euclidean',
                                        linkage='ward',
                                        compute_distances=True
                                        )

yp = h_clust.fit(input_scaled)
h_clust2 = skcls.AgglomerativeClustering(n_clusters=3,
                                        distance_threshold=None,
                                        metric='manhattan',
                                        linkage='complete',
                                        compute_distances=True
                                        )

yp2 = h_clust2.fit(input_scaled)

h_clust2 = skcls.AgglomerativeClustering(n_clusters=3,
                                        distance_threshold=None,
                                        metric='manhattan',
                                        linkage='complete',
                                        compute_distances=True
                                        )

plot_dendrogram(h_clust,
                truncate_mode='lastp', p=20,#h_clust.n_clusters_,
                orientation='right',
                )

try: plt.axvline(h_clust.distance_threshold, ls='--', c='grey')
except: plt.axvline(h_clust.distances_[-(h_clust.n_clusters_)], ls='--', c='grey')
finally: plt.show()


#### Distance plot
plt.plot(np.arange(len(h_clust.distances_))+1, h_clust.distances_[::-1], '-o', c='tab:orange')

try: plt.axhline(h_clust.distance_threshold, ls='--', c='grey')
except: plt.axhline(h_clust.distances_[-(h_clust.n_clusters_)], ls='--', c='grey')

plt.title('Agglomerative')
plt.xlabel('remaining cluster'); plt.xlim([0,22])
plt.ylabel('distances')
plt.savefig('Dendogram Euclid.png')

plot_dendrogram(h_clust2,
                truncate_mode='lastp', p=20,#h_clust.n_clusters_,
                orientation='right',
                )

try: plt.axvline(h_clust2.distance_threshold, ls='--', c='grey')
except: plt.axvline(h_clust2.distances_[-(h_clust2.n_clusters_)], ls='--', c='grey')
finally: plt.show()


#### Distance plot
plt.plot(np.arange(len(h_clust2.distances_))+1, h_clust2.distances_[::-1], '-o', c='tab:orange')

try: plt.axhline(h_clust2.distance_threshold, ls='--', c='grey')
except: plt.axhline(h_clust2.distances_[-(h_clust2.n_clusters_)], ls='--', c='grey')

plt.title('Agglomerative')
plt.xlabel('remaining cluster'); plt.xlim([0,22])
plt.ylabel('distances')
plt.savefig('Dendogram Manhattan.png')

clf = NearestCentroid()
clf.fit(input_scaled, yp2.labels_)

k_clust = skcls.KMeans(n_clusters=3,
                       init='k-means++',
                       )
k_clust.fit(input_scaled)
k_clust2 = skcls.KMeans(n_clusters=3,
                       init=clf.centroids_,
                       )
k_clust2.fit(input_scaled)

dflabel = pd.DataFrame([h_clust.labels_, k_clust.labels_],
                       index=['agglomerative', 'kmeans'],
                       columns=input_scaled.index,
                       ).T

# Convert DataFrame to Dataset
dslabel = dflabel.to_xarray()
dflabel2 = pd.DataFrame([h_clust2.labels_, k_clust2.labels_],
                       index=['agglomerative', 'kmeans'],
                       columns=input_scaled.index,
                       ).T

# Convert DataFrame to Dataset
dslabel2 = dflabel2.to_xarray()


for var in ['agglomerative', 'kmeans']:
  fig = plt.figure(figsize=(16,8))
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.coastlines()

  levels = np.arange(-0.5, 4.5, 1)
  dslabel[var]\
              .plot(ax=ax, levels=levels, cmap="Spectral",
                    cbar_kwargs={'ticks': range(5), 'shrink':0.5})

  plt.title(f'Hasil {var} Clustering 1 Curah Hujan di Indonesia', fontsize=14)
  plt.savefig(f'Hasil {var} Clustering 1 Curah Hujan di Indonesia.png')

  fig = plt.figure(figsize=(16,8))
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.coastlines()
  dslabel2[var]\
              .plot(ax=ax, levels=levels, cmap="Spectral",
                    cbar_kwargs={'ticks': range(5), 'shrink':0.5})

  plt.title(f'Hasil {var} Clustering 2 Curah Hujan di Indonesia', fontsize=14)
  plt.savefig(f'Hasil {var} Clustering 2 Curah Hujan di Indonesia.png')

sample = input_scaled.stack().to_xarray()
sample.sel(lat=-7.303091861192951, lon=109.60929300085719, method='nearest').plot()
plt.savefig(f'Sampel Clust 1.png')

sample.sel(lat=0.37702210553651144, lon=113.55376281634466, method='nearest').plot()
plt.savefig(f'Sampel Clust 2.png')

sample.sel(lat=-0.10294741209579893, lon=127.82308225598152, method='nearest').plot()
plt.savefig(f'Sampel Clust 3.png')