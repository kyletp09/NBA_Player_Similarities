import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# Import Cleaned Dataset
df = pd.read_csv('Datasets/career_avgs.csv')
df.head()

# Scaling the Features using standardization
scaler = StandardScaler().fit(df.loc[:, "REB": "FT_PCT"])
np_standardized = scaler.fit_transform(df.loc[:, "REB": "FT_PCT"])
np_standardized_df = pd.DataFrame(data= np_standardized, columns= df.loc[:, "REB": "FT_PCT"].columns).fillna(0)
np_standardized_df.head()

# Performing PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(np_standardized_df)
principal_df = pd.DataFrame(data= principal_components, columns= ['Principal Component 1', 'Principal Component 2'])

# K Means Clustering
kmeans = KMeans(n_clusters = 5, init= 'k-means++', random_state= 42332)
kmeans.fit_transform(principal_df)

# Naming the Clusters
clustered_df = df.copy()
clustered_df['Clusters'] = kmeans.labels_
cluster_meaning = { 0 : "Scoring Shot Creator", 1 : "3 and D Wing", 2 : "Interior Role Player", 3 : "Energetic Role Player", 4 : "Defensive Anchor Big" } 
clustered_df['Clusters'] = clustered_df['Clusters'].replace(cluster_meaning)
clustered_df[['Principal Component 1', 'Principal Component 2']] = principal_df
clustered_df.head()

# Streamlit UI
st.markdown('# NBA Comparison Project')
sns.set_style('darkgrid')
sns.set_palette('Paired')
sns.relplot(x= 'Principal Component 1', y= 'Principal Component 2',  data= principal_df, hue = clustered_df['Clusters'])
plt.title('K Means Clustering of Current NBA Players')
st.pyplot(plt.gcf())

cluster_option = st.selectbox(
    'What players of what archetypes would you like to see?',
     clustered_df['Clusters'].unique())

st.dataframe(clustered_df[clustered_df['Clusters'] == cluster_option])

player_comparison_option = st.selectbox(
    'What player would you want to see comparisons to?',
    clustered_df['PLAYER_NAME']
)

# K Nearest Neighbors
KNN = NearestNeighbors(n_neighbors = 6, metric='euclidean')
KNN.fit(clustered_df[['Principal Component 1', 'Principal Component 2']])
player_coords = clustered_df.loc[clustered_df['PLAYER_NAME'] == player_comparison_option][['Principal Component 1', 'Principal Component 2']]
distances, indices = KNN.kneighbors(player_coords)
selected_df = clustered_df.iloc[indices[0]].reset_index().drop(0)

st.dataframe(selected_df)
