import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

def plot_similarity_radar(df, target_player, stats, player_indices):
    
    # Convert stats to a consistent list of column names
    labels = list(stats)   # <--- FIXED
    
    # Include target player at the start
    players_to_plot = [target_player] + df['PLAYER_NAME'].iloc[player_indices].tolist()
    
    # Filter dataframe to only needed players
    df_plot = df[df['PLAYER_NAME'].isin(players_to_plot)].copy()

    # Normalize stats
    df_norm = df_plot.copy()
    for stat in labels:
        df_norm[stat] = (df_plot[stat] - df[stat].min()) / (df[stat].max() - df[stat].min())
    
    # Radar chart setup
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    plt.figure(figsize=(8, 8))
    
    for i, row in df_norm.iterrows():
        values = row[labels].tolist()  # <--- FIXED
        values += values[:1]
        plt.polar(angles, values, label=row['PLAYER_NAME'])
        plt.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], labels)
    plt.title(f"Similarity Radar: {target_player} vs Top 5 Similar Players")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.show()

def plot_role_hierarchy(cluster_df, top_n=10):
    """
     visual dendrogram for the top-N most similar players.
    """

    # find role-defining player
    X = cluster_df[['Principal Component 1', 'Principal Component 2']].values
    centroid = X.mean(axis=0)
    dist_to_centroid = cdist(X, [centroid], metric='euclidean').flatten()
    rep_idx = np.argmin(dist_to_centroid)
    rep_player = cluster_df.iloc[rep_idx]['PLAYER_NAME']

    st.markdown(f"###  Role-Defining Player: **{rep_player}**")

    # Distances from representative player
    rep_coords = X[rep_idx].reshape(1, -1)
    dist_from_rep = cdist(X, rep_coords, metric='euclidean').flatten()

    cluster_df = cluster_df.copy()
    cluster_df['rep_distance'] = dist_from_rep
    cluster_top = cluster_df.sort_values('rep_distance').iloc[:top_n]

    X_top = cluster_top[['Principal Component 1', 'Principal Component 2']].values

    # 
    Z = linkage(X_top, method='ward')

    
    plt.figure(figsize=(14, 7))

    dendrogram(
        Z,
        labels=cluster_top['PLAYER_NAME'].tolist(),
        leaf_rotation=30,     
        leaf_font_size=16,   
        color_threshold=0.4 * max(Z[:, 2]),  
        above_threshold_color="gray"
    )

    
    plt.title(f"Hierarchy of Top {top_n} Similar Players — {cluster_top['Clusters'].iloc[0]}",
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Players", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.grid(False)                   
    plt.box(False)                     
    plt.tight_layout()

    st.pyplot(plt.gcf())


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

cluster_players = clustered_df[clustered_df['Clusters'] == cluster_option].reset_index(drop=True)

st.markdown("## Player Hierarchy (Top 10 Similar)")
plot_role_hierarchy(cluster_players, top_n=10)

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

plot_similarity_radar(clustered_df, player_comparison_option, clustered_df.loc[:, "REB": "FT_PCT"], selected_df['index'].to_list())
st.pyplot(plt.gcf())

st.dataframe(selected_df)
