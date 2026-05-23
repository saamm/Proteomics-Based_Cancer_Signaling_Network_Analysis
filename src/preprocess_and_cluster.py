import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.makedirs("results", exist_ok=True)

df = pd.read_csv(
    "data/raw/HS_CPTAC_BRCA_2018_Proteome_Ratio_Norm_gene_Median.cct",
    sep="\t"
)
print("Original Shape:", df.shape)

#CLEAN DATA
#renaming the column
df = df.rename(columns={"IDX": "Protein"})
# Set protein names as index
df = df.set_index("Protein")
# Transpose:
# rows = samples
# columns = proteins
df = df.T
print("Transposed Shape:", df.shape)
#null values
#print(df.isnull().sum())

# Convert everything to numeric
df = df.apply(pd.to_numeric, errors="coerce")


# HANDLE MISSING VALUES
# Remove proteins with >30% missing values
missing_fraction = df.isna().mean()
good_cols = missing_fraction[missing_fraction < 0.3].index

df = df[good_cols]
# Fill remaining missing values using median
df = df.fillna(df.median())

# Save cleaned matrix
df.to_csv("data/processed/cleaned_proteomics_matrix.csv")

# STANDARDIZE DATA
scaler = StandardScaler()

scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(
    scaled,
    index=df.index,
    columns=df.columns
)
# Save scaled matrix
df_scaled.to_csv("data/processed/scaled_proteomics_matrix.csv")

# SELECT TOP VARIABLE PROTEINS
variance = df_scaled.var()

top_proteins = variance.sort_values(
    ascending=False
).head(200).index

df_top = df_scaled[top_proteins]
print("Top protein matrix:", df_top.shape)

# Save top proteins
pd.Series(top_proteins).to_csv(
    "data/processed/top_200_variable_proteins.csv",
    index=False
)

# PCA ANALYSIS
pca = PCA(n_components=2)

X_pca = pca.fit_transform(df_top) #distinct molecular states

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1]
}, index=df_top.index)

# Save PCA coordinates
pca_df.to_csv("data/processed/pca_coordinates.csv")

# Plot PCA
plt.figure(figsize=(8, 6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of CPTAC Breast Cancer Proteomics")

plt.tight_layout()

plt.savefig(
    "results/pca_plot.png",
    dpi=300
)

plt.close()

# KMEANS CLUSTERING
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(df_top)

pca_df["Cluster"] = clusters

# Save cluster assignments
pca_df.to_csv(
    "data/processed/pca_clusters.csv"
)

# Plot clustered PCA
plt.figure(figsize=(8, 6))

for cluster in sorted(pca_df["Cluster"].unique()):

    subset = pca_df[
        pca_df["Cluster"] == cluster
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster}"
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Tumor Clusters from Proteomics Data")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/kmeans_pca_clusters.png",
    dpi=300
)

plt.close()

# PROTEIN CORRELATION NETWORK
corr = df_top.corr()

# Save correlation matrix
corr.to_csv(
    "data/processed/protein_correlation_matrix.csv"
)


## BUILD NETWORKX GRAPH for signaling interaction network approximation
G = nx.Graph()

threshold = 0.7

proteins = corr.columns

for i in range(len(proteins)):

    for j in range(i + 1, len(proteins)):

        p1 = proteins[i]
        p2 = proteins[j]

        value = corr.loc[p1, p2]

        if abs(value) > threshold:

            G.add_edge(
                p1,
                p2,
                weight=value
            )

print("Network Nodes:", G.number_of_nodes())
print("Network Edges:", G.number_of_edges())

# NETWORK VISUALIZATION
plt.figure(figsize=(12, 12))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw_networkx(
    G,
    pos=pos,
    with_labels=False,
    node_size=40,
    width=0.5
)

plt.title("Protein Signaling Correlation Network")

plt.tight_layout()

plt.savefig(
    "results/protein_network.png",
    dpi=300
)

plt.close()

# HUB PROTEIN ANALYSIS
centrality = nx.degree_centrality(G)

centrality_df = pd.DataFrame({
    "Protein": list(centrality.keys()),
    "Centrality": list(centrality.values())
})

centrality_df = centrality_df.sort_values(
    by="Centrality",
    ascending=False
)

# Save hub proteins
centrality_df.to_csv(
    "data/processed/hub_proteins.csv",
    index=False
)

print("\nTop Hub Proteins:")
print(centrality_df.head(20))

# SAVE NETWORK EDGES
edges = nx.to_pandas_edgelist(G)

edges.to_csv(
    "data/processed/network_edges.csv",
    index=False
)

print("\nAnalysis Complete.")


