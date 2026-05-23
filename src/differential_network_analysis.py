import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# CREATE RESULTS DIRECTORY

os.makedirs("results/differential_networks", exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df_scaled = pd.read_csv(
    "data/processed/scaled_proteomics_matrix.csv",
    index_col=0
)

pca_clusters = pd.read_csv(
    "data/processed/pca_clusters.csv",
    index_col=0
)

top_proteins = pd.read_csv(
    "data/processed/top_200_variable_proteins.csv"
).iloc[:, 0].tolist()

# Keep only top proteins
df_top = df_scaled[top_proteins]

# Add cluster labels
df_top["Cluster"] = pca_clusters["Cluster"]


# DIFFERENTIAL NETWORK ANALYSIS

network_summary = []

for cluster_id in sorted(df_top["Cluster"].unique()):

    print(f"\n========== Cluster {cluster_id} ==========")

    # Subset samples

    cluster_samples = df_top[
        df_top["Cluster"] == cluster_id
    ].drop(columns=["Cluster"])

    print("Samples:", cluster_samples.shape[0])


    # Correlation Matrix

    corr = cluster_samples.corr()

    # Save correlation matrix
    corr.to_csv(
        f"results/differential_networks/"
        f"cluster_{cluster_id}_correlation.csv"
    )


    # Build Network
    # -----------------------------------------------------

    G = nx.Graph()

    threshold = 0.6

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


    # Network Statistics

    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    print("Nodes:", nodes)
    print("Edges:", edges)

    density = nx.density(G)

    network_summary.append({
        "Cluster": cluster_id,
        "Samples": cluster_samples.shape[0],
        "Nodes": nodes,
        "Edges": edges,
        "Density": density
    })

       # Hub Protein Analysis


    centrality = nx.degree_centrality(G)

    hub_df = pd.DataFrame({
        "Protein": list(centrality.keys()),
        "Centrality": list(centrality.values())
    })

    hub_df = hub_df.sort_values(
        by="Centrality",
        ascending=False
    )

    # Save hubs
    hub_df.to_csv(
        f"results/differential_networks/"
        f"cluster_{cluster_id}_hub_proteins.csv",
        index=False
    )

    print("\nTop Hub Proteins:")
    print(hub_df.head(10))

     # Network Visualization

    plt.figure(figsize=(10, 10))

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

    plt.title(
        f"Cluster {cluster_id} Signaling Network"
    )

    plt.tight_layout()

    plt.savefig(
        f"results/differential_networks/"
        f"cluster_{cluster_id}_network.png",
        dpi=300
    )

    plt.close()

# SAVE NETWORK SUMMARY

summary_df = pd.DataFrame(network_summary)

summary_df.to_csv(
    "results/differential_networks/network_summary.csv",
    index=False
)

print("\nDifferential Network Analysis Complete.")