import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np


def plot_3d_embeddings(tokens,embeddings):
    if len(tokens) < 3:
        return go.Figure().add_annotation(text="You need at least 3 tokens for the 3D visualization.",showarrow=False)

    n_components = min(3,len(tokens))
    pca = PCA(n_components=n_components)
    reduced_embs = pca.fit_transform(embeddings)

    if reduced_embs.shape[1] < 3:
        reduced_embs = np.hstack((reduced_embs,np.zeros((reduced_embs.shape[0],3 - reduced_embs.shape[1]))))

    fig = px.scatter_3d(
        x=reduced_embs[:,0],y=reduced_embs[:,1],z=reduced_embs[:,2],
        text=tokens,
        title="PCA Projected Embeddings",
        labels={'x': 'Dim 1','y': 'Dim 2','z': 'Dim 3'},
        opacity=0.8,
        template="plotly_white"  # <- Forza tema chiaro
    )

    fig.update_traces(
        textposition='top center',
        marker=dict(size=8,color='#0ea5e9',line=dict(width=1,color='white')),
        textfont=dict(color='#0f172a',size=12,family="Arial")  # <- Testo scuro
    )

    fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=40),
        scene=dict(
            xaxis=dict(gridcolor="#e2e8f0",backgroundcolor="#f8fafc",zerolinecolor="#cbd5e1"),
            yaxis=dict(gridcolor="#e2e8f0",backgroundcolor="#f8fafc",zerolinecolor="#cbd5e1"),
            zaxis=dict(gridcolor="#e2e8f0",backgroundcolor="#f8fafc",zerolinecolor="#cbd5e1")
        )
    )
    return fig


def plot_attention_heatmap(tokens,attention_matrix_2d,title_suffix=""):
    unique_tokens = [f"{i}: {tok}" for i,tok in enumerate(tokens)]

    fig = px.imshow(
        attention_matrix_2d,
        labels=dict(x="Observed Token (Past)",y="Current Token",color="Attention Weight"),
        x=unique_tokens,
        y=unique_tokens,
        color_continuous_scale="Blues",
        title=f"Attention Map {title_suffix}",
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        width=800,
        height=800,
        margin=dict(t=50,l=50,r=50,b=50),
        font=dict(color="#1e293b")  # <- Font scuro per gli assi
    )
    return fig


def plot_probabilities(words,probs,title="Probability"):
    fig = px.bar(
        x=words,
        y=probs,
        range_y=[0.0,1.0],
        title=title,
        labels={'x': 'Word','y': 'Probability (%)'},
        color=probs,
        color_continuous_scale="Teal",
        template="plotly_white"
    )
    fig.update_yaxes(tickformat=".2%",gridcolor="#e2e8f0")
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        font=dict(color="#1e293b")
    )
    return fig