import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Animal domain reference sentences
REFERENCE_SENTENCES = [
    "Cats are often kept as pets.",
    "Lions live in prides and hunt in groups.",
    "Elephants are the largest land animals.",
    "Dogs are known for their loyalty to humans.",
    "Wolves hunt in packs and communicate through howls.",
    "Cows are raised for milk and meat.",
    "Horses have been used for transport for centuries.",
    "Sheep are farmed for their wool.",
    "Bats are the only mammals capable of true flight.",
    "Bears hibernate during the winter months.",
    "Owls are nocturnal birds with excellent hearing.",
    "Penguins are flightless birds that live in cold climates.",
    "Eagles build large nests high in the mountains.",
    "Ducks can fly, swim, and walk with ease.",
    "Snakes use their tongues to sense their surroundings.",
    "Frogs live near water and can jump long distances.",
    "Spiders build webs to catch their prey.",
    "Bees pollinate flowers and produce honey.",
    "Butterflies undergo metamorphosis.",
    "Ants work in colonies with a clear social structure."
]

# Load model and fit PCA
@st.cache_data()
def setup_model_and_pca():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_embeddings = model.encode(REFERENCE_SENTENCES)
    pca = PCA(n_components=2)
    pca.fit(ref_embeddings)
    return model, pca

model, pca = setup_model_and_pca()

st.title("🐾 2D Semantic Space for Animals")
st.markdown("Enter animal names below. Each one will be embedded and projected into a space shaped by animal-related knowledge.")

input_text = st.text_area("Enter animal names (one per line):", "cat\ndog\nlion\nelephant\nowl\nbee")

if input_text.strip():
    animal_names = [line.strip() for line in input_text.strip().splitlines() if line.strip()]
    embeddings = model.encode(animal_names)
    vectors_2d = pca.transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = cm.get_cmap("tab10", len(animal_names))

    for i, (x, y) in enumerate(vectors_2d):
        ax.arrow(0, 0, x, y, head_width=0.02, head_length=0.05, length_includes_head=True, alpha=0.6, color=cmap(i))
        ax.text(x * 1.05, y * 1.05, animal_names[i], fontsize=10, color=cmap(i))

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title("Animal Names in 2D Semantic Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axis('equal')
    ax.grid(True)
    st.pyplot(fig)
