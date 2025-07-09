import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load embedding model once
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.title("🐾 Build a 2D Semantic Space for Animals")

st.markdown("""
This tool creates a **2D semantic space** using text you provide about animals — facts, behaviours, habitats, etc.

1. In the first box, enter **reference sentences** about animals. These shape the semantic space.
2. In the second box, enter **animal names** (one per line). These will be embedded and projected into that space.

You can explore how different reference texts change the structure of meaning.
""")

# --- Reference Text Input ---
st.header("Step 1: Provide Reference Texts")
default_reference = "\n".join([
    "Lions hunt in groups called prides.",
    "Owls can rotate their heads almost fully around.",
    "Penguins live in cold climates and cannot fly.",
    "Dogs are domesticated and known for loyalty.",
    "Bees produce honey and pollinate flowers.",
    "Cats are independent and often kept as pets."
])
ref_input = st.text_area("Reference sentences (one per line):", default_reference, height=200)

reference_sentences = [line.strip() for line in ref_input.splitlines() if line.strip()]

if reference_sentences:
    # Fit PCA on the user's reference texts
    ref_embeddings = model.encode(reference_sentences)
    pca = PCA(n_components=2)
    pca.fit(ref_embeddings)

    # --- Animal Name Input ---
    st.header("Step 2: Add Animal Names to Visualise")

    animal_input = st.text_area("Animal names (one per line):", "cat\ndog\nlion\nbee\nelephant\nowl", height=150)

    animal_names = [line.strip() for line in animal_input.splitlines() if line.strip()]
    if animal_names:
        animal_embeddings = model.encode(animal_names)
        projected = pca.transform(animal_embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = cm.get_cmap("tab10", len(animal_names))

        for i, (x, y) in enumerate(projected):
            ax.arrow(0, 0, x, y, head_width=0.02, head_length=0.05, length_includes_head=True,
                     alpha=0.7, color=cmap(i))
            ax.text(x * 1.05, y * 1.05, animal_names[i], fontsize=10, color=cmap(i))

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_title("Animals in 2D Semantic Space")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)
        ax.axis('equal')
        st.pyplot(fig)
