import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("🧠 Embeddings en vivo")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

words_input = st.text_input("Introduce palabras separadas por comas", "gato, perro, coche")

if words_input:
    words = [w.strip() for w in words_input.split(",")]
    embeddings = model.encode(words)

    st.subheader("📏 Similitudes")
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            st.write(f"{words[i]} vs {words[j]} → {sim:.3f}")

    if len(words) > 2:
        st.subheader("🗺️ Mapa semántico")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        fig, ax = plt.subplots()

        for i, word in enumerate(words):
            x, y = reduced[i]
            ax.scatter(x, y)
            ax.text(x, y, word)

        st.pyplot(fig)
