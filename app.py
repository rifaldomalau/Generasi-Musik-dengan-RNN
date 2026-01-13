import streamlit as st
import tensorflow as tf
import numpy as np
import os
import mitdeeplearning as mdl
from music21 import converter
import urllib.parse

# --- 1. KONFIGURASI INTERNAL ---
os.environ["COMET_API_KEY"] = "JPM33OpbFxafASglpKXm1giB7"

st.set_page_config(page_title="AI Music Composer", page_icon="🎼", layout="wide")

@st.cache_resource
def load_resources():
    # MEMUAT DATASET ASLI UNTUK MEMBENTUK VOCAB
    songs = mdl.lab1.load_training_data()
    all_songs_concat = "\n\n".join(songs)
    vocab = sorted(set(all_songs_concat))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    return vocab, char2idx, idx2char

vocab, char2idx, idx2char = load_resources()

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])

@st.cache_resource
def get_model():
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024
    
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    
    # Path file bobot - PASTIKAN NAMA INI BENAR
    checkpoint_path = 'my_ckpt.weights.h5' 
    
    if os.path.exists(checkpoint_path):
        try:
            # Langkah Krusial: Build model dengan input shape sebelum load
            model.build(tf.TensorShape([1, None])) 
            # Load weights dengan skip_mismatch jika ada sedikit perbedaan versi
            model.load_weights(checkpoint_path)
            return model, True
        except Exception as e:
            # Cetak error ke terminal agar kita bisa diagnosa
            print(f"DEBUG ERROR: {e}")
            return model, False
    return None, False

# --- FUNGSI GENERASI ---
def generate_music_text(model, start_string, length, temp):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()

    for i in range(length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# --- UI UTAMA ---
st.title("🎼 AI Music Generator (RNN)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Konfigurasi")
    seed = st.text_input("Seed Text", "X:1") # Gunakan X:1 sebagai standar ABC
    gen_length = st.slider("Panjang Karakter", 100, 1000, 500)
    temp = st.slider("Kreativitas", 0.1, 1.5, 0.8) # 0.8 biasanya lebih stabil
    generate_btn = st.button("🎹 Gubah Musik")

if generate_btn:
    with st.spinner("Sedang memproses nada..."):
        model, is_loaded = get_model()
        
        if model is None:
            st.error(f"❌ File '{checkpoint_path}' tidak ditemukan!")
            st.stop()
        elif not is_loaded:
            st.warning("⚠️ Gagal memuat bobot (Weights Mismatch). Menampilkan hasil acak.")
        
        abc_output = generate_music_text(model, seed, gen_length, temp)
        
        with col2:
            st.subheader("📝 Hasil Notasi ABC")
            st.code(abc_output, language="text")

            # Online Player (Paling Stabil)
            encoded_abc = urllib.parse.quote(abc_output)
            player_url = f"https://abcjs.net/abcjs-editor.html?abc={encoded_abc}"
            st.markdown(f'<iframe src="{player_url}" width="100%" height="400px"></iframe>', unsafe_allow_html=True)