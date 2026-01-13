import streamlit as st
import tensorflow as tf
import numpy as np
import os
import mitdeeplearning as mdl
import urllib.parse

# --- 1. KONFIGURASI ---
os.environ["COMET_API_KEY"] = "JPM33OpbFxafASglpKXm1giB7"
st.set_page_config(page_title="AI Music Composer", page_icon="🎼", layout="wide")

@st.cache_resource
def load_resources():
    songs = mdl.lab1.load_training_data()
    all_songs_concat = "\n\n".join(songs)
    vocab = sorted(set(all_songs_concat))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    return vocab, char2idx, idx2char

vocab, char2idx, idx2char = load_resources()

# --- 2. MODEL (Update Keras 3 Compatibility) ---
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # Menggunakan InputLayer agar batch_size terdefinisi dengan jelas
        tf.keras.layers.InputLayer(batch_shape=[batch_size, None]),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

@st.cache_resource
def get_model():
    vocab_size = len(vocab)
    model = build_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=1)
    
    checkpoint_path = 'my_ckpt.weights.h5'
    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
            return model, True
        except Exception as e:
            print(f"Error: {e}")
            return model, False
    return None, False

# --- 3. GENERASI TEKS (Perbaikan reset_states) ---
def generate_music_text(model, start_string, length, temp):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    
    # --- PERBAIKAN STATE RESET (Anti-Error) ---
    if hasattr(model, 'reset_states'):
        model.reset_states()
    elif hasattr(model, 'reset_state'):
        model.reset_state()
    else:
        # Jika keduanya tidak ada (jarang terjadi), kita buat state manual
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
    # ------------------------------------------

    for i in range(length):
        predictions = model(input_eval)
        # Hapus dimensi batch
        predictions = tf.squeeze(predictions, 0)
        
        # Gunakan temperature untuk prediksi
        predictions = predictions / temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        # Masukkan hasil prediksi sebagai input berikutnya
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# --- 4. UI ---
st.title("🎹 AI Music Generator")
col1, col2 = st.columns([1, 2])

with col1:
    seed = st.text_input("Seed Text", "X:1")
    gen_length = st.slider("Panjang Karakter", 100, 1000, 500)
    temp = st.slider("Kreativitas", 0.1, 1.5, 0.8)
    generate_btn = st.button("🎼 Gubah Musik")

if generate_btn:
    with st.spinner("Menggubah nada..."):
        model, is_loaded = get_model()
        
        if model is None:
            st.error("File weights tidak ditemukan!")
        else:
            if not is_loaded:
                st.warning("Bobot gagal dimuat, menggunakan nada acak.")
            
            abc_output = generate_music_text(model, seed, gen_length, temp)
            
            with col2:
                st.subheader("📝 Hasil Notasi ABC")
                st.code(abc_output, language="text")
                
                # Player Online (Paling Aman untuk Cloud)
                encoded_abc = urllib.parse.quote(abc_output)
                player_url = f"https://abcjs.net/abcjs-editor.html?abc={encoded_abc}"
                st.markdown(f'<iframe src="{player_url}" width="100%" height="500px" style="border-radius:10px;"></iframe>', unsafe_allow_html=True)