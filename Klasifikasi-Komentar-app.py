import streamlit as st
import pandas as pd
import gdown
import io

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Konfigurasi halaman
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title("📊 Aplikasi Analisis Sentimen Menggunakan Naive Bayes")

st.write("Web ini membantu mengklasifikasikan komentar atau teks berdasarkan model analisis sentimen.")

# Load dataset
@st.cache_data

def load_data():
    url = "https://drive.google.com/uc?id=1-BLXCH-ywSkVQo-L5HQfOk63sNbTD-6M"
    output = "data.csv"
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df

data = load_data()

# Downsampling
positive_class = data[data['sentiment'] == 'positive']
negative_class = data[data['sentiment'] == 'negative']
neutral_class = data[data['sentiment'] == 'neutral']

min_class_size = min(len(positive_class), len(negative_class), len(neutral_class))

positive_class = resample(positive_class, replace=False, n_samples=min_class_size, random_state=42)
negative_class = resample(negative_class, replace=False, n_samples=min_class_size, random_state=42)
neutral_class = resample(neutral_class, replace=False, n_samples=min_class_size, random_state=42)

df_balanced = pd.concat([positive_class, negative_class, neutral_class])

# Split data
X = df_balanced['final_text'].fillna('')
y = df_balanced['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate Model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# Input User
st.markdown("---")
st.subheader("✏️ Masukkan komentar atau teks:")
user_input = st.text_area("Teks input")

if st.button("🔍 Classify"):
    if user_input.strip():
        user_input_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vec)[0]
        if prediction == "negative":
            st.markdown(f"Komentar tersebut diklasifikasikan sebagai: <span style='color:red; font-weight:bold;'>{prediction.upper()}</span>", unsafe_allow_html=True)
        elif prediction == "positive":
            st.markdown(f"Komentar tersebut diklasifikasikan sebagai: <span style='color:green; font-weight:bold;'>{prediction.upper()}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"Komentar tersebut diklasifikasikan sebagai: <span style='color:blue; font-weight:bold;'>{prediction.upper()}</span>", unsafe_allow_html=True)
    else:
        st.warning("Silakan masukkan komentar yang valid.")

# Tampilkan Akurasi
st.markdown("---")
st.write("🎯 Akurasi Model:", f"{accuracy:.4f}")

# Upload File Batch
st.markdown("---")
st.subheader("📂 Upload File Komentar (CSV)")
uploaded_file = st.file_uploader("Unggah file CSV yang berisi kolom komentar", type=["csv"])

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file)
        if 'komentar' not in df_upload.columns:
            st.error("File harus memiliki kolom bernama 'komentar'")
        else:
            df_upload['komentar'] = df_upload['komentar'].fillna('')
            vec_comments = vectorizer.transform(df_upload['komentar'])
            df_upload['klasifikasi'] = model.predict(vec_comments)

            # Tampilkan hasil dengan warna
            def highlight_sentiment(val):
                color = 'red' if val == 'negative' else 'green' if val == 'positive' else 'blue'
                return f'background-color: {color}; color: white'

            st.write("### Hasil Klasifikasi:")
            st.dataframe(df_upload.style.applymap(highlight_sentiment, subset=['klasifikasi']))

            # Tampilkan total klasifikasi
            st.markdown("### 📊 Total Klasifikasi:")
            klasifikasi_counts = df_upload['klasifikasi'].value_counts()
            st.write(klasifikasi_counts.to_frame().rename(columns={"klasifikasi": "Jumlah"}))

            # Unduh hasil
            csv_out = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil sebagai CSV",
                data=csv_out,
                file_name="hasil_klasifikasi.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
# Upload File Batch
st.markdown("---")
st.subheader("📂 Upload File Komentar (CSV)")
uploaded_file = st.file_uploader("Unggah file CSV yang berisi kolom komentar", type=["csv"])

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file)
        if 'komentar' not in df_upload.columns:
            st.error("File harus memiliki kolom bernama 'komentar'")
        else:
            df_upload['komentar'] = df_upload['komentar'].fillna('')
            vec_comments = vectorizer.transform(df_upload['komentar'])
            df_upload['klasifikasi'] = model.predict(vec_comments)

            # Tampilkan hasil dengan warna
            def highlight_sentiment(val):
                color = 'red' if val == 'negative' else 'green' if val == 'positive' else 'blue'
                return f'background-color: {color}; color: white'

            st.write("### Hasil Klasifikasi:")
            st.dataframe(df_upload.style.applymap(highlight_sentiment, subset=['klasifikasi']))

            # Tampilkan total klasifikasi
            st.markdown("### 📊 Total Klasifikasi:")
            klasifikasi_counts = df_upload['klasifikasi'].value_counts()
            st.write(klasifikasi_counts.to_frame().rename(columns={"klasifikasi": "Jumlah"}))

            # Unduh hasil
            csv_out = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil sebagai CSV",
                data=csv_out,
                file_name="hasil_klasifikasi.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        
