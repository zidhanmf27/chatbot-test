# 🍽️ Chatbot Rekomendasi Kuliner UMKM Bandung

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![NLP](https://img.shields.io/badge/Natural%20Language%20Processing-TF--IDF-green)

Aplikasi chatbot cerdas berbasis AI untuk membantu Anda menemukan rekomendasi kuliner UMKM terbaik di Bandung. Menggunakan kombinasi **TF-IDF**, **Cosine Similarity**, dan **Advanced Ranking Mechanics** untuk memberikan hasil yang akurat dan relevan.

## ✨ Fitur Unggulan

### 1. 🧠 Search Engine Cerdas

Sistem pencarian tidak hanya mencocokkan kata kunci, tetapi memahami konteks:

- **Phrase Match Boost:** Memprioritaskan restoran yang memiliki menu persis dengan query Anda (misal: "Nasi Goreng Pedas").
- **Content Relevance Coverage:** Semakin banyak kata kunci yang cocok (baik di Nama Restoran maupun Menu), semakin tinggi peringkatnya.
- **Auto-Correction:** Typo kecil? Tidak masalah. Sistem akan memperbaikinya otomatis (misal: "aym gprk" -> "ayam geprek").
- **Semantic Expansion:** Memahami sinonim (misal: cari "Murah" akan mencakup "Terjangkau", "Hemat").
- **Semantic Category Mapping:** Memetakan makanan spesifik ke kategori luas secara otomatis (misal: "Soto", "Bakso" -> "Masakan Indonesia"; "Ramen", "Sushi" -> "Japanese Food") untuk memastikan hasil pencarian selalu penuh dengan rekomendasi relevan.
- **Smart Fallback:** Jika rekomendasi spesifik (misal: "Ramen Murah") habis, sistem akan otomatis mengisi sisa slot dengan rekomendasi sejenis (misal: "Ramen Mahal" atau "Japanese Food" lainnya) agar pengguna tidak menerima hasil kosong.

### 2. ⚠️ Intelligent Warning System

Sistem secara proaktif memberi tahu jika hasil pencarian terbatas untuk preferensi Anda:

- **Quality-Based Filter:** Hanya menghitung ketersediaan data dari restoran yang benar-benar relevan, mengabaikan hasil yang tidak cocok.
- **Smart Notification:** Jika Anda mencari "Ayam Geprek Murah" tapi datanya sedikit, sistem akan menyarankan untuk melihat opsi harga lain.

### 3. 🎨 User Interface Premium

- **Dark Mode Modern:** Tampilan visual yang elegan dan nyaman di mata.
- **Interactive Sidebar:** Filter harga dan kategori yang mudah diakses.
- **Responsive Design:** Tampilan tetap rapi di berbagai ukuran layar.

### Penjelasan Tahapan:

1. **Input Pengguna**: User memasukkan query dalam bahasa natural (misal: "kopi murah di dago").

2. **Preprocessing**: Membersihkan teks dari noise (simbol, spasi ganda) dan mengubah kata ke bentuk dasar (stemming).

3. **Smart Processing**:

   - **Normalisasi Sinonim**: Menyatukan variasi kata (kafe/cafe/ngopi → cafe)
   - **Auto-Correct**: Memperbaiki typo otomatis
   - **Semantic Expansion**: Menambah konteks relevan (nugas → wifi, stopkontak)

4. **Calculation**: Menghitung kemiripan menggunakan TF-IDF dan Cosine Similarity.

5. **Ranking**:

   - Menerapkan Strict/Flexible Mode berdasarkan kategori
   - Memberikan boost untuk lokasi yang sesuai (+15 poin)
   - Memberikan boost besar untuk exact match (+2000 poin)

6. **Output**: Menampilkan Top 5 rekomendasi dengan informasi lengkap dan link Google Maps.

## 📂 Struktur Proyek

```
chatbot-kuliner/
├── app.py                          # Main application file (Frontend Streamlit)
├── chatbot_engine.py               # Core logic for search engine, ranking, and warning system
├── preprocessing.py                # Text cleaning and preparation utilities
├── requirements.txt                # Python dependencies
├── utility/
│   ├── precompute_dataset.py      # Script untuk optimasi dataset
│   └── metadata.py                # Metadata management
├── dataset/
│   └── data-kuliner-umkm-optimized.csv  # Culinary dataset
├── style/
│   └── custom.css                 # CSS files for custom UI styling
└── assets/                        # Images and static resources
```

## 🚀 Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### Langkah Instalasi

1. **Clone repository**

   ```bash
   git clone https://github.com/zidhanmf27/chatbot-kuliner-umkm-bandung.git
   cd chatbot-kuliner-umkm-bandung
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**

   ```bash
   streamlit run app.py
   ```

4. **Akses aplikasi**

   Buka browser dan akses `http://localhost:8501`

## 💡 Cara Penggunaan

1. **Masukkan Query**: Ketik pertanyaan Anda dalam bahasa natural, contoh:

   - "Cafe murah di Dago"
   - "Ayam geprek enak"
   - "Tempat ngopi untuk nugas"

2. **Filter (Opsional)**: Gunakan sidebar untuk memfilter berdasarkan:

   - Kategori harga (Murah/Mahal)
   - Jenis makanan

3. **Lihat Hasil**: Sistem akan menampilkan 5 rekomendasi terbaik dengan:
   - Nama restoran
   - Menu unggulan
   - Harga
   - Lokasi
   - Link Google Maps

## 🛠️ Teknologi

- **Backend:** Python, Pandas, Numpy, Scikit-learn
- **Frontend:** Streamlit
- **NLP:** Sastrawi (Stemming Bahasa Indonesia), RapidFuzz (Fuzzy Matching)
- **Machine Learning:** TF-IDF, Cosine Similarity

## 📄 Sumber Data

Dataset ini bersumber dari [Open Data Bandung - Data Rumah Makan Restoran Cafe](https://opendata.bandung.go.id/dataset/data-rumah-makan-restoran-cafe-di-kota-bandung), dikelola oleh Dinas Kebudayaan dan Pariwisata Kota Bandung.

## 📝 Lisensi

Project ini dibuat untuk keperluan akademik dan mendukung UMKM Kuliner Bandung.

---

_Dibuat dengan ❤️ untuk mendukung UMKM Kuliner Bandung_
