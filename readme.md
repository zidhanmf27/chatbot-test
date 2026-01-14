# 🍽️ Chatbot Rekomendasi Kuliner UMKM Bandung

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![NLP](https://img.shields.io/badge/Natural%20Language%20Processing-TF--IDF-green)

Aplikasi chatbot cerdas berbasis AI untuk membantu Anda menemukan rekomendasi kuliner UMKM terbaik di Bandung. Menggunakan kombinasi **TF-IDF**, **Cosine Similarity**, dan **Advanced Ranking Mechanics** untuk memberikan hasil yang akurat dan relevan.

## ✨ Fitur Unggulan

### 1. 🧠 Search Engine Cerdas

Sistem pencarian tidak hanya mencocokkan kata kunci, tetapi memahami konteks:

- **Auto-Correction Cerdas:** Sistem memperbaiki typo secara otomatis dengan memprioritaskan kata-kata penting (misal: "kopu" → "kopi" bukan "kopo").
- **Normalisasi Sinonim:** Menyatukan variasi kata yang sama (kafe/cafe/ngopi → cafe & dessert).
- **Semantic Expansion:** Memahami konteks pencarian (misal: "nugas" → menambahkan "wifi", "stopkontak").
- **Phrase Match Boost:** Memprioritaskan restoran yang memiliki menu persis dengan query Anda (+50 poin).
- **Content Relevance:** Semakin banyak kata kunci yang cocok di Nama Restoran dan Menu, semakin tinggi peringkatnya.
- **Perfect Match Boost:** Kombinasi kategori + harga yang tepat mendapat boost besar (+50 poin).
- **Exact Name Matching:** Pencarian nama restoran langsung mendapat boost tertinggi (+2000 poin).

### 2. 🎯 Sistem Ranking Multi-Layer

- **Strict Mode untuk Kategori:** Jika Anda mencari "Cafe", hanya cafe yang akan muncul (bukan restoran lain).
- **Flexible Mode untuk Detail:** Pencarian dengan banyak filter (lokasi + harga + menu) tetap fleksibel.
- **Location Boost:** Lokasi yang sesuai mendapat +15 poin, yang tidak sesuai -50 poin.
- **Price Filter Intelligence:** Sistem memahami berbagai variasi kata harga (murah/terjangkau/hemat).

### 3. ⚠️ Intelligent Warning System

Sistem secara proaktif memberi tahu jika hasil pencarian terbatas:

- **Quality-Based Filter:** Hanya menghitung ketersediaan dari restoran yang benar-benar relevan.
- **Smart Notification:** Memberi tahu jika kombinasi kategori + harga yang dicari tidak tersedia di top 5.

### 4. 🎨 User Interface Premium

- **Dark/Light Mode:** Pilihan tema yang dapat diubah secara real-time.
- **Interactive Sidebar:** Filter harga dan statistik kategori yang mudah diakses.
- **Responsive Design:** Tampilan tetap rapi di berbagai ukuran layar.
- **Quick Search Buttons:** Tombol cepat untuk pencarian populer (Kopi, Ramen, Masakan Sunda, Roti).
- **Google Maps Integration:** Setiap rekomendasi dilengkapi tombol untuk melihat lokasi di Google Maps.

## 🔄 Alur Kerja Sistem

### Pipeline Pemrosesan Query:

1. **Input Pengguna**: User memasukkan query dalam bahasa natural (misal: "kopi murah di dago").

2. **Pembersihan Dasar**: Mengubah ke huruf kecil, menghapus karakter khusus.

3. **Auto-Correction**: Memperbaiki typo dengan prioritas kata penting.

4. **Normalisasi Sinonim**: Menyatukan variasi kata (kafe → cafe & dessert).

5. **Ekspansi Semantik**: Menambahkan konteks relevan (nugas → wifi, stopkontak).

6. **Preprocessing Akhir**: Stemming dan penghapusan stopwords untuk TF-IDF.

7. **Perhitungan Kemiripan**: Menggunakan TF-IDF dan Cosine Similarity.

8. **Ranking Multi-Layer**:

   - Category Matching (Strict/Flexible Mode)
   - Location Boost (+15/-50 poin)
   - Content Boost (+10 poin per kata cocok)
   - Price Boost (+15 poin)
   - Perfect Match Boost (+50 poin)
   - Exact Name Matching (+2000 poin)

9. **Output**: Menampilkan Top 5 rekomendasi dengan informasi lengkap.

## 💡 Cara Penggunaan

1. **Masukkan Query**: Ketik pertanyaan Anda dalam bahasa natural, contoh:

   - "Cafe murah di Dago"
   - "Ayam geprek enak"
   - "Tempat ngopi untuk nugas"
   - "Ramen di Sukajadi"

2. **Filter (Opsional)**: Gunakan sidebar untuk memfilter berdasarkan:

   - Kategori harga (Murah/Sedang/Mahal)

3. **Lihat Hasil**: Sistem akan menampilkan 5 rekomendasi terbaik dengan:

   - Nama restoran & kategori
   - Alamat lengkap
   - Menu unggulan
   - Range harga & kategori harga
   - Deskripsi
   - Persentase kesesuaian (Match %)
   - Tombol ke Google Maps

4. **Muat Lebih Banyak**: Klik tombol "Lebih Banyak" untuk melihat rekomendasi tambahan.

## 🛠️ Teknologi

- **Backend:** Python 3.8+, Pandas, NumPy, Scikit-learn
- **Frontend:** Streamlit 1.28+
- **NLP:**
  - Sastrawi (Stemming Bahasa Indonesia)
  - RapidFuzz (Fuzzy Matching)
  - Custom Auto-Correction dengan Priority Vocabulary
- **Machine Learning:** TF-IDF Vectorization, Cosine Similarity
- **Data Processing:** Custom Metadata Generation, Dataset Optimization

## 📁 Struktur Project

```
chatbot-kuliner/
├── app.py                          # Aplikasi Streamlit utama
├── chatbot_engine.py               # Mesin rekomendasi & ranking
├── preprocessing.py                # Modul preprocessing teks
├── dataset/
│   ├── data-test.csv              # Dataset asli
│   └── dataset-kuliner-umkm-optimized.csv  # Dataset teroptimasi
├── utility/
│   ├── generate_metadata.py       # Script generate metadata
│   └── precompute_dataset.py      # Script optimasi dataset
├── style/
│   ├── app.css                    # Custom styling
│   └── icon.png                   # Icon aplikasi
└── README.md
```

## 🚀 Instalasi & Menjalankan

1. **Clone Repository**

   ```bash
   git clone https://github.com/zidhanmf27/chatbot-kuliner-umkm-bandung.git
   cd chatbot-kuliner-umkm-bandung
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Aplikasi**

   ```bash
   streamlit run app.py
   ```

4. **Akses Aplikasi**
   - Buka browser di `http://localhost:8501`

## 📊 Optimasi Performa

- **Dataset Pre-processing:** Dataset di-preprocess terlebih dahulu untuk menghindari stemming berulang.
- **Caching:** Menggunakan `@st.cache_resource` untuk memuat chatbot engine sekali saja.
- **Efficient Filtering:** Sistem filtering yang optimal untuk pencarian cepat.

## 📄 Sumber Data

Dataset ini bersumber dari [Open Data Bandung - Data Rumah Makan Restoran Cafe](https://opendata.bandung.go.id/dataset/data-rumah-makan-restoran-cafe-di-kota-bandung), dikelola oleh Dinas Kebudayaan dan Pariwisata Kota Bandung.

## � Pengembangan

### Menambah Data Baru

1. Edit file `dataset/data-test.csv`
2. Jalankan `utility/generate_metadata.py` untuk generate metadata
3. Jalankan `utility/precompute_dataset.py` untuk optimasi
4. Restart aplikasi

### Kustomisasi

- **Sinonim:** Edit `SYNONYM_MAP` di `chatbot_engine.py`
- **Semantic Expansion:** Edit `SEMANTIC_EXPANSION` di `chatbot_engine.py`
- **Stopwords:** Edit `CULINARY_STOPWORDS` di `preprocessing.py`
- **Styling:** Edit `style/app.css`

## �📝 Lisensi

Project ini dibuat untuk keperluan akademik dan mendukung UMKM Kuliner Bandung.

## 👨‍💻 Developer

Dikembangkan oleh **[zidhanmf](https://zidhanmf-portofolio.vercel.app/)**

## 🙏 Kontribusi

Kontribusi, issues, dan feature requests sangat diterima!

---

_Dibuat dengan ❤️ untuk mendukung UMKM Kuliner Bandung_
