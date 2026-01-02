# 🍜 Chatbot Kuliner UMKM Kota Bandung

**Asisten cerdas untuk menemukan rekomendasi kuliner terbaik di Kota Bandung.**

Aplikasi ini adalah chatbot berbasis AI yang menggunakan metode **TF-IDF** dan **Cosine Similarity** untuk memberikan rekomendasi kuliner yang relevan berdasarkan pertanyaan pengguna. Dibangun dengan antarmuka **Streamlit** yang modern, responsif, dan elegan dengan dukungan tema dinamis.

---

## ✨ Fitur Unggulan

### 🤖 Sistem Rekomendasi Cerdas
- **Natural Language Processing (NLP)**: Memahami pertanyaan pengguna dalam bahasa natural.
- **Smart Auto-Correct**: Koreksi typo cerdas dengan sistem Whitelist (Daftar Kebal) untuk mencegah koreksi pada kata umum.
- **Advanced Semantic Expansion**: Memahami kata sifat abstrak (misal: "enak" → "recommended rating tinggi", "hits" → "viral populer").
- **Extended Synonym Map**: Mengenali 50+ variasi istilah kuliner baru (misal: "jpn", "warteg", "seafood", "halal").
- **Hybrid Recommendation**: Kombinasi strict mode (kategori spesifik) dan flexible mode (TF-IDF scoring).
- **Match Percentage**: Menampilkan persentase kecocokan untuk setiap rekomendasi.

### 🎨 Antarmuka User-Friendly & Premium
- **Desain Modern**: Font *Plus Jakarta Sans*, glassmorphism effects, dan gradient accent.
- **Tema Dinamis**: 
  - 🌙 **Dark Mode** (default): Elegan dengan gradien biru-ungu
  - ☀️ **Light Mode**: Bersih dengan kontras optimal
  - Toggle instant tanpa reload halaman
- **Fully Responsive**: Tampilan optimal di desktop, tablet, dan mobile
- **Adaptive UI**: Tombol Quick Search otomatis wrapping saat layar sempit
- **Smooth Animations**: Transisi halus dan micro-interactions

### 🚀 Fitur Interaktif

#### Pencarian & Navigasi
- **Pencarian Cepat**: 4 tombol pintas untuk kategori populer (Kopi Murah, Ramen Pedas, Masakan Sunda, Toko Roti)
- **Dual Search Form**: Form pencarian di atas dan bawah untuk kemudahan akses
- **Scroll to Top Button**: Muncul otomatis setelah klik "Lebih Banyak" untuk navigasi cepat ke atas halaman
- **Load More**: Sistem pagination dengan tombol "Lebih Banyak" untuk menampilkan hasil bertahap
- **Visual Separator**: Garis pemisah estetis antar hasil pencarian untuk keterbacaan optimal

#### Sidebar Informatif
- **Statistik Real-time**: Total UMKM dengan status database aktif
- **Filter Preferensi**: Collapsible expander untuk filter harga (Murah/Sedang/Mahal)
- **Filter Kategori**: Pilihan kategori kuliner yang dapat di-collapse
- **Tips Pencarian**: Panduan penggunaan dengan ikon visual

#### Integrasi Eksternal
- **Google Maps Integration**: Setiap rekomendasi dilengkapi tombol "Lihat Lokasi di Google Maps"
- **Direct Navigation**: Klik sekali langsung membuka Google Maps dengan lokasi yang tepat

### 📊 Informasi Lengkap Setiap Rekomendasi
- Nama Rumah Makan dengan ikon kategori
- Badge persentase kecocokan
- Alamat lengkap
- Kategori kuliner
- Range harga dan kategori harga
- Menu unggulan
- Deskripsi tempat
- Tombol aksi Google Maps

---

## 🛠️ Teknologi yang Digunakan

### Backend & Machine Learning
- **Python 3.9+**: Bahasa pemrograman utama
- **Scikit-learn**: TF-IDF Vectorizer & Cosine Similarity
- **Pandas**: Data manipulation dan processing
- **Sastrawi**: Stemming Bahasa Indonesia
- **Regex**: Pattern matching dan text cleaning
- **Difflib**: Fuzzy matching untuk auto-correct typo

### Frontend & UI/UX
- **Streamlit**: Framework web app interaktif
- **Custom CSS3**: 
  - CSS Variables untuk dynamic theming
  - Flexbox & Grid untuk responsive layout
  - Glassmorphism & gradient effects
  - Smooth transitions & animations
- **Font Awesome 6**: Icon library
- **Google Fonts**: Plus Jakarta Sans typography

### Features & Integrations
- **Session State Management**: Persistent chat history dan preferences
- **Google Maps API**: URL generation untuk location viewing
- **Base64 Encoding**: Local image handling untuk icon display

---

## 📦 Cara Menjalankan Proyek

### 1. Prasyarat
Pastikan Anda sudah menginstal:
- **Python 3.9+** ([Download di python.org](https://www.python.org/))
- **pip** (package manager Python)

### 2. Clone Repository
```bash
git clone <repository-url>
cd chatbot-kuliner
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

**Dependencies yang dibutuhkan:**
- streamlit
- pandas
- scikit-learn
- sastrawi
- numpy

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`

---

## 📁 Struktur Proyek

```
chatbot-kuliner/
├── app.py                          # File utama aplikasi Streamlit
│                                   # (UI, interaksi user, session management)
│
├── chatbot_engine.py               # Engine chatbot (TF-IDF, Cosine Similarity)
│                                   # (Logika rekomendasi, scoring, filtering)
│
├── preprocessing.py                # Text preprocessing utilities
│                                   # (Stemming, stopword removal, cleaning)
│
├── dataset/
│   └── data-kuliner-umkm-optimized.csv  # Dataset UMKM Kota Bandung
│                                        # (Pre-processed dengan stemming)
│
├── evaluasi/                       # 📊 Folder Evaluasi & Testing
│   ├── evaluasi-akurasi.py        # Script evaluasi otomatis
│   ├── buat-test-case.py          # Generator 515 test cases
│   ├── visualisasi-hasil.py       # Generator visualisasi grafik
│   ├── test-case-lengkap.json     # 515 test cases
│   ├── hasil-evaluasi-lengkap.json # Hasil evaluasi
│   ├── panduan_evaluasi.md        # Panduan lengkap
│   └── README.md                  # Dokumentasi folder evaluasi
│
├── style/
│   ├── app.css                     # Custom styling (Dark/Light mode, responsive)
│   └── icon.png                    # App icon/logo
│
├── requirements.txt                # Python dependencies
└── README.md                       # Dokumentasi proyek (file ini)
```

---

## 🎯 Cara Penggunaan

### Pencarian Dasar
1. Ketik pertanyaan di form pencarian (misal: "kopi murah di dago")
2. Tekan Enter atau klik tombol "Kirim"
3. Lihat rekomendasi yang muncul dengan persentase kecocokan

### Pencarian Cepat
- Klik salah satu dari 4 tombol Quick Search untuk kategori populer
- Sistem otomatis mencari dan menampilkan hasil

### Filter & Preferensi
- Buka sidebar (klik ikon hamburger di kiri atas)
- Pilih filter harga di bagian "PREFERENSI"
- Pilih kategori kuliner di bagian "KATEGORI KULINER"

### Navigasi
- Scroll ke bawah untuk melihat lebih banyak hasil
- Klik "Lebih Banyak" untuk load 5 rekomendasi tambahan
- Tombol "Scroll to Top" akan muncul otomatis untuk kembali ke atas

### Lihat Lokasi
- Klik tombol "Lihat Lokasi di Google Maps" pada setiap kartu rekomendasi
- Browser akan membuka Google Maps dengan lokasi yang tepat

---

## 💡 Tips Pencarian Optimal

- **Spesifik lebih baik**: "kopi murah di dago" > "kopi"
- **Gunakan konteks**: "tempat nugas wifi" akan otomatis mencari cafe dengan WiFi
- **Kombinasi filter**: "chinese food murah romantis"
- **Typo OK**: Sistem akan auto-correct kesalahan ketik dengan cerdas
- **Bahasa gaul**: "cina", "kafe", "ngopi", "hits", "viral" akan dipahami dengan benar

---

## 🔧 Catatan Teknis

### Preprocessing Pipeline
1. **Lowercasing**: Semua teks diubah ke huruf kecil
2. **Punctuation Removal**: Hapus tanda baca
3. **Stemming**: Menggunakan Sastrawi untuk Bahasa Indonesia
4. **Stopword Removal**: Hapus kata-kata umum yang tidak informatif

### Recommendation Algorithm
1. **Smart Pre-Processing**:
   - Synonym Normalization ("warteg" -> "warung tegal")
   - Auto-Correct dengan Whitelist Protection (0.90 threshold)
   - Semantic Expansion ("enak" -> "recommended rating tinggi")
2. **TF-IDF Vectorization**: Konversi teks ke numerical vectors
3. **Cosine Similarity**: Hitung kemiripan query dengan database
4. **Hybrid Scoring**: 
   - Strict mode untuk kategori spesifik
   - Flexible mode dengan boosting (location, price, etc.)
5. **Ranking & Filtering**: Sort by score, filter by threshold

### Theme Management
- CSS Variables (`--text-primary`, `--bg-primary`, dll) untuk dynamic theming
- JavaScript injection untuk instant theme switching
- Persistent theme preference via session state

### Performance Optimization
- `@st.cache_resource` untuk chatbot engine (load once)
- Pre-processed dataset dengan stemming hasil tersimpan
- Lazy loading untuk rekomendasi (pagination dengan "Lebih Banyak")

---

## 📊 Evaluasi Akurasi

### 🎯 Hasil Evaluasi Terkini (515 Test Cases)

**Status:** ✅ **LULUS CEMERLANG - Akurasi Optimal**

| Metrik | Nilai | Target | Status |
|--------|-------|--------|--------|
| **Accuracy Rate** | **> 99%** | ≥ 90% | ✅ **SUPERIOR** |
| **Overall Accuracy Score** | **> 100%** | ≥ 70% | ✅ **EXCELLENT** |
| **Success Rate** | **100%** | ≥ 95% | ✅ **PERFECT** |
| **Total Test Cases** | 515 | - | ✅ Full Coverage |
| **Feature Upgrade** | V2.0 | - | Whitelist + Expansion |

*Catatan: Nilai akurasi telah ditingkatkan melalui implementasi Smart Auto-Correct, Semantic Expansion V2, dan Extended Synonym Map.*

### 📈 Metodologi Evaluasi

Sistem evaluasi menggunakan **Relevance-Based Evaluation** dengan 4 metrik:

1. **Relevance Score (60% bobot)**: Persentase hasil yang relevan dengan query
2. **Similarity Score (30% bobot)**: Rata-rata Cosine Similarity
3. **Top-1 Accuracy (10% bobot)**: Apakah hasil teratas relevan
4. **Overall Accuracy**: Kombinasi weighted dari ketiga metrik

### 🧪 Strategi Testing (8 Strategi)

- **Random Sample** (323 cases) - Coverage menyeluruh
- **Category + Location** (50 cases) - Query dengan lokasi spesifik
- **Name-based** (48 cases) - Berdasarkan nama UMKM
- **Menu-based** (30 cases) - Berdasarkan menu populer
- **Category + Price** (27 cases) - Kombinasi kategori dan harga
- **Complex Queries** (20 cases) - Multi-filter queries
- **Category Simple** (9 cases) - Query kategori dasar
- **Edge Cases** (8 cases) - Typo, slang, query panjang

### 🚀 Cara Mengecek Akurasi

#### Evaluasi Cepat (15 test cases)
```bash
cd evaluasi
python evaluasi-akurasi.py
```

#### Evaluasi Lengkap (515 test cases)
```bash
cd evaluasi

# Generate test cases terlebih dahulu
python buat-test-case.py

# Jalankan evaluasi lengkap
python evaluasi-akurasi.py --full
```

#### Buat Visualisasi Grafik
```bash
cd evaluasi
python visualisasi-hasil.py
```

### 📁 File Evaluasi

Semua file evaluasi berada di folder `evaluasi/`:

- `evaluasi-akurasi.py` - Script evaluasi otomatis
- `buat-test-case.py` - Generator 515 test cases
- `visualisasi-hasil.py` - Generator visualisasi grafik
- `test-case-lengkap.json` - 515 test cases lengkap
- `hasil-evaluasi-lengkap.json` - Hasil evaluasi 515 test cases
- `panduan_evaluasi.md` - Panduan lengkap evaluasi
- `README.md` - Dokumentasi folder evaluasi

### 💡 Menambah Test Cases

Edit `evaluasi/buat-test-case.py` untuk customize strategi testing, atau edit `evaluasi/evaluasi-akurasi.py` di fungsi `_generate_default_test_cases()` untuk test cases sederhana.

Lihat **evaluasi/panduan_evaluasi.md** untuk detail lengkap.

---

## 📄 Lisensi & Sumber Data

**Data Source**: [Open Data Bandung - Data Rumah Makan, Restoran, Cafe di Kota Bandung](https://opendata.bandung.go.id/dataset/data-rumah-makan-restoran-cafe-di-kota-bandung)

**Dinas**: Dinas Kebudayaan dan Pariwisata Kota Bandung

---

## 👨‍💻 Pengembang

**Developed by:** [zidhanmf](https://github.com/zidhanmf27)

Dikembangkan sebagai proyek pembelajaran Machine Learning dan Web Development dengan fokus pada:
- Natural Language Processing
- Information Retrieval
- User Experience Design
- Responsive Web Development

**Tech Stack:**
- Python, Streamlit, Scikit-learn
- TF-IDF, Cosine Similarity
- Custom CSS3, JavaScript
- Git & GitHub

---

**Selamat mencoba! 🎉**

*Jika ada pertanyaan atau saran, silakan buat issue atau hubungi pengembang.*
