# ============================================================================
# TEXT PREPROCESSING MODULE
# ============================================================================
# Pemrosesan Input User: Menangani Case Folding, Cleaning, Tokenizing,
# Stopword Removal, dan Stemming menggunakan Sastrawi.
# ============================================================================

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# ============================================================================
# STOPWORDS KUSTOM
# ============================================================================

CULINARY_STOPWORDS = {
    'makan', 'makanan', 'minum', 'minuman', 'kuliner', 'masak', 'masakan', 
    'tempat', 'resto', 'restoran', 'warung', 'kafe', 'cafe', 'kedai',
    'cari', 'mau', 'ingin', 'tolong', 'rekomendasi', 'rekomendasiin',
    'daerah', 'wilayah', 'sekitar', 'dekat', 'deket',
    'nya', 'dong', 'sih', 'tuh', 'nih', 'banget', 'sangat',
    'buat', 'untuk', 'yg', 'yang', 'dgn', 'dengan'
}


# ============================================================================
# TEXT PREPROCESSOR CLASS
# ============================================================================

class TextPreprocessor:
    """Kelas untuk preprocessing text menggunakan Sastrawi"""
    
    def __init__(self):
        """Inisialisasi Sastrawi stemmer dan stopword remover"""
        self.stemmer = self._initialize_stemmer()
        self.stopwords = self._initialize_stopwords()
        print("[INFO] Text Preprocessor dengan Sastrawi siap digunakan!")
    
    def _initialize_stemmer(self):
        """Inisialisasi Sastrawi stemmer"""
        stemmer_factory = StemmerFactory()
        return stemmer_factory.create_stemmer()
    
    def _initialize_stopwords(self):
        """Inisialisasi stopwords dengan tambahan kustom"""
        stopword_factory = StopWordRemoverFactory()
        stopwords = set(stopword_factory.get_stop_words())
        stopwords.update(CULINARY_STOPWORDS)
        return stopwords
    
    ## Proses Preprocessing
    def clean_text(self, text):
        """
        Case Folding & Cleaning: Mengubah teks menjadi huruf kecil dan 
        membersihkan karakter yang tidak diinginkan seperti angka, simbol, dan URL.
        """
        if not isinstance(text, str):
            return ""
        
<<<<<<< HEAD
        text = text.lower()  # Case Folding
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Hapus URL
        text = re.sub(r'\S+@\S+', '', text)  # Hapus Email
        # text = re.sub(r'\d+', '', text)  # Hapus Angka (DINONAKTIFKAN: Banyak nama UMKM pakai angka)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Hapus Tanda Baca (Sisakan huruf & angka)
        text = re.sub(r'\s+', ' ', text)  # Hapus Spasi Ganda
=======
        text = text.lower() # Case Folding
        text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Hapus URL
        text = re.sub(r'\S+@\S+', '', text) # Hapus Email
        text = re.sub(r'[^a-z0-9\s]', ' ', text) # Hapus Tanda Baca (Sisakan huruf & angka)
        text = re.sub(r'\s+', ' ', text) # Hapus Spasi Ganda
>>>>>>> b0245ed991ebf7f624d072005a3d19ef5bfdba2c
        return text.strip()
    
    def tokenize(self, text):
        """
        Tokenizing: Memecah kalimat menjadi daftar kata-kata terpisah (list of tokens).
        """
        return text.split()
    
    def remove_stopwords(self, tokens):
        """
        Stopword Removal: Menghapus kata-kata umum yang tidak bermakna spesifik 
        (seperti 'yang', 'dan', 'di') agar analisis lebih fokus.
        """
        return [word for word in tokens if word not in self.stopwords]
    
    def stem_tokens(self, tokens):
        """
        Stemming: Mengubah setiap kata menjadi bentuk dasarnya 
        (misal: 'memakan' -> 'makan') menggunakan algoritma Sastrawi.
        """
        return [self.stemmer.stem(word) for word in tokens]
    
    def preprocess(self, text):
        """
        Pipeline Preprocessing Lengkap: Menjalankan semua tahapan 
        cleaning hingga stemming secara berurutan.
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        return " ".join(tokens)
    
    def preprocess_dataframe_column(self, df, column_name):
        """
        Preprocessing untuk kolom DataFrame.
        Menerapkan preprocessing pada setiap baris di kolom yang ditentukan.
        """
        print("[INFO] Sedang memproses data...")
        return df[column_name].apply(lambda x: self.preprocess(str(x)))