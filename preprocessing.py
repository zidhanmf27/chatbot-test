# ============================================================================
# TEXT PREPROCESSING MODULE
# ============================================================================

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# ============================================================================
# STOPWORDS KUSTOM
# ============================================================================

CULINARY_STOPWORDS = {
    'makan', 'makanan', 
    'minum', 'minuman', 
    'kuliner', 'masak', 'masakan', 
    'tempat', 'resto', 'restoran', 'warung', 'kafe', 'cafe', 'kedai',
    'cari', 'mau', 'ingin', 'tolong', 'rekomendasi', 'rekomendasiin', 'minta', 'bantu', 'kasih', 'info', 'spill',
    'daerah', 'wilayah', 'sekitar', 'dekat', 'deket', 'kawasan',
    'nya', 'dong', 'sih', 'tuh', 'nih', 'banget', 'sangat', 'paling', 'bgt',
    'buat', 'untuk', 'yg', 'yang', 'dgn', 'dengan', 'dan', 'atau', 'di', 'ke', 'dari',
    'enak', 'lezat', 'nikmat', 'mantap', 'bagus', 'terbaik', 'populer', 'favorit',
    'hits', 'viral', 'kekinian', 'legend',
    'halo', 'hai', 'hi', 'pagi', 'siang', 'malam', 'sore',
    'kak', 'min', 'gan', 'sis', 'bro', 'pak', 'bu', 'mas', 'mba'
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
    
    def clean_text(self, text):
        """Case Folding & Cleaning"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text):
        """Tokenizing: Memecah kalimat menjadi list of tokens"""
        return text.split()
    
    def remove_stopwords(self, tokens):
        """Stopword Removal"""
        return [word for word in tokens if word not in self.stopwords]
    
    def stem_tokens(self, tokens):
        """Stemming menggunakan Sastrawi"""
        return [self.stemmer.stem(word) for word in tokens]
    
    def preprocess(self, text):
        """Pipeline Preprocessing Lengkap"""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        return " ".join(tokens)
    
    def preprocess_dataframe_column(self, df, column_name):
        """Preprocessing untuk kolom DataFrame"""
        print("[INFO] Sedang memproses data...")
        return df[column_name].apply(lambda x: self.preprocess(str(x)))
