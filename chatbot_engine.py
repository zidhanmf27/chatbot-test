# ============================================================================
# CHATBOT ENGINE - CORE RECOMMENDATION LOGIC
# ============================================================================
# Logic Utama Chatbot: Menangani loading data, perhitungan TF-IDF,
# Cosine Similarity, dan logika rekomendasi (Strict/Flexible).
# ============================================================================

import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import TextPreprocessor
from difflib import get_close_matches


# ============================================================================
# KONSTANTA DAN KONFIGURASI
# ============================================================================

SYNONYM_MAP = {
    # === KATEGORI UTAMA (Semua 9 kategori di dataset) ===
    'cafe': 'cafe & dessert',
    'kafe': 'cafe & dessert',
    'kopi': 'cafe & dessert',
    'coffee': 'cafe & dessert',
    'coffe': 'cafe & dessert',
    'kedai kopi': 'cafe & dessert',
    'ngopi': 'cafe & dessert',
    'cina': 'chinese food',
    'chinese': 'chinese food',
    'china': 'chinese food',
    'jepang': 'japanese food',
    'japan': 'japanese food',
    'jpn': 'japanese food',
    'korea': 'korean food',
    'korean': 'korean food',
    'korsel': 'korean food',
    'barat': 'western food',
    'western': 'western food',
    'arab': 'middle eastern',
    'timur tengah': 'middle eastern',
    'indo': 'masakan indonesia',
    'indonesia': 'masakan indonesia',
    'nusantara': 'masakan indonesia',
    'aneka': 'aneka masakan',
    'campur': 'aneka masakan',
    'non halal': 'masakan non halal',
    'babi': 'masakan non halal',
    'pork': 'masakan non halal',
    
    # === MENU POPULER (Bilingual Indo-Eng) ===
    'ayam': 'chicken', 
    'chicken': 'ayam',
    'sapi': 'beef', 
    'beef': 'sapi', 
    'daging': 'beef',
    'ikan': 'fish', 
    'fish': 'ikan',
    'udang': 'shrimp', 
    'shrimp': 'udang',
    'nasi': 'rice', 
    'rice': 'nasi',
    'mie': 'noodle', 
    'noodle': 'mie', 
    'mi': 'mie',
    'goreng': 'fried', 
    'fried': 'goreng',
    'bakar': 'grilled', 
    'grilled': 'bakar', 
    'panggang': 'bakar', 
    'ikan bakar': 'bakar',
    'kuah': 'soup', 
    'soup': 'kuah',
    'pedas': 'spicy', 
    'spicy': 'pedas', 
    'pedes': 'pedas',
    'manis': 'sweet', 
    'sweet': 'manis',
    
    # === SINGKATAN & VARIASI MENU ===
    'nasgor': 'nasi goreng',
    'nasigoreng': 'nasi goreng',
    'miegoreng': 'mie goreng',
    'miekuah': 'mie kuah',
    'migor': 'mie goreng',
    'aygor': 'ayam goreng',
    'ramen': 'ramen',
    
    # === TYPO UMUM MENU ===
    'stiek': 'steak', 
    'piza': 'pizza', 
    'ayem': 'ayam', 
    'aym': 'ayam',
    
    # === VARIASI EJAAN ===
    'resto': 'restoran', 
    'rm': 'rumah makan',
    'dimsum': 'dim sum', 
    'dumpling': 'dim sum',
    'jus': 'juice', 
    'es krim': 'ice cream'
}

# Semantic expansion untuk konteks pencarian (DISESUAIKAN DENGAN FASILITAS & SUASANA AKTUAL)
SEMANTIC_EXPANSION = {
    # === AKTIVITAS (Berdasarkan tipe pengunjung & fasilitas) ===
    'nugas': 'wifi stopkontak colokan tenang nyaman kerja laptop',
    'kerja': 'wifi stopkontak tenang nyaman laptop',
    'meeting': 'wifi tenang nyaman',
    
    # === SUASANA (Berdasarkan kolom suasana dataset) ===
    'date': 'romantis nyaman santai',
    'pacaran': 'romantis nyaman santai',
    'nongkrong': 'santai nyaman trendi',
    'santai': 'nyaman tenang',
    
    # === TIPE PENGUNJUNG (Berdasarkan kolom tipe_pengunjung) ===
    'keluarga': 'keluarga anak berkelompok parkiran',
    'rombongan': 'berkelompok parkiran',
    'mahasiswa': 'mahasiswa wifi santai trendi',
    
    # === HARGA ===
    'murah': 'terjangkau ekonomis hemat',
    'mahal': 'premium mewah kelas atas',
    
    # === POPULARITAS ===
    'enak': 'recommended populer',
    'hits': 'populer trendi',
    'viral': 'populer trendi'
}

# Location expansion untuk boost lokasi (SEMUA LOKASI DI DATASET - 38 AREA, 138 SUB-LOKASI)
LOCATION_EXPANSION = {
    # Kecamatan & Area Utama
    'andir': ['andir', 'ciroyom', 'dungus', 'cariang', 'maleber', 'garuda'],
    'antapani': ['antapani', 'cisaranten', 'kulon', 'wetan', 'kidul'],
    'arcamanik': ['arcamanik', 'cisaranten', 'sukamiskin'],
    'astanaanyar': ['astanaanyar', 'karanganyar'],
    'babakan': ['babakan', 'ciparay', 'sukahaji'],
    'bandungkidul': ['kidul', 'batununggal', 'kujangsari'],
    'bandungkulon': ['kulon', 'cigondewah', 'cijerah', 'muncang'],
    'bandungwetan': ['wetan', 'citarum', 'tamansari', 'cihapit'],
    'batununggal': ['batununggal', 'kiaracondong', 'binong', 'cibangkong'],
    'bojongloa': ['bojongloa', 'kaler', 'kidul', 'jamika', 'suka', 'asih', 'babakan'],
    'buahbatu': ['buahbatu', 'kopo'],
    'cibeunying': ['cibeunying', 'kaler', 'kidul', 'cigadung', 'sukagalih', 'sukapada'],
    'cibiru': ['cibiru', 'pasir', 'hilir', 'wetan'],
    'cicendo': ['cicendo', 'arjuna', 'husen', 'sukaraja'],
    'cidadap': ['cidadap', 'hegarmanah', 'lebakgede', 'ciumbuleuit'],
    'cinambo': ['cinambo', 'babakan', 'cisaranten', 'pakemitan'],
    'coblong': ['coblong', 'cipaganti', 'dago', 'lebakgede', 'sekeloa'],
    'gedebage': ['gedebage', 'cisaranten', 'kidul', 'rancabolang'],
    'kiaracondong': ['kiaracondong', 'cicaheum', 'kebon'],
    'lengkong': ['lengkong', 'burangrang', 'cijagra', 'cikawao', 'malabar', 'paledang', 'turangga'],
    'mandalajati': ['karang', 'pasir'],
    'panyileukan': ['panyileukan', 'cipadung', 'kulon', 'wetan', 'mekar'],
    'rancasari': ['mekar'],
    'regol': ['regol', 'ancol', 'balonggede', 'ciateul', 'cigereleng', 'pasirluyu', 'pungkur'],
    'sukajadi': ['sukajadi', 'pasteur', 'cipedes', 'sukabungah', 'sukagalih', 'sukawarna'],
    'sukasari': ['sukasari', 'gegerkalong', 'isola', 'sarijadi', 'sukarasa'],
    'sumurbandung': ['sumur', 'braga', 'kebon', 'babakan'],
    'ujungberung': ['pasir', 'endah', 'wangi'],
    
    # Jalan Utama & Area Populer (LEBIH PRESISI)
    'braga': ['braga', 'sumur'],
    'dago': ['dago', 'juanda', 'coblong', 'dipatiukur'],
    'cihampelas': ['cihampelas', 'cipaganti'],
    'merdeka': ['merdeka', 'asia', 'afrika'],
    'martadinata': ['martadinata', 'riau'],
    'pasteur': ['pasteur', 'sukajadi'],
    'setiabudi': ['setiabudi', 'cidadap'],
    'siliwangi': ['siliwangi', 'tamansari'],
    'soekarno': ['soekarno', 'hatta'],
    'gatot': ['gatot', 'subroto']
}

# Whitelist kata umum yang tidak perlu dikoreksi (HANYA YANG ADA DI DATASET + USER INTENT)
COMMON_WORDS = {
    # User Actions
    'cari', 'mencari', 'temukan', 'rekomendasi', 'saran', 'info',
    
    # Tempat (yang ada di dataset)
    'warung', 'rumah', 'makan', 'minum', 'tempat', 'resto', 'restoran', 
    'kafe', 'cafe', 'kedai',
    
    # Deskripsi (yang ada di dataset)
    'enak', 'lezat', 'murah', 'mahal', 'bagus', 'favorit', 'terbaik',
    
    # User Intent (tidak di dataset tapi valid untuk query)
    'date', 'nugas', 'kerja', 'meeting', 'pacaran', 'keluarga',
    'pagi', 'siang', 'sore', 'malam', 'lunch',
    
    # Stopwords
    'di', 'ke', 'dari', 'yang', 'dan', 'dengan', 'buat', 'untuk', 'sama',
    
    # Rasa
    'pedas', 'manis', 'asin', 'gurih', 'segar', 'panas', 'dingin'
}

# Concept mapping untuk menghindari over-boosting (TOP 10 MENU POPULER)
CONCEPT_MAP = {
    # Top Menu Items
    'nasi': 'RICE', 'rice': 'RICE',
    'goreng': 'FRIED', 'fried': 'FRIED',
    'ayam': 'CHICKEN', 'chicken': 'CHICKEN',
    'kopi': 'COFFEE', 'coffee': 'COFFEE', 'latte': 'COFFEE',
    'beef': 'BEEF', 'sapi': 'BEEF', 'steak': 'BEEF',
    'bakar': 'GRILLED', 'grilled': 'GRILLED',
    'ramen': 'RAMEN', 'mie': 'RAMEN',
    'sate': 'SATAY', 'satay': 'SATE',
    'katsu': 'KATSU',
    'pizza': 'PIZZA',
    
    # Locations
    'braga': 'LOC_BRAGA', 'sukajadi': 'LOC_SUKAJADI', 'dago': 'LOC_DAGO',
    
    # Price
    'murah': 'PRICE_LOW', 'terjangkau': 'PRICE_LOW'
}

# Simple synonyms untuk content boosting (TOP MENU DENGAN VARIASI)
SIMPLE_SYNONYMS = {
    'nasi': ['rice', 'goreng'],
    'kopi': ['coffee', 'latte', 'cafe', 'kafe', 'koffie'],
    'cafe': ['kopi', 'coffee', 'kafe', 'koffie'],
    'chicken': ['ayam'],
    'beef': ['sapi', 'steak'],
    'ramen': ['mie'],
    'bakar': ['grilled', 'panggang']
}


# ============================================================================
# CHATBOT ENGINE CLASS
# ============================================================================

class ChatbotEngine:
    """
    Mesin utama chatbot yang mengatur logika rekomendasi.
    Menggunakan metode TF-IDF Vectorization untuk mengubah teks menjadi angka,
    dan Cosine Similarity untuk mengukur kemiripan antar teks.
    """
    
    def __init__(self, csv_path):
        """Inisialisasi chatbot dengan memuat data dan membuat TF-IDF matrix"""
        self.df = self._load_dataset(csv_path)
        self.preprocessor = self._initialize_preprocessor()
        self._preprocess_dataset()
        self._build_vocabulary()
        self._create_tfidf_matrix()
        
        print(f"[SUCCESS] Chatbot Engine berhasil dimuat!")
        print(f"[INFO] Total UMKM: {len(self.df)}")
        print(f"[INFO] TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
    
    def _load_dataset(self, csv_path):
        """Memuat dataset dari CSV"""
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='ISO-8859-1')
            
            if df.empty:
                raise ValueError("Dataset kosong! Tidak ada data untuk diproses.")
            
            # Isi nilai kosong di kolom penting
            df['metadata_tfidf'] = df['metadata_tfidf'].fillna('')
            df['deskripsi'] = df['deskripsi'].fillna('')
            
            if df['metadata_tfidf'].str.strip().eq('').all():
                raise ValueError("[ERROR] Kolom metadata_tfidf kosong! Periksa dataset Anda.")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def _initialize_preprocessor(self):
        """Inisialisasi text preprocessor"""
        try:
            return TextPreprocessor()
        except Exception as e:
            raise Exception(f"Error inisialisasi preprocessor: {str(e)}")
    
    def _preprocess_dataset(self):
        """Preprocessing data UMKM"""
        try:
            # Cek apakah sudah ada kolom processed (optimized dataset)
            bypass_processing = 'metadata_tfidf_processed' in self.df.columns
            
            if bypass_processing:
                self.df['metadata_tfidf_processed'] = self.df['metadata_tfidf_processed'].fillna('')
                print("[SUCCESS] Dataset teroptimasi ditemukan! Lewati stemming manual.")
            else:
                print("[INFO] Dataset belum teroptimasi. Melakukan preprocessing awal...")
                self.df['metadata_tfidf_original'] = self.df['metadata_tfidf'].copy()
                self.df['metadata_tfidf_processed'] = self.df['metadata_tfidf'].apply(
                    lambda x: self.preprocessor.preprocess(str(x))
                )
                print("[SUCCESS] Preprocessing dataset selesai!")
                
        except Exception as e:
            raise Exception(f"Error preprocessing dataset: {str(e)}")
    
    def _build_vocabulary(self):
        """Membangun vocabulary untuk koreksi typo"""
        try:
            print("[INFO] Membuat TF-IDF matrix...")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['metadata_tfidf_processed'])
            
            if self.tfidf_matrix.shape[0] == 0:
                raise ValueError("TF-IDF matrix kosong! Periksa data metadata_tfidf.")
            
        except Exception as e:
            raise Exception(f"Error membuat TF-IDF matrix: {str(e)}")
        
        print(f"[SUCCESS] Chatbot Engine berhasil dimuat!")
        print(f"[INFO] Total UMKM: {len(self.df)}")
        print(f"[INFO] TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
    
    def get_recommendations(self, query, price_filter=None, top_n=5):
        """Mendapatkan rekomendasi UMKM berdasarkan query pengguna"""
        
        if not query or not isinstance(query, str):
            raise ValueError("Query harus berupa string yang tidak kosong!")
            
        recommendation_warning = None # Default warning string
        
        query = query.strip()
        if not query:
            raise ValueError("Query tidak boleh kosong atau hanya spasi!")
        
        try:
            # Preprocessing Query: Membersihkan input pengguna (huruf kecil, hapus simbol, stemming)
            processed_query = self.preprocessor.preprocess(query)
            
            # --- PRE-CHECK EXACT MATCH ---
            # Cek apakah raw query adalah nama restoran EXACT match.
            # Ini penting untuk kasus "Ini Itu Cafe" dimana semua katanya terhapus jadi stopword.
            import re
            def normalize_raw(text):
                return str(text).lower().strip().replace("   ", " ").replace("  ", " ")
            
            raw_match_exists = (self.df['nama_rumah_makan'].apply(normalize_raw) == normalize_raw(query)).any()
            
            # VALIDASI TAMBAHAN: Abaikan query 1 huruf KECUALI jika ada exact match
            if not raw_match_exists:
                if len(processed_query.strip()) < 2:
                    print(f"[INFO] Query '{processed_query}' diabaikan karena terlalu pendek.")
                    return pd.DataFrame(), None # Kembalikan DataFrame kosong dan No Warning
                    
                if not processed_query.strip():
                    return pd.DataFrame(), None

            # --- NORMALISASI SINONIM (SEBELUM AUTO-CORRECT) ---
            # Mengubah kata gaul/singkatan menjadi istilah baku di database
            # Contoh: 'cina' -> 'chinese food', 'indo' -> 'masakan indonesia'
            # Mencegah kata seperti 'cina' dikoreksi menjadi 'cinta'
            synonym_map = {
                # Kategori Makanan Dasar
                'makanan': 'masakan', 'makan': 'masakan',
                
                # Tempat Makan (Variasi Nama)
                'kafe': 'kafe/kedai kopi cafe & dessert', 'cafe': 'kafe/kedai kopi cafe & dessert', 
                'kedai kopi': 'kafe/kedai kopi cafe & dessert', 'coffee': 'kafe/kedai kopi cafe & dessert',
                'coffe shop': 'kafe/kedai kopi cafe & dessert', 'kopi': 'kafe/kedai kopi cafe & dessert',
                'resto': 'restoran', 'rm': 'rumah makan',
                'warung': 'rumah makan', 'warteg': 'warung tegal',
                'kedai': 'kafe', 'angkringan': 'kafe',
                
                # Kategori Kuliner Internasional
                'cina': 'chinese food', 'chinese': 'chinese food', 'china': 'chinese food', 'chinesse': 'chinese food',
                'jepang': 'japanese food', 'japan': 'japanese food', 'jpn': 'japanese food',
                'korea': 'korean food', 'korean': 'korean food', 'korsel': 'korean food',
                'barat': 'western food', 'western': 'western food',
                'arab': 'middle eastern', 'timur tengah': 'middle eastern', 'middle east': 'middle eastern',
                'thai': 'thailand', 'thailand': 'thailand', 'tom yum': 'thailand',
                'italia': 'italian', 'italian': 'italian', 'pizza': 'italian', 'pasta': 'italian',
                
                # Kategori Kuliner Lokal
                'indo': 'masakan indonesia', 'indonesia': 'masakan indonesia', 
                'nusantara': 'masakan indonesia', 'lokal': 'masakan indonesia',
                'padang': 'masakan padang', 'minang': 'masakan padang',
                'sunda': 'masakan sunda', 'sundanese': 'masakan sunda',
                'jawa': 'masakan jawa', 'javanese': 'masakan jawa',
                
                # Top Locations
                'braga', 'sukajadi', 'cibeunying', 'coblong', 'lengkong', 
                'citarum', 'antapani', 'cicendo', 'dago', 'juanda', 'gegerkalong',
                
                # Jenis Protein/Daging
                'ayam': 'chicken', 'chicken': 'ayam',
                'sapi': 'beef', 'beef': 'sapi',
                'kambing': 'lamb', 'lamb': 'kambing', 'domba': 'lamb',
                'ikan': 'fish', 'fish': 'ikan',
                'udang': 'shrimp', 'shrimp': 'udang', 'prawn': 'udang',
                
                # Preferensi Diet
                'vegetarian': 'sayur', 'vegan': 'sayur', 'nabati': 'sayur',
                'non halal': 'masakan non halal', 'babi': 'masakan non halal', 'pork': 'masakan non halal',
                'halal': 'halal food',
                
                # Minuman & Dessert
                'ngopi': 'cafe & dessert', 'dessert': 'cafe & dessert',
                'es krim': 'ice cream', 'ice cream': 'es krim',
                'jus': 'juice', 'juice': 'jus', 'smoothie': 'juice',
                
                # Jenis Hidangan
                'nasi goreng': 'fried rice', 'fried rice': 'nasi goreng',
                'mie': 'noodle', 'noodle': 'mie', 'mi': 'mie',
                'bakso': 'meatball', 'meatball': 'bakso',
                'soto': 'soup', 'soup': 'soto', 'sup': 'soto',
                'sate': 'satay', 'satay': 'sate',
                'gado gado': 'salad', 'salad': 'gado gado'
            }
            
            # Gunakan penggantian berbasis token (kata per kata) untuk akurasi
            # Menghindari masalah substring replace (misal "makan" keganti di dalam "memakan")
            current_words = processed_query.split()
            new_words = []
            
            for synonym, replacement in synonym_map.items():
                # Gunakan regex untuk exact word match: \b kata \b
                import re
                pattern = r'\b' + re.escape(synonym) + r'\b'
                processed_query = re.sub(pattern, replacement, processed_query)
            
            # --- KOREKSI TYPO (AUTO-CORRECT) ---
            # Whitelist: Kata-kata umum yang JANGAN dikoreksi meskipun tidak ada di vocabulary dataset
            COMMON_WORDS = {
                'toko', 'warung', 'rumah', 'makan', 'minum', 'tempat', 'resto', 'restoran', 
                'kafe', 'cafe', 'kedai', 'jualan', 'dagang',
                'enak', 'lezat', 'murah', 'mahal', 'bagus', 'keren', 'hits', 'viral',
                'populer', 'favorit', 'terbaik', 'best', 'recommended', 'rekomen',
                'di', 'ke', 'dari', 'yang', 'dan', 'dengan', 'buat', 'untuk', 'sama',
                'date', 'nugas', 'kerja', 'meeting', 'pacaran', 'family', 'keluarga',
                'pagi', 'siang', 'sore', 'malam', 'bukber', 'sarapan', 'dinner', 'lunch',
                'pedas', 'manis', 'asin', 'gurih', 'segar', 'panas', 'dingin'
            }
            
            corrected_words = []
            query_words = processed_query.split()
            was_corrected = False
            
            for word in query_words:
                # Cek 1: Apakah kata ada di vocabulary dataset?
                if word in self.vocabulary:
                    corrected_words.append(word)
                
                # Cek 2: Apakah kata adalah kata umum (Whitelist)?
                elif word in COMMON_WORDS:
                    corrected_words.append(word) # Jangan koreksi, biarkan apa adanya
                
                # Cek 3: Coba koreksi typo dengan threshold ketat (0.90)
                else:
                    # Threshold dinaikkan dari 0.85 ke 0.90 agar tidak "sok tahu"
                    matches = get_close_matches(word, self.vocabulary, n=1, cutoff=0.82)
                    if matches:
                        suggestion = matches[0]
                        corrected_words.append(suggestion)
                        print(f"[INFO] Auto-correct: '{word}' -> '{suggestion}'")
                        was_corrected = True
                    else:
                        corrected_words.append(word)
            
            if was_corrected:
                processed_query = " ".join(corrected_words)
                query = processed_query 
                print(f"[INFO] Corrected Query: {processed_query}")

        except Exception as e:
            raise Exception(f"Error preprocessing query: {str(e)}")
        
        try:
            query_lower = query.lower()
            
            # --- EKSPANSI SEMANTIK ---
            # Menambahkan konteks ke pencarian. Jika user cari "nugas", sistem otomatis
            # menambah kata kunci "wifi", "stopkontak", "nyaman" agar hasil lebih relevan.
            expansion_map = {
                'nugas': 'wifi stopkontak colokan tenang nyaman kerja laptop cafe coffe shop',
                'kerja': 'wifi stopkontak colokan tenang nyaman kerja laptop',
                'meeting': 'ruang privat tenang wifi nyaman',
                'date': 'romantis malam minggu mewah cantik instagramable',
                'pacaran': 'romantis malam minggu mewah cantik',
                'bukber': 'luas rombongan keluarga parkir musholla',
                'reuni': 'luas rombongan keluarga parkir',
                'family': 'keluarga anak ramah kursi bayi luas',
                'sarapan': 'pagi bubur soto kupat nasi kuning',
                'malam': 'malam hari 24 jam',
                'hemat': 'murah terjangkau ekonomis promo',
                'anak kos': 'murah banyak kenyang hemat',
                'sehat': 'sayur salad jus organik vegetarian',
                
                # Ekspansi untuk Query Umum (Adjective)
                'enak': 'recommended populer favorit rating tinggi lezat nikmat',
                'bagus': 'recommended populer favorit rating tinggi nyaman instagramable',
                'best': 'recommended populer favorit rating tinggi terbaik',
                'murah': 'terjangkau ekonomis hemat budget friendly murah meriah',
                'mahal': 'premium mewah fancy high class eksklusif',
                
                # Ekspansi untuk Tren
                'hits': 'populer viral trending favorit kekinian',
                'viral': 'populer hits trending favorit ramai',
                'populer': 'hits viral trending favorit ramai',
                'terkenal': 'legendaris hits viral populer'
            }
            
            expanded_terms = []
            for term, keywords in expansion_map.items():
                if term in query_lower:
                    expanded_terms.append(keywords)
                    print(f"[INFO] Semantic Expansion: '{term}' -> '{keywords}'")
            
            if expanded_terms:
                query_expanded = processed_query + " " + " ".join(expanded_terms)
                query_vector = self.vectorizer.transform([query_expanded])
            else:
                query_vector = self.vectorizer.transform([processed_query])

            query_normalized = query_lower
            
            # Normalisasi Sinonim (Late Stage) agar matching kategori berfungsi
            synonym_map_late = {
                # Kategori Makanan Dasar
                'makanan': 'masakan', 'makan': 'masakan',
                
                # Tempat Makan (Variasi Nama)
                'kafe': 'cafe', 
                'kedai kopi': 'cafe', 'coffee': 'cafe',
                'coffe shop': 'cafe', 'kopi': 'cafe', 'ngopi': 'cafe',
                'resto': 'restoran', 'rm': 'rumah makan',
                'warung': 'rumah makan', 'warteg': 'warung tegal',
                'kedai': 'kafe', 'angkringan': 'kafe',
                
                # Kategori Kuliner Internasional
                'cina': 'chinese food', 'chinese': 'chinese food', 'china': 'chinese food', 'chinesse': 'chinese food',
                'jepang': 'japanese food', 'japan': 'japanese food', 'jpn': 'japanese food',
                'korea': 'korean food', 'korean': 'korean food', 'korsel': 'korean food',
                'barat': 'western food', 'western': 'western food',
                'arab': 'middle eastern', 'timur tengah': 'middle eastern', 'middle east': 'middle eastern',
                'thai': 'thailand', 'tom yum': 'thailand',
                'italia': 'italian', 'pizza': 'italian', 'pasta': 'italian',
                
                # Kategori Kuliner Lokal
                'indo': 'masakan indonesia', 'indonesia': 'masakan indonesia', 
                'nusantara': 'masakan indonesia', 'lokal': 'masakan indonesia',
                'padang': 'masakan padang', 'minang': 'masakan padang',
                'sunda': 'masakan sunda', 'sundanese': 'masakan sunda',
                'jawa': 'masakan jawa', 'javanese': 'masakan jawa',
                
                # Jenis Makanan Spesifik
                'seafood': 'makanan laut', 'makanan laut': 'seafood',
                'bakery': 'toko roti', 'pastry': 'toko roti', 'roti': 'toko roti',
                'dimsum': 'dim sum', 'dumpling': 'dim sum',
                'bbq': 'barbeque', 'panggang': 'barbeque', 'bakar': 'barbeque',
                'hotpot': 'hot pot', 'shabu': 'shabu-shabu',
                
                # Jenis Protein/Daging
                'ayam': 'chicken', 'chicken': 'ayam',
                'sapi': 'beef', 'beef': 'sapi',
                'kambing': 'lamb', 'lamb': 'kambing', 'domba': 'lamb',
                'ikan': 'fish', 'fish': 'ikan',
                'udang': 'shrimp', 'shrimp': 'udang', 'prawn': 'udang',
                
                # Preferensi Diet
                'vegetarian': 'sayur', 'vegan': 'sayur', 'nabati': 'sayur',
                'non halal': 'masakan non halal', 'babi': 'masakan non halal', 'pork': 'masakan non halal',
                'halal': 'halal food',
                
                # Minuman & Dessert
                'ngopi': 'cafe & dessert', 'dessert': 'cafe & dessert',
                'es krim': 'ice cream', 'ice cream': 'es krim',
                'jus': 'juice', 'juice': 'jus', 'smoothie': 'juice',
                
                # Jenis Hidangan
                'nasi goreng': 'fried rice', 'fried rice': 'nasi goreng',
                'mie': 'noodle', 'noodle': 'mie', 'mi': 'mie',
                'bakso': 'meatball', 'meatball': 'bakso',
                'soto': 'soup', 'soup': 'soto', 'sup': 'soto',
                'sate': 'satay', 'satay': 'sate',
                'gado gado': 'salad', 'salad': 'gado gado'
            }
            for synonym, replacement in synonym_map_late.items():
                query_normalized = query_normalized.replace(synonym, replacement)

            # --- PERHITUNGAN KECOCOKAN (COSINE SIMILARITY) ---
            # Inti algoritma: Mengukur sudut antar vektor query dan data.
            # Nilai 1.0 berarti persis sama, 0.0 berarti tidak ada kemiripan sama sekali.
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # LANGKAH 1: Identifikasi target pencarian (Kategori/Tipe Pengunjung)
            all_categories = set(str(cat).lower() for cat in self.df['kategori'].dropna().unique())
            all_tipe_pengunjung = set()
            for val in self.df['tipe_pengunjung'].dropna():
                for item in str(val).split(','):
                    cleaned = item.strip().lower()
                    if len(cleaned) >= 4:
                        all_tipe_pengunjung.add(cleaned)
            
            # LANGKAH 2: Pencocokan dengan Query
            matched_category = None
            matched_tipe_pengunjung = None
            
            sorted_categories = sorted(all_categories, key=len, reverse=True)
            for category in sorted_categories:
                if category in query_normalized:
                    matched_category = category
                    print(f"[DEBUG] MATCHED CATEGORY: '{category}'")
                    break
            
            if not matched_category:
                sorted_tipe = sorted(all_tipe_pengunjung, key=len, reverse=True)
                for tipe in sorted_tipe:
                    if tipe in query_normalized:
                        matched_tipe_pengunjung = tipe
                        print(f"[DEBUG] MATCHED TIPE: '{tipe}'")
                        break
            
            # LANGKAH 3: Deteksi Filter Tambahan (Lokasi, Harga, Suasana, Fasilitas)
            additional_filters = set()
            
            try:
                alamat_words = ' '.join(self.df['alamat'].dropna().astype(str)).lower()
                location_keywords = set([w for w in alamat_words.split() if len(w) >= 4 and w.isalpha()])
                
                # [CRITICAL FIX] Hapus kata umum kuliner dari deteksi lokasi
                # Agar "kopi" tidak dianggap sebagai filter lokasi yang menyebabkan penalti massal
                ignore_location_terms = {
                    'kopi', 'cafe', 'kafe', 'resto', 'warung', 'makan', 'minum',
                    'jalan', 'kota', 'bandung', 'kecamatan', 'kelurahan', 'nomor',
                    'utara', 'selatan', 'barat', 'timur', 'tengah', 'jawa',
                    'coffee', 'shop', 'store', 'food', 'beverage',
                    'bakso', 'mie', 'nasi', 'soto', 'ayam', 'bebek', 'sapi'
                }
                location_keywords = location_keywords - ignore_location_terms
                
                additional_filters.update(location_keywords)
            except: pass
            
            price_keywords = {'murah', 'mahal', 'sedang', 'terjangkau', 'hemat', 'premium', 'mewah', 'budget', 'promo'}
            additional_filters.update(price_keywords)
            
            try:
                suasana_words = ' '.join(self.df['suasana'].dropna().astype(str)).lower()
                additional_filters.update([w.strip() for w in suasana_words.split(',') if len(w.strip()) >= 4])
            except: pass
            
            try:
                fasilitas_words = ' '.join(self.df['fasilitas'].dropna().astype(str)).lower()
                additional_filters.update([w.strip() for w in fasilitas_words.split(',') if len(w.strip()) >= 4])
            except: pass
            
            additional_filters = {w.strip() for w in additional_filters if len(w.strip()) >= 3 and w.strip() not in {'dan', 'yang', 'untuk', 'dari', 'dengan'}}
            
            final_active_filters = set()
            detected_raw = [kw for kw in additional_filters if kw in query_lower]
            
            for flt in detected_raw:
                is_conflict = False
                if matched_category and flt in matched_category: is_conflict = True
                if matched_tipe_pengunjung and flt in matched_tipe_pengunjung: is_conflict = True
                if not is_conflict: final_active_filters.add(flt)

            has_additional_filter = len(final_active_filters) > 0
            
            # --- LOGIKA REKOMENDASI HIBRIDA (Hybrid Recommendation Logic) ---
            
            # MODE KETAT (Strict Mode):
            # Aktif jika user menyebutkan Kategori atau Tipe Pengunjung secara spesifik.
            # Sistem akan membuang semua hasil yang bukan dari kategori tersebut (-999 poin).
            strict_mode_activated = False
            
            if matched_category and not has_additional_filter:
                print(f"[STRICT MODE] Category: '{matched_category}'")
                category_mask = self.df['kategori'].astype(str).str.lower() == matched_category
                similarity_scores[~category_mask] = -999
                similarity_scores[category_mask] += 1.0
                strict_mode_activated = True
                
            elif matched_tipe_pengunjung and not has_additional_filter:
                print(f"[STRICT MODE] Tipe: '{matched_tipe_pengunjung}'")
                tipe_mask = self.df['tipe_pengunjung'].astype(str).str.lower().str.contains(matched_tipe_pengunjung, na=False, regex=False)
                similarity_scores[~tipe_mask] = -999
                similarity_scores[tipe_mask] += 1.0
                strict_mode_activated = True
            
            else:
                # Mode Fleksibel: Menggunakan TF-IDF jika tidak ada match pasti
                if matched_category or matched_tipe_pengunjung:
                    if matched_category:
                         # [SMART CATEGORY MATCHING]
                         if any(x in matched_category for x in ['kopi', 'cafe', 'kafe']):
                             category_mask = self.df['kategori'].astype(str).str.lower().str.contains('kopi|cafe|kafe|coffee', na=False, regex=True)
                         else:
                             category_mask = self.df['kategori'].astype(str).str.lower() == matched_category
                         similarity_scores[category_mask] += 20.0 
                    
                    if matched_tipe_pengunjung:
                        tipe_mask = self.df['tipe_pengunjung'].astype(str).str.lower().str.contains(matched_tipe_pengunjung, na=False, regex=False)
                        similarity_scores[tipe_mask] += 5.0
                
                # [MANUAL CAFE INTENT]
                # Jika kategori tidak terdeteksi via nama, tapi ada kata 'cafe' (hasil normalisasi kopi/kafe),
                # paksa boost kategori cafe.
                elif 'cafe' in query_normalized:
                     print("[INFO] Cafe intent detected (manual fallback)")
                     category_mask = self.df['kategori'].astype(str).str.lower().str.contains('kopi|cafe|kafe|coffee', na=False, regex=True)
                     similarity_scores[category_mask] += 20.0
                    
                    # Boosting Lokasi: Memberikan prioritas pada lokasi yang cocok
            # [PERBAIKAN LOKASI] Definisi sinonim dipindahkan ke sini agar bisa dipakai untuk filter lokasi
            # Contoh: User cari "Dago", sistem juga harus boost alamat "Jl. Ir. H. Juanda"
            
            # Mapping manual untuk expansion, karena boost_synonyms di atas formatnya Concept Mapping
            location_expansion = {
                'dago': ['juanda', 'bukit pakar', 'dago'],
                'bandung': ['bdg', 'paris van java'],
                'braga': ['sumur bandung', 'asia afrika']
            }

            # Boosting Lokasi: Memberikan prioritas pada lokasi yang cocok
            if len(final_active_filters) > 0:
                for flt in final_active_filters:
                    # Ambil variasi kata lokasi (misal: dago -> [dago, juanda])
                    search_terms = location_expansion.get(flt, [flt])
                    
                    # Buat mask gabungan: Benar jika salah satu kata kunci lokasi ditemukan di alamat
                    addr_mask = pd.Series([False] * len(self.df), index=self.df.index)
                    for term in search_terms:
                        term_mask = self.df['alamat'].astype(str).str.lower().str.contains(term, na=False)
                        addr_mask = addr_mask | term_mask
                    
                    # Boost Match (Stronger: +15.0) -> Prioritas Mutlak Lokasi
                    similarity_scores[addr_mask] += 15.0
                    
                    # Penalty Mismatch (STRICT Location Filter)
                    if addr_mask.sum() > 0: 
                        similarity_scores[~addr_mask] -= 50.0
                        print(f"[DEBUG] Applied Location Boost (+15.0) & Penalty (-50.0) for '{flt}' (Expanded: {search_terms})")
            # --- CONTENT RELEVANCE BOOST ---
            # Prioritizing content (Name/Menu) over price filters
            price_terms = {'murah', 'mahal', 'sedang', 'terjangkau', 'hemat', 'premium', 'mewah', 'budget', 'promo', 'murmer'}
            common_stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'atau', 'ini', 'itu', 'makan', 'minum', 'tempat', 'warung', 'resto'}
            ignore_terms = price_terms | common_stopwords
            
            core_words = [w for w in query_lower.split() if w not in ignore_terms and len(w) > 2]
            
            if core_words:
                # [BARU] Perluasan sinonim khusus untuk boosting
                # Agar pencarian "kopi" juga men-boost "koffie", "coffee", "cafe"
                
                # [REVISI BOOSTING] Concept Mapping untuk mencegah over-boosting
                # boost_synonyms sudah didefinisikan sebelumnya di blok Location Boost
                
                # Definisi ulang Concept Map (untuk grouping skor)
                concept_map = {
                    'kopi': 'COFFEE', 'coffee': 'COFFEE', 'cafe': 'COFFEE', 'kafe': 'COFFEE', 'koffie': 'COFFEE', 'kopii': 'COFFEE', 'bean': 'COFFEE',
                    'dago': 'LOCATION_DAGO', 'juanda': 'LOCATION_DAGO', 'pakar': 'LOCATION_DAGO',
                    'murah': 'PRICE_LOW', 'terjangkau': 'PRICE_LOW', 'hemat': 'PRICE_LOW'
                }
                
                processed_concepts = set()
                
                # Gabungkan core_words dengan synonym expansion
                # Kita perlu synonym dictionary sederhana untuk expansion ini (word -> [list of synonyms])
                # Gunakan ulang location_expansion atau buat simple map
                simple_synonyms = {
                     'kopi': ['koffie', 'coffee', 'cafe', 'kafe', 'bean'],
                     'cafe': ['coffee', 'kopi'],
                     'dago': ['juanda', 'bukit pakar']
                }

                all_search_terms = set(core_words)
                for word in core_words:
                    if word in simple_synonyms:
                        all_search_terms.update(simple_synonyms[word])
                
                for word in all_search_terms:
                    # Tentukan konsep kata ini
                    concept = concept_map.get(word, word) # Jika tidak ada di map, gunakan kata itu sendiri sebagai konsep unik
                    
                    if concept in processed_concepts:
                        continue # Skip jika konsep ini sudah di-boost sebelumnya
                    
                    # Boost +3.0 per CONCEPT found in Name OR Menu
                    import re
                    safe_word = re.escape(word)
                    
                    name_mask = self.df['nama_rumah_makan'].astype(str).str.lower().str.contains(safe_word, na=False)
                    menu_mask = self.df['menu'].astype(str).str.lower().str.contains(safe_word, na=False)
                    
                    anywhere_mask = name_mask | menu_mask
                    
                    if anywhere_mask.any():
                        similarity_scores[anywhere_mask] += 3.0
                        processed_concepts.add(concept) # Tandai konsep sudah diproses
                        # print(f"[DEBUG] Applied Boost (+3.0) for concept '{concept}' (trigger: '{word}')")
                
                # EXACT PHRASE MATCH BOOST (+5.0)
                # Ensure specific menu items (e.g., "nasi goreng pedas") rank highest
                if len(core_words) >= 2:
                    phrase = " ".join(core_words)
                    safe_phrase = re.escape(phrase)
                    
                    phrase_name_mask = self.df['nama_rumah_makan'].astype(str).str.lower().str.contains(safe_phrase, na=False)
                    phrase_menu_mask = self.df['menu'].astype(str).str.lower().str.contains(safe_phrase, na=False)
                    
                    phrase_mask = phrase_name_mask | phrase_menu_mask
                    similarity_scores[phrase_mask] += 5.0

            # --- BOOSTING HARGA ---
            # Lower boost factor used as tie-breaker (0.5) -> UPDATED to 5.0 for stronger sorting
            BOOST_FACTOR = 5.0
            
            is_murah = any(k in query_lower for k in ['murah', 'terjangkau', 'hemat', 'low budget'])
            is_sedang = any(k in query_lower for k in ['sedang', 'standar', 'menengah', 'reasonable'])
            is_mahal = any(k in query_lower for k in ['mahal', 'premium', 'mewah', 'fancy'])
            
            if price_filter and price_filter != "Semua":
                if price_filter == "Murah": is_murah, is_sedang, is_mahal = True, False, False
                elif price_filter == "Menengah": is_murah, is_sedang, is_mahal = False, True, False
                elif price_filter == "Mahal": is_murah, is_sedang, is_mahal = False, False, True
            
            # Apply price boosting only if relevant content found
            relevant_mask = similarity_scores > -500 # Apply to all not-penalized items
            
            if is_murah:
                mask = self.df['kategori_harga'].astype(str).str.contains('Murah', case=False, na=False)
                similarity_scores[mask & relevant_mask] += BOOST_FACTOR 
            elif is_mahal:
                mask = self.df['kategori_harga'].astype(str).str.contains('Mahal', case=False, na=False)
                similarity_scores[mask & relevant_mask] += BOOST_FACTOR
            elif is_sedang:
                mask = self.df['kategori_harga'].astype(str).str.contains('Sedang', case=False, na=False)
                similarity_scores[mask & relevant_mask] += BOOST_FACTOR
            
        except Exception as e:
            raise Exception(f"Error menghitung similarity: {str(e)}")
        
        # --- EXACT/FUZZY NAME MATCH BOOST (PRIORITY TERTINGGI) ---
        # PINDAHKAN KE LUAR try-except agar tidak ter-interrupt
        # Gunakan vectorized operation untuk performa
        try:
            from rapidfuzz import fuzz
            import re
            
            # Fungsi normalisasi teks
            def normalize_text(text):
                text = str(text).lower().strip()
                text = re.sub(r'\s+', ' ', text)
                text = text.replace("'", "'").replace("`", "'")
                return text
            
            query_clean = normalize_text(query)
            query_len = len(query_clean)
            
            # Vectorized exact match check (JAUH LEBIH CEPAT)
            df_names_normalized = self.df['nama_rumah_makan'].apply(normalize_text)
            exact_matches = df_names_normalized == query_clean
            
            if exact_matches.any():
                # Ada exact match - berikan boost SANGAT BESAR untuk override penalty
                # CRITICAL: Boost harus > 999 untuk mengalahkan strict mode penalty (-999)
                exact_matches_array = exact_matches.values
                
                # BOOST 2000.0 untuk memastikan exact match SELALU menang
                similarity_scores[exact_matches_array] += 2000.0
                
                matched_names = self.df.loc[exact_matches, 'nama_rumah_makan'].tolist()
                for name in matched_names:
                    print(f"[EXACT MATCH 100%] '{name}' matched query '{query}'")
            
            # Fuzzy match hanya jika tidak ada exact match DAN query cukup panjang
            elif query_len >= 8:
                # Hanya cek top 100 berdasarkan TF-IDF untuk efisiensi
                top_indices = similarity_scores.argsort()[-100:][::-1]
                
                for idx in top_indices:
                    nama_resto = normalize_text(self.df.iloc[idx]['nama_rumah_makan'])
                    
                    # Hitung similarity
                    similarity_ratio = fuzz.ratio(query_clean, nama_resto)
                    partial_ratio = fuzz.partial_ratio(query_clean, nama_resto)
                    best_ratio = max(similarity_ratio, partial_ratio)
                    
                    if best_ratio >= 95.0:
                        similarity_scores[idx] += 8.0
                        print(f"[NEAR MATCH {best_ratio:.1f}%] '{self.df.iloc[idx]['nama_rumah_makan']}' matched query '{query}'")
                        break  # Hanya ambil 1 best fuzzy match
                    elif best_ratio >= 90.0 and query_len >= 12:
                        similarity_scores[idx] += 5.0
                        print(f"[GOOD MATCH {best_ratio:.1f}%] '{self.df.iloc[idx]['nama_rumah_makan']}' matched query '{query}'")
                        break
                        
        except Exception as e:
            print(f"[WARNING] Fuzzy matching error: {str(e)}")
            # Lanjutkan tanpa fuzzy matching jika error
            pass
        

        
        try:
            result_df = self.df.copy()
            result_df['similarity_score'] = similarity_scores
            
            top_recommendations = result_df.nlargest(top_n, 'similarity_score')
            top_recommendations = top_recommendations[top_recommendations['similarity_score'] > 0]
            
            # Fallback search
            if top_recommendations.empty:
                print(f"[INFO] Fallback search for: {query}")
                keyword = query.lower()
                
                if len(keyword) < 3:
                     mask = pd.Series([False] * len(self.df))
                else:
                    mask = self.df['metadata_tfidf'].str.lower().str.contains(keyword, na=False)
                
                if mask.any():
                    fallback_df = self.df[mask].copy()
                    fallback_df['similarity_score'] = 0.5
                    top_recommendations = fallback_df.head(top_n)
                    print(f"[SUCCESS] Fallback found {len(top_recommendations)} results.")
            
            # --- INTELLIGENT WARNING SYSTEM (REVISED) ---
            target_price = None
            if is_murah: target_price = "Murah"
            elif is_mahal: target_price = "Mahal"
            elif is_sedang: target_price = "Sedang"

            if target_price and not top_recommendations.empty:
                if 'kategori_harga' in top_recommendations.columns:
                    # Intelligent Warning System based on Relevance Score
                    # Filter out matches that are significantly less relevant than the top result
                    
                    max_score = top_recommendations['similarity_score'].max()
                    # Threshold: Score must be within 3.0 points of the top score (Relaxed)
                    relevance_threshold = max(0, max_score - 3.0) 
                    
                    # Check Top 5
                    checked_recs = top_recommendations.head(5)
                    relevant_in_top5 = checked_recs[checked_recs['similarity_score'] >= relevance_threshold]
                    
                    matched_count = relevant_in_top5['kategori_harga'].astype(str).str.contains(target_price, case=False, na=False).sum()
                    
                    # Check Top 50 for available data
                    relevant_in_all = top_recommendations[top_recommendations['similarity_score'] >= relevance_threshold]
                    matched_all_count = relevant_in_all['kategori_harga'].astype(str).str.contains(target_price, case=False, na=False).sum()

                    # [REVISI INTELLIGENT] Warning hanya muncul jika Benar-benar TIDAK ADA di Top 5
                    # User complain jika ada 1 hasil tapi dibilang terbatas.
                    if matched_count == 0:
                        category_msg = f" Kategori {matched_category.title()}" if matched_category else ""
                        recommendation_warning = f"Maaf, kami tidak menemukan rekomendasi yang pas untuk '{query}' dengan harga '{target_price}' di top 5 hasi. Berikut adalah rekomendasi terbaik yang kami temukan."
                        print(f"[DEBUG] Warning set: No Matches in Top 5 (Top5={matched_count}, Total={matched_all_count})")
            
            return top_recommendations, recommendation_warning
            
        except Exception as e:
            raise Exception(f"Error memproses hasil rekomendasi: {str(e)}")
    
    def get_statistics(self):
        """Mendapatkan statistik dataset"""
        try:
            stats = {
                'total_umkm': len(self.df),
                'total_kategori': self.df['kategori'].nunique(),
                'kategori_terbanyak': self.df['kategori'].value_counts().head(5).to_dict(),
                'harga_distribution': self.df['kategori_harga'].value_counts().to_dict()
            }
            return stats
        except Exception as e:
            raise Exception(f"Error mendapatkan statistik: {str(e)}")
    
    def search_by_category(self, category, top_n=10):
        """Mencari UMKM berdasarkan kategori spesifik"""
        if not category or not isinstance(category, str):
            raise ValueError("Kategori harus berupa string yang tidak kosong!")
        
        try:
            filtered = self.df[self.df['kategori'].str.contains(category, case=False, na=False)]
            return filtered.head(top_n)
        except Exception as e:
            raise Exception(f"Error mencari berdasarkan kategori: {str(e)}")
    
    def search_by_price(self, price_category):
        """Mencari UMKM berdasarkan kategori harga"""
        if not price_category or not isinstance(price_category, str):
            raise ValueError("Kategori harga harus berupa string yang tidak kosong!")
        
        try:
            filtered = self.df[self.df['kategori_harga'].str.contains(price_category, case=False, na=False)]
            return filtered
        except Exception as e:
            raise Exception(f"Error mencari berdasarkan harga: {str(e)}")
    
    def search_by_location(self, location):
        """Mencari UMKM berdasarkan lokasi"""
        if not location or not isinstance(location, str):
            raise ValueError("Lokasi harus berupa string yang tidak kosong!")
        
        try:
            filtered = self.df[self.df['alamat'].str.contains(location, case=False, na=False)]
            return filtered
        except Exception as e:
            raise Exception(f"Error mencari berdasarkan lokasi: {str(e)}")