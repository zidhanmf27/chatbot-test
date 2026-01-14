# ============================================================================
# MESIN CHATBOT - LOGIKA REKOMENDASI UTAMA
# ============================================================================
# Menangani pemuatan data, perhitungan TF-IDF, Cosine Similarity,
# dan logika rekomendasi (Strict/Flexible).
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
# KELAS MESIN CHATBOT
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
            print("[INFO] Membangun vocabulary untuk koreksi otomatis...")
            # Menggunakan metadata_tfidf (kata asli/unstemmed) agar koreksi typo lebih akurat
            # Contoh: "makanann" akan lebih mudah dideteksi sebagai "makanan" daripada "makan"
            all_text = " ".join(self.df['metadata_tfidf'].astype(str).tolist())
            self.vocabulary = set(all_text.lower().split())
            
            # Membangun Priority Vocabulary (Kategori & Kata Kunci Penting)
            self.priority_vocabulary = set()
            
            # 1. Masukkan semua kategori unik
            if 'kategori' in self.df.columns:
                categories = self.df['kategori'].dropna().unique()
                for cat in categories:
                    clean_cat = str(cat).lower().replace('/', ' ').replace('&', ' ')
                    self.priority_vocabulary.update(clean_cat.split())
            
            # 2. Masukkan target dari SYNONYM_MAP
            for val in SYNONYM_MAP.values():
                self.priority_vocabulary.update(val.split())
            
            # 3. Analisis Otomatis Dataset (Top Frequent Words)
            # Mengekstrak kata populer dari Menu, Alamat, dan Fasilitas
            # agar kata-kata ini diprioritaskan saat auto-correct
            print("[INFO] Menganalisis dataset untuk mengisi Priority Vocabulary...")
            
            def extract_top_keywords(column, n=50):
                """Fungsi pembantu untuk mengambil n kata teratas dari kolom"""
                if column not in self.df.columns:
                    return []
                
                # Gabungkan seluruh teks di kolom
                text = ' '.join(self.df[column].dropna().astype(str).tolist()).lower()
                # Bersihkan karakter non-huruf
                text = re.sub(r'[^a-z\s]', ' ', text)
                tokens = text.split()
                
                # Stopwords sederhana untuk filtering
                stopwords = {
                    'dan', 'yg', 'di', 'ke', 'dari', 'yang', 'dengan', 'untuk', 'utk', 
                    'jl', 'jalan', 'no', 'kota', 'bandung', 'jawa', 'barat', 
                    'kecamatan', 'kelurahan', 'rt', 'rw', 'nomor', 'blok', 'lantai',
                    'memiliki', 'tersedia', 'area', 'ada'
                }
                
                # Hanya ambil kata > 3 huruf dan bukan stopword
                valid_tokens = [t for t in tokens if len(t) > 3 and t not in stopwords]
                
                # Ambil top N paling sering muncul
                top_words = [word for word, count in Counter(valid_tokens).most_common(n)]
                return top_words

            # Eksekusi analisis
            top_menus = extract_top_keywords('menu', 60)
            top_locs = extract_top_keywords('alamat', 50)
            top_facilities = extract_top_keywords('fasilitas', 30)
            
            self.priority_vocabulary.update(top_menus)
            self.priority_vocabulary.update(top_locs)
            self.priority_vocabulary.update(top_facilities)
            
            # 4. Keyword Manual (BERDASARKAN TOP MENU & LOKASI AKTUAL)
            manual_priority = {
                # Top Menu Items
                'chicken', 'kopi', 'coffee', 'latte', 'beef', 'bakar', 'ramen', 
                'tahu', 'sate', 'katsu', 'rice', 'steak', 'cheese', 'susu', 
                'pizza', 'cream', 'ikan', 'sambal', 'udang', 'burger', 'matcha',
                
                # Top Locations
                'braga', 'sukajadi', 'cibeunying', 'coblong', 'lengkong', 
                'citarum', 'antapani', 'cicendo', 'dago', 'juanda', 'gegerkalong',
                
                # Fasilitas
                'wifi', 'parkiran', 'toilet',
                
                # Suasana
                'santai', 'nyaman', 'trendi', 'tenang', 'romantis'
            }
            self.priority_vocabulary.update(manual_priority)
            
            # 5. Semantic Keys (Penting agar "nugs" -> "nugas" bisa terdeteksi)
            self.priority_vocabulary.update(SEMANTIC_EXPANSION.keys())
            self.vocabulary.update(SEMANTIC_EXPANSION.keys())
            
            print(f"[INFO] Ukuran Vocabulary: {len(self.vocabulary)} kata unik")
            print(f"[INFO] Priority Vocabulary: {len(self.priority_vocabulary)} kata kunci utama (Auto-Generated)")
            
        except Exception as e:
            raise Exception(f"Error building vocabulary: {str(e)}")
    
    def _create_tfidf_matrix(self):
        """Membuat TF-IDF matrix"""
        try:
            print("[INFO] Membuat TF-IDF matrix...")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            # Rumus TF-IDF
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['metadata_tfidf_processed'])
            
            if self.tfidf_matrix.shape[0] == 0:
                raise ValueError("TF-IDF matrix kosong! Periksa data metadata_tfidf.")
                
        except Exception as e:
            raise Exception(f"Error membuat TF-IDF matrix: {str(e)}")
    
    # ========================================================================
    # METODE PEMBANTU UNTUK PEMROSESAN QUERY
    # ========================================================================
    
    def _normalize_raw_text(self, text):
        """Normalisasi teks mentah untuk exact matching"""
        return str(text).lower().strip().replace("   ", " ").replace("  ", " ")
    
    def _check_exact_match(self, query):
        """Cek apakah query adalah exact match dengan nama restoran"""
        normalized_query = self._normalize_raw_text(query)
        return (self.df['nama_rumah_makan'].apply(self._normalize_raw_text) == normalized_query).any()
    
    def _apply_synonym_normalization(self, query):
        """Menerapkan normalisasi sinonim pada query"""
        for synonym, replacement in SYNONYM_MAP.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            query = re.sub(pattern, replacement, query)
        return query
    
    def _apply_autocorrect(self, query):
        """Menerapkan auto-correct pada query"""
        corrected_words = []
        query_words = query.split()
        was_corrected = False
        
        for word in query_words:
            # Cek 1: Apakah kata ada di vocabulary dataset?
            if word in self.vocabulary:
                corrected_words.append(word)
            # Cek 2: Apakah kata adalah kata umum (Whitelist)?
            elif word in COMMON_WORDS:
                corrected_words.append(word)
            # Cek 3: Coba koreksi typo
            else:
                # Threshold dinamis: 0.82 default, tapi 0.70 untuk kata pendek (karena 1 huruf salah dari 4 = 0.75)
                threshold = 0.82
                if len(word) <= 4:
                    threshold = 0.70
                
                # Ambil lebih banyak kandidat (n=3) untuk pengecekan prioritas
                matches = get_close_matches(word, self.vocabulary, n=3, cutoff=threshold)
                
                if matches:
                    suggestion = matches[0] # Default: ambil yang paling mirip score-nya
                    
                    # Koreksi Cerdas: Prioritaskan kata yang lebih penting
                    # Contoh: "kopu" -> "kopi" (kategori) bukan "kopo" (nama jalan)
                    for match in matches:
                        if match in self.priority_vocabulary:
                            suggestion = match
                            break
                    
                    corrected_words.append(suggestion)
                    if word != suggestion:
                        print(f"[INFO] Auto-correct: '{word}' -> '{suggestion}'")
                        was_corrected = True
                else:
                    corrected_words.append(word)
        
        if was_corrected:
            corrected_query = " ".join(corrected_words)
            print(f"[INFO] Corrected Query: {corrected_query}")
            return corrected_query
        
        return query
    
    def _apply_semantic_expansion(self, query, processed_query):
        """Menerapkan ekspansi semantik pada query"""
        query_lower = query.lower()
        expanded_terms = []
        
        for term, keywords in SEMANTIC_EXPANSION.items():
            if term in query_lower:
                expanded_terms.append(keywords)
                print(f"[INFO] Semantic Expansion: '{term}' -> '{keywords}'")
        
        if expanded_terms:
            return processed_query + " " + " ".join(expanded_terms)
        return processed_query
    
    def _extract_filters(self, query_normalized):
        """Ekstrak filter tambahan dari query (lokasi, harga, suasana, fasilitas)"""
        additional_filters = set()
        
        # Ekstrak lokasi
        try:
            alamat_words = ' '.join(self.df['alamat'].dropna().astype(str)).lower()
            location_keywords = set([w for w in alamat_words.split() if len(w) >= 4 and w.isalpha()])
            
            # Hapus kata umum kuliner dari deteksi lokasi (BERDASARKAN KATEGORI & MENU AKTUAL)
            ignore_location_terms = {
                # Kategori
                'cafe', 'dessert', 'chinese', 'food', 'japanese', 'korean', 
                'western', 'middle', 'eastern', 'masakan', 'indonesia', 'aneka',
                
                # Menu Populer
                'chicken', 'kopi', 'coffee', 'latte', 'beef', 'bakar', 'ramen',
                'tahu', 'sate', 'katsu', 'rice', 'steak', 'cheese', 'pizza',
                
                # Kata Umum
                'jalan', 'kota', 'bandung', 'kecamatan', 'kelurahan', 'nomor',
                'utara', 'selatan', 'barat', 'timur', 'tengah', 'jawa'
            }
            location_keywords = location_keywords - ignore_location_terms
            additional_filters.update(location_keywords)
        except:
            pass
        
        # Ekstrak harga
        price_keywords = {'murah', 'mahal', 'sedang', 'terjangkau', 'hemat', 'premium', 'mewah', 'budget', 'promo'}
        additional_filters.update(price_keywords)
        
        # Ekstrak suasana
        try:
            suasana_words = ' '.join(self.df['suasana'].dropna().astype(str)).lower()
            additional_filters.update([w.strip() for w in suasana_words.split(',') if len(w.strip()) >= 4])
        except:
            pass
        
        # Ekstrak fasilitas
        try:
            fasilitas_words = ' '.join(self.df['fasilitas'].dropna().astype(str)).lower()
            additional_filters.update([w.strip() for w in fasilitas_words.split(',') if len(w.strip()) >= 4])
        except:
            pass
        
        # Filter kata-kata yang tidak relevan
        additional_filters = {w.strip() for w in additional_filters 
                             if len(w.strip()) >= 3 and w.strip() not in {'dan', 'yang', 'untuk', 'dari', 'dengan'}}
        
        # Exclude price terms agar tidak dianggap filter lokasi/fasilitas context
        price_terms = {'murah', 'mahal', 'sedang', 'terjangkau', 'ekonomis', 'hemat', 'premium', 'mewah', 'standar', 'menengah'}
        additional_filters = additional_filters - price_terms
        
        # Deteksi filter aktif dari query
        detected_raw = [kw for kw in additional_filters if kw in query_normalized.lower()]
        
        return detected_raw
    
    # ========================================================================
    # METODE PEMBANTU UNTUK PENILAIAN SKOR
    # ========================================================================
    
    def _apply_category_matching(self, similarity_scores, query_normalized, has_additional_filter):
        """Menerapkan category matching logic"""
        all_categories = set(str(cat).lower() for cat in self.df['kategori'].dropna().unique())
        all_tipe_pengunjung = set()
        
        for val in self.df['tipe_pengunjung'].dropna():
            for item in str(val).split(','):
                cleaned = item.strip().lower()
                if len(cleaned) >= 4:
                    all_tipe_pengunjung.add(cleaned)
        
        # Cari kategori yang cocok
        matched_category = None
        sorted_categories = sorted(all_categories, key=len, reverse=True)
        for category in sorted_categories:
            if category in query_normalized:
                matched_category = category
                print(f"[DEBUG] MATCHED CATEGORY: '{category}'")
                break
        
        # Cari tipe pengunjung yang cocok
        matched_tipe_pengunjung = None
        if not matched_category:
            sorted_tipe = sorted(all_tipe_pengunjung, key=len, reverse=True)
            for tipe in sorted_tipe:
                if tipe in query_normalized:
                    matched_tipe_pengunjung = tipe
                    print(f"[DEBUG] MATCHED TIPE: '{tipe}'")
                    break
        
        # Terapkan strict mode atau flexible mode
        strict_mode_activated = False
        
        # 1. ENFORCE STRICT MODE FOR CATEGORY (Selalu aktif jika kategori terdeteksi)
        if matched_category:
            print(f"[STRICT MODE] Enforcing Category: '{matched_category}'")
            
            # Handle special case for cafe aliases to be safe
            if any(x in matched_category for x in ['kopi', 'cafe', 'kafe', 'coffee', 'dessert']):
                 category_mask = self.df['kategori'].astype(str).str.lower().str.contains(
                    'kopi|cafe|kafe|coffee|dessert', na=False, regex=True
                )
            else:
                category_mask = self.df['kategori'].astype(str).str.lower() == matched_category
                
            # PENALTI BERAT untuk kategori yang salah (-1000)
            similarity_scores[~category_mask] = -1000.0
            similarity_scores[category_mask] += 1.0
            strict_mode_activated = True
            
        # 2. STRICT MODE FOR VISITOR TYPE (Hanya jika tidak ada filter lain)
        elif matched_tipe_pengunjung and not has_additional_filter:
            print(f"[STRICT MODE] Tipe: '{matched_tipe_pengunjung}'")
            tipe_mask = self.df['tipe_pengunjung'].astype(str).str.lower().str.contains(
                matched_tipe_pengunjung, na=False, regex=False
            )
            similarity_scores[~tipe_mask] = -1000.0
            similarity_scores[tipe_mask] += 1.0
            strict_mode_activated = True
        
        # 3. FLEXIBLE MODE (Hanya tersisa untuk Tipe Pengunjung dengan filter lain)
        else:
            if matched_tipe_pengunjung:
                tipe_mask = self.df['tipe_pengunjung'].astype(str).str.lower().str.contains(
                    matched_tipe_pengunjung, na=False, regex=False
                )
                similarity_scores[tipe_mask] += 5.0
            
            # Manual cafe intent fallback
            elif 'cafe' in query_normalized:
                print("[INFO] Cafe intent detected (manual fallback)")
                category_mask = self.df['kategori'].astype(str).str.lower().str.contains(
                    'kopi|cafe|kafe|coffee', na=False, regex=True
                )
                similarity_scores[~category_mask] = -1000.0 # Enforce strict here too
                similarity_scores[category_mask] += 1.0
                matched_category = 'cafe & dessert'
        
        return similarity_scores, matched_category, strict_mode_activated
    
    def _apply_location_boost(self, similarity_scores, active_filters):
        """Menerapkan boost untuk lokasi"""
        if len(active_filters) > 0:
            for flt in active_filters:
                search_terms = LOCATION_EXPANSION.get(flt, [flt])
                
                addr_mask = pd.Series([False] * len(self.df), index=self.df.index)
                for term in search_terms:
                    term_mask = self.df['alamat'].astype(str).str.lower().str.contains(term, na=False)
                    addr_mask = addr_mask | term_mask
                
                # Boost Match
                similarity_scores[addr_mask] += 15.0
                
                # Penalty Mismatch
                if addr_mask.sum() > 0:
                    similarity_scores[~addr_mask] -= 50.0
                    print(f"[DEBUG] Applied Location Boost (+15.0) & Penalty (-50.0) for '{flt}' (Expanded: {search_terms})")
        
        return similarity_scores
    
    def _apply_content_boost(self, similarity_scores, query_lower):
        """Menerapkan boost untuk konten (nama/menu)"""
        price_terms = {'murah', 'mahal', 'sedang', 'terjangkau', 'hemat', 'premium', 'mewah', 'budget', 'promo', 'murmer'}
        common_stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'atau', 'ini', 'itu', 'makan', 'minum', 'tempat', 'warung', 'resto'}
        ignore_terms = price_terms | common_stopwords
        
        core_words = [w for w in query_lower.split() if w not in ignore_terms and len(w) > 2]
        
        if core_words:
            processed_concepts = set()
            all_search_terms = set(core_words)
            
            for word in core_words:
                if word in SIMPLE_SYNONYMS:
                    all_search_terms.update(SIMPLE_SYNONYMS[word])
            
            for word in all_search_terms:
                concept = CONCEPT_MAP.get(word, word)
                
                if concept in processed_concepts:
                    continue
                
                # Cek kata lokasi untuk menghindari boost ganda pada nama jalan
                is_loc = word in LOCATION_EXPANSION or word in ['dago', 'braga', 'riau', 'juanda']
                boost_val = 2.0 if is_loc else 10.0
                
                safe_word = re.escape(word)
                name_mask = self.df['nama_rumah_makan'].astype(str).str.lower().str.contains(safe_word, na=False)
                menu_mask = self.df['menu'].astype(str).str.lower().str.contains(safe_word, na=False)
                anywhere_mask = name_mask | menu_mask
                
                if anywhere_mask.any():
                    similarity_scores[anywhere_mask] += boost_val
                    processed_concepts.add(concept)
            
            # Exact phrase match boost (MENU ADALAH PRIORITAS UTAMA: +50.0)
            if len(core_words) >= 2:
                phrase = " ".join(core_words)
                safe_phrase = re.escape(phrase)
                
                phrase_name_mask = self.df['nama_rumah_makan'].astype(str).str.lower().str.contains(safe_phrase, na=False)
                phrase_menu_mask = self.df['menu'].astype(str).str.lower().str.contains(safe_phrase, na=False)
                phrase_mask = phrase_name_mask | phrase_menu_mask
                similarity_scores[phrase_mask] += 50.0
        
        return similarity_scores
    
    def _apply_price_boost(self, similarity_scores, query_lower, price_filter):
        """Menerapkan boost untuk harga"""
        BOOST_FACTOR = 15.0
        
        is_murah = any(k in query_lower for k in ['murah', 'terjangkau', 'hemat', 'low budget'])
        is_sedang = any(k in query_lower for k in ['sedang', 'standar', 'menengah', 'reasonable'])
        is_mahal = any(k in query_lower for k in ['mahal', 'premium', 'mewah', 'fancy'])
        
        if price_filter and price_filter != "Semua":
            if price_filter == "Murah":
                is_murah, is_sedang, is_mahal = True, False, False
            elif price_filter == "Sedang":
                is_murah, is_sedang, is_mahal = False, True, False
            elif price_filter == "Mahal":
                is_murah, is_sedang, is_mahal = False, False, True
        
        relevant_mask = similarity_scores > -500
        
        if is_murah:
            mask = self.df['kategori_harga'].astype(str).str.contains('Murah', case=False, na=False)
            similarity_scores[mask & relevant_mask] += BOOST_FACTOR
        elif is_mahal:
            mask = self.df['kategori_harga'].astype(str).str.contains('Mahal', case=False, na=False)
            similarity_scores[mask & relevant_mask] += BOOST_FACTOR
        elif is_sedang:
            mask = self.df['kategori_harga'].astype(str).str.contains('Sedang', case=False, na=False)
            similarity_scores[mask & relevant_mask] += BOOST_FACTOR
        
        return similarity_scores, is_murah, is_sedang, is_mahal
    
    def _apply_exact_name_matching(self, similarity_scores, query):
        """Menerapkan exact/fuzzy name matching dengan boost tinggi"""
        try:
            from rapidfuzz import fuzz
            
            def normalize_text(text):
                text = str(text).lower().strip()
                text = re.sub(r'\s+', ' ', text)
                text = text.replace("'", "'").replace("`", "'")
                return text
            
            query_clean = normalize_text(query)
            query_len = len(query_clean)
            
            # Cek kecocokan persis
            df_names_normalized = self.df['nama_rumah_makan'].apply(normalize_text)
            exact_matches = df_names_normalized == query_clean
            
            if exact_matches.any():
                exact_matches_array = exact_matches.values
                similarity_scores[exact_matches_array] += 2000.0
                
                matched_names = self.df.loc[exact_matches, 'nama_rumah_makan'].tolist()
                for name in matched_names:
                    print(f"[EXACT MATCH 100%] '{name}' matched query '{query}'")
            
            # Cek kecocokan fuzzy
            elif query_len >= 8:
                top_indices = similarity_scores.argsort()[-100:][::-1]
                
                for idx in top_indices:
                    nama_resto = normalize_text(self.df.iloc[idx]['nama_rumah_makan'])
                    similarity_ratio = fuzz.ratio(query_clean, nama_resto)
                    partial_ratio = fuzz.partial_ratio(query_clean, nama_resto)
                    best_ratio = max(similarity_ratio, partial_ratio)
                    
                    if best_ratio >= 88.0:
                        similarity_scores[idx] += 8.0
                        print(f"[NEAR MATCH {best_ratio:.1f}%] '{self.df.iloc[idx]['nama_rumah_makan']}' matched query '{query}'")
                        break
                    elif best_ratio >= 80.0 and query_len >= 10:
                        similarity_scores[idx] += 5.0
                        print(f"[GOOD MATCH {best_ratio:.1f}%] '{self.df.iloc[idx]['nama_rumah_makan']}' matched query '{query}'")
                        break
                        
        except Exception as e:
            print(f"[WARNING] Fuzzy matching error: {str(e)}")
        
        return similarity_scores
    
    def _generate_warning_message(self, top_recommendations, is_murah, is_sedang, is_mahal, query, matched_category):
        """Membuat pesan peringatan cerdas"""
        target_price = None
        if is_murah:
            target_price = "Murah"
        elif is_mahal:
            target_price = "Mahal"
        elif is_sedang:
            target_price = "Sedang"
        
        if target_price and not top_recommendations.empty:
            if 'kategori_harga' in top_recommendations.columns:
                max_score = top_recommendations['similarity_score'].max()
                relevance_threshold = max(0, max_score - 3.0)
                
                checked_recs = top_recommendations.head(5)
                relevant_in_top5 = checked_recs[checked_recs['similarity_score'] >= relevance_threshold]
                matched_count = relevant_in_top5['kategori_harga'].astype(str).str.contains(target_price, case=False, na=False).sum()
                
                if matched_count == 0:
                    return f"Maaf, kami tidak menemukan rekomendasi yang pas untuk '{query}' dengan harga '{target_price}' di top 5 hasil. Berikut adalah rekomendasi terbaik yang kami temukan."
        
        return None
    
    # ========================================================================
    # METODE REKOMENDASI UTAMA
    # ========================================================================
    
    def get_recommendations(self, query, price_filter=None, top_n=5):
        """Mendapatkan rekomendasi UMKM berdasarkan query pengguna"""
        
        if not query or not isinstance(query, str):
            raise ValueError("Query harus berupa string yang tidak kosong!")
        
        query = query.strip()
        if not query:
            raise ValueError("Query tidak boleh kosong atau hanya spasi!")
        
        # Pre-check exact match
        raw_match_exists = self._check_exact_match(query)
        
        # Pipeline Optimasi: Bersihkan -> Koreksi -> Normalisasi -> Ekstrak -> Stemming
        try:
            # 1. Pembersihan dasar dan case folding (belum stemming)
            query_clean = self.preprocessor.clean_text(query)
            
            # 2. Koreksi otomatis typo pada kata asli
            # Contoh: "kopu" -> "kopi", "dgo" -> "dago"
            query_corrected = self._apply_autocorrect(query_clean)
            
            # 3. Normalisasi sinonim setelah koreksi typo
            query_normalized = self._apply_synonym_normalization(query_corrected.lower())
            
            # 4. Ekspansi semantik pada query yang sudah dikoreksi
            query_expanded = self._apply_semantic_expansion(query_normalized, query_normalized)
            
            # 5. Preprocessing akhir (Stemming dan penghapusan stopwords) untuk TF-IDF
            processed_query = self.preprocessor.preprocess(query_expanded)
            
            # Validasi query terlalu pendek
            if not raw_match_exists and len(processed_query.strip()) < 2:
                print(f"[INFO] Query '{processed_query}' diabaikan karena terlalu pendek.")
                return pd.DataFrame(), None, query_corrected
            
            if not processed_query.strip():
                return pd.DataFrame(), None, query_corrected
                
            # Gunakan query yang sudah dikoreksi untuk ditampilkan ke user
            # Bukan query yang sudah dinormalisasi ke kategori
            query = query_corrected
            
        except Exception as e:
            raise Exception(f"Error preprocessing query pipeline: {str(e)}")
        
        # Perhitungan kemiripan dan filter
        try:
            # Vectorization
            query_vector = self.vectorizer.transform([processed_query])
            
            # Rumus Cosine Similarity
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Ekstrak filter tambahan (LOKASI/HARGA/FASILITAS)
            # Menggunakan query_normalized agar filter bisa mendeteksi kata yang sudah dikoreksi
            query_lower = query_normalized
            active_filters = self._extract_filters(query_normalized)
            has_additional_filter = len(active_filters) > 0
            
            # Apply category matching
            similarity_scores, matched_category, strict_mode = self._apply_category_matching(
                similarity_scores, query_normalized, has_additional_filter
            )
            
            # Apply location boost
            similarity_scores = self._apply_location_boost(similarity_scores, active_filters)
            
            # Apply content boost
            similarity_scores = self._apply_content_boost(similarity_scores, query_lower)
            
            # Apply price boost
            similarity_scores, is_murah, is_sedang, is_mahal = self._apply_price_boost(
                similarity_scores, query_lower, price_filter
            )
            
            # Boost kombinasi sempurna: Kategori + Harga (Lokasi opsional)
            detected_price = "Murah" if is_murah else ("Sedang" if is_sedang else ("Mahal" if is_mahal else None))
            if matched_category and detected_price:
                self._apply_perfect_match_boost(similarity_scores, matched_category, active_filters, detected_price)
            
        except Exception as e:
            raise Exception(f"Error menghitung similarity: {str(e)}")
        # Terapkan pencocokan nama persis
        similarity_scores = self._apply_exact_name_matching(similarity_scores, query)
        
        # Buat hasil rekomendasi
        try:
            result_df = self.df.copy()
            result_df['similarity_score'] = similarity_scores
            
            top_recommendations = result_df.nlargest(top_n, 'similarity_score')
            top_recommendations = top_recommendations[top_recommendations['similarity_score'] > 0]
            
            # Pencarian cadangan
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
            
            # Buat pesan peringatan
            warning_msg = self._generate_warning_message(
                top_recommendations, is_murah, is_sedang, is_mahal, query, matched_category
            )
            
            return top_recommendations, warning_msg, query
            
        except Exception as e:
            raise Exception(f"Error memproses hasil rekomendasi: {str(e)}")

    def _apply_perfect_match_boost(self, similarity_scores, matched_category, active_filters, price_filter):
        """Memberikan boost besar untuk perfect match (Kategori + Lokasi + Harga)"""
        # Inisialisasi mask filter
        cat_mask = pd.Series([False] * len(self.df), index=self.df.index)
        
        # 1. Mask Kategori
        if any(x in matched_category for x in ['kopi', 'cafe', 'kafe', 'coffee', 'dessert']):
             cat_mask = self.df['kategori'].astype(str).str.lower().str.contains(
                'kopi|cafe|kafe|coffee|dessert', na=False, regex=True
            )
        else:
            cat_mask = self.df['kategori'].astype(str).str.lower() == matched_category
            
        # 2. Mask Harga
        price_mask = pd.Series([False] * len(self.df), index=self.df.index)
        query_lower = price_filter.lower()
        
        if any(k in query_lower for k in ['murah', 'terjangkau', 'hemat', 'low budget']):
            price_mask = self.df['kategori_harga'] == 'Murah'
        elif any(k in query_lower for k in ['sedang', 'standar', 'menengah']):
            price_mask = self.df['kategori_harga'] == 'Sedang'
        elif any(k in query_lower for k in ['mahal', 'premium', 'mewah']):
             price_mask = self.df['kategori_harga'] == 'Mahal'
             
        # 3. Mask Lokasi (OPSIONAL)
        if active_filters and len(active_filters) > 0:
            loc_mask = pd.Series([False] * len(self.df), index=self.df.index)
            for flt in active_filters:
                search_terms = LOCATION_EXPANSION.get(flt, [flt])
                for term in search_terms:
                    term_mask = self.df['alamat'].astype(str).str.lower().str.contains(term, na=False)
                    loc_mask = loc_mask | term_mask
        else:
            # Jika tidak ada filter lokasi, anggap SEMUA lokasi cocok (boost hanya berdasarkan Kategori + Harga)
            loc_mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        # PERFECT COMBO
        perfect_mask = cat_mask & price_mask & loc_mask
        
        if perfect_mask.any():
            similarity_scores[perfect_mask] += 50.0 # BUMP UP TO TOP!
            
    # ========================================================================
    # METODE UTILITAS
    # ========================================================================
    
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