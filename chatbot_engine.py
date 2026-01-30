# ============================================================================
# MESIN CHATBOT - LOGIKA REKOMENDASI UTAMA
# ============================================================================

import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import TextPreprocessor
from difflib import get_close_matches


# ============================================================================
# KONFIGURASI CHATBOT (DATA-DRIVEN)
# ============================================================================

SYNONYM_MAP = {
    # 1. Normalisasi Kategori (Menggabungkan variasi ke kategori baku dataset)
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
    'jepang': 'japanese food', 'japan': 'japanese food', 'jpn': 'japanese food',
    'korea': 'korean food', 'korean': 'korean food', 'korsel': 'korean food',
    'barat': 'western food', 'western': 'western food',
    'arab': 'middle eastern', 'timur tengah': 'middle eastern',
    'indo': 'masakan indonesia', 'indonesia': 'masakan indonesia', 'nusantara': 'masakan indonesia',
    'aneka': 'aneka masakan', 'campur': 'aneka masakan',
    'non halal': 'masakan non halal', 'babi': 'masakan non halal', 'pork': 'masakan non halal',
    
    # 2. Bilingual Mapping (Diarahkan ke bahasa yang DOMINAN di dataset)
    'chicken': 'ayam',
    'sapi': 'beef', 'daging': 'beef',
    'fish': 'ikan',
    'shrimp': 'udang',
    'rice': 'nasi',
    'noodle': 'mie',
    'fried': 'goreng',
    'grilled': 'bakar', 'panggang': 'bakar',
    'soup': 'kuah',
    'spicy': 'pedas', 'pedes': 'pedas',
    'sweet': 'manis',
    
    # 3. Singkatan & Variasi (Memperluas coverage keyword user)
    'nasgor': 'nasi goreng', 'nasigoreng': 'nasi goreng',
    'miegoreng': 'mie goreng', 'miekuah': 'mie kuah',
    'migor': 'mie goreng', 'aygor': 'ayam goreng',
    'geprek': 'ayam geprek', 'burayam': 'bubur ayam',
    'sotayam': 'soto ayam', 'yamin': 'mie yamin', 'padang': 'nasi padang',
    'wifi': 'wi fi', 'wi-fi': 'wi fi', 'hotspot': 'wi fi', 'internet': 'wi fi',
    
    # Lokasi Multi-kata -> Single Token
    'buah batu': 'buahbatu',
    'babakan ciparay': 'babakanciparay',
    'bandung kidul': 'bandungkidul', 'bandung kulon': 'bandungkulon', 'bandung wetan': 'bandungwetan',
    'bojongloa kaler': 'bojongloakaler', 'bojongloa kidul': 'bojongloakidul',
    'cibeunying kaler': 'cibeunyingkaler', 'cibeunying kidul': 'cibeunyingkidul',
    'sumur bandung': 'sumurbandung',
    'ujung berung': 'ujungberung',
    
    # 4. Normalisasi Ejaan (Berdasarkan bentuk yang paling sering muncul)
    'resto': 'restoran', 'rm': 'rumah makan',
    'dim sum': 'dimsum', 'dumpling': 'dimsum',
    'jus': 'juice', 'es krim': 'ice cream'
}

SEMANTIC_EXPANSION = {
    # 1. Aktivitas & Fasilitas (Berdasarkan Fasilitas Riil: Wi-Fi(133), Parkiran(412), Toilet(424))
    'nugas': 'wi fi', 
    'kerja': 'wi fi', 
    'laptop': 'wi fi',
    'meeting': 'wi fi',
    
    'keluarga': 'parkiran toilet',
    'rombongan': 'parkiran toilet',
    'bukber': 'parkiran toilet',
    
    'wc': 'toilet', 'kamar mandi': 'toilet',
    'mobil': 'parkiran', 'motor': 'parkiran',
    
    'date': 'romantis', 'pacaran': 'romantis', 'couple': 'romantis', 'pasangan': 'romantis',
    'nongkrong': 'santai', 'hangout': 'santai',
    
    # 2. Mapping Kategori Makanan
    'sushi': 'japanese food', 'ramen': 'japanese food', 'udon': 'japanese food',
    'steak': 'western food', 'pasta': 'western food', 'burger': 'western food', 'pizza': 'western food', 
    'spaghetti': 'western food', 'lasagna': 'western food',
    'dim sum': 'chinese food', 'bakpao': 'chinese food', 'kwetiau': 'chinese food', 'capcay': 'chinese food',
    'roti': 'cafe & dessert', 'cake': 'cafe & dessert', 'croissant': 'cafe & dessert', 
    'pancake': 'cafe & dessert', 'waffle': 'cafe & dessert',
    'kimchi': 'korean food', 'bibimbap': 'korean food',
    'kebab': 'middle eastern',
    'sate': 'masakan indonesia', 'soto': 'masakan indonesia', 'bakso': 'masakan indonesia', 'gudeg': 'masakan indonesia',
    'padang': 'masakan indonesia', 'nasi padang': 'masakan indonesia', 'rendang': 'masakan indonesia'
}

LOCATION_EXPANSION = {
    'andir': ['andir', 'ciroyom', 'dungus cariang', 'garuda', 'kebon jeruk', 'maleber'],
    'antapani': ['antapani', 'antapani kidul', 'antapani tengah', 'antapani wetan'],
    'arcamanik': ['arcamanik', 'cisaranten kulon', 'sukamiskin'],
    'astanaanyar': ['astanaanyar', 'cibadak', 'ebeh no', 'karanganyar'],
    'babakanciparay': ['babakan ciparay', 'babakanciparay', 'babakan', 'sukahaji'],
    'bandungkidul': ['bandung kidul', 'bandungkidul', 'batununggal', 'kujangsari'],
    'bandungkulon': ['bandung kulon', 'bandungkulon', 'cigondewah rahayu', 'cijerah'],
    'bandungwetan': ['bandung wetan', 'bandungwetan', 'cihapit', 'citarum', 'merdeka', 'tamansari'],
    'batununggal': ['batununggal', 'gumuruh', 'kacapiring', 'kebonwaru'],
    'bojongloakaler': ['bojongloa kaler', 'bojongloakaler', 'babakan asih', 'jamika', 'suka asih'],
    'bojongloakidul': ['bojongloa kidul', 'bojongloakidul', 'mekarwangi', 'situsaeur'],
    'buahbatu': ['buahbatu', 'buah batu', 'cijaura', 'margasari', 'sekejati'],
    'cibeunyingkaler': ['cibeunying kaler', 'cibeunyingkaler', 'cigadung', 'cihaur geulis', 'neglasari', 'sukaluyu'],
    'cibeunyingkidul': ['cibeunying kidul', 'cibeunyingkidul', 'cicadas', 'cihaur geulis', 'cikutra', 'padasuka', 'sukamaju', 'sukapada'],
    'cibiru': ['cibiru', 'cipadung'],
    'cicendo': ['arjuna', 'cicendo', 'ciroyom', 'gunung batu', 'pajajaran', 'pamoyanan', 'pasir kaliki', 'sukaraja'],
    'cidadap': ['cidadap', 'ciumbuleuit', 'hegarmanah', 'ledeng'],
    'cinambo': ['cinambo', 'pakemitan'],
    'coblong': ['cipaganti', 'coblong', 'dago', 'lebak gede', 'lebakgede', 'sadang serang', 'sekeloa'],
    'gedebage': ['cisaranten kidul', 'gedebage'],
    'kiaracondong': ['babakan sari', 'binong', 'cicaheum', 'kiaracondong'],
    'lengkong': ['burangrang', 'cibangkong', 'cijagra', 'cikawao', 'lengkong', 'lingkar selatan', 'malabar', 'paledang', 'turangga'],
    'panyileukan': ['cipadung kidul', 'cipadung wetan', 'panyileukan'],
    'regol': ['ancol', 'balonggede', 'ciateul', 'cigereleng', 'pasirluyu', 'pungkur', 'regol'],
    'sukajadi': ['cipedes', 'gegerkalong', 'pasteur', 'sukabungah', 'sukagalih', 'sukajadi', 'sukawarna'],
    'sukasari': ['gegerkalong', 'isola', 'sarijadi', 'sukarasa', 'sukasari'],
    'sumurbandung': ['sumur bandung', 'sumurbandung', 'babakan ciamis', 'braga', 'kebon pisang', 'merdeka'],
    'ujungberung': ['ujung berung', 'ujungberung', 'pasir endah']
}


# ============================================================================
# KELAS MESIN CHATBOT
# ============================================================================

class ChatbotEngine:
    """Mesin utama chatbot untuk rekomendasi kuliner UMKM Kota Bandung.
    
    Sistem ini menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) dan
    Cosine Similarity untuk mencocokkan query user dengan dataset restoran.
    
    Komponen Utama:
    - Text Preprocessing: Sastrawi stemmer, stopword removal, normalization
    - Synonym & Semantic Mapping: Menangani variasi bahasa dan konteks
    - Category Matching: Strict mode untuk kategori makanan spesifik
    - Location Filtering: Boost restoran di area yang diminta user
    - Warning System: Mendeteksi konflik semantik dan mismatch filter
    
    Attributes:
        df (DataFrame): Dataset restoran yang sudah dimuat
        preprocessor (TextPreprocessor): Instance untuk text preprocessing
        vectorizer (TfidfVectorizer): Model TF-IDF untuk similarity calculation
        tfidf_matrix (sparse matrix): Matrix TF-IDF dari seluruh dataset
        vocabulary (set): Kumpulan kata unik dari dataset (untuk autocorrect)
        priority_vocabulary (set): Kata kunci prioritas (kategori, menu populer, lokasi)
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
                raise ValueError("Dataset kosong!")
            
            df['metadata_tfidf'] = df['metadata_tfidf'].fillna('')
            df['deskripsi'] = df['deskripsi'].fillna('')
            
            if df['metadata_tfidf'].str.strip().eq('').all():
                raise ValueError("[ERROR] Kolom metadata_tfidf kosong!")
            
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
            all_text = " ".join(self.df['metadata_tfidf'].astype(str).tolist())
            self.vocabulary = set(all_text.lower().split())
            
            self.priority_vocabulary = set()
            
            if 'kategori' in self.df.columns:
                categories = self.df['kategori'].dropna().unique()
                for cat in categories:
                    clean_cat = str(cat).lower().replace('/', ' ').replace('&', ' ')
                    self.priority_vocabulary.update(clean_cat.split())
            
            for val in SYNONYM_MAP.values():
                self.priority_vocabulary.update(val.split())
            
            print("[INFO] Menganalisis dataset untuk mengisi Priority Vocabulary...")
            
            def extract_top_keywords(column, n=50):
                if column not in self.df.columns:
                    return []
                
                text = ' '.join(self.df[column].dropna().astype(str).tolist()).lower()
                text = re.sub(r'[^a-z\s]', ' ', text)
                tokens = text.split()
                
                stopwords = {
                    'dan', 'yg', 'di', 'ke', 'dari', 'yang', 'dengan', 'untuk', 'utk', 
                    'jl', 'jalan', 'no', 'kota', 'bandung', 'jawa', 'barat', 
                    'kecamatan', 'kelurahan', 'rt', 'rw', 'nomor', 'blok', 'lantai',
                    'memiliki', 'tersedia', 'area', 'ada'
                }
                
                valid_tokens = [t for t in tokens if len(t) > 3 and t not in stopwords]
                top_words = [word for word, count in Counter(valid_tokens).most_common(n)]
                return top_words

            top_menus = extract_top_keywords('menu', 60)
            top_locs = extract_top_keywords('alamat', 50)
            top_facilities = extract_top_keywords('fasilitas', 30)
            
            self.priority_vocabulary.update(top_menus)
            self.priority_vocabulary.update(top_locs)
            self.priority_vocabulary.update(top_facilities)
            
            manual_priority = {
                'chicken', 'kopi', 'coffee', 'latte', 'beef', 'bakar', 'ramen', 
                'tahu', 'sate', 'katsu', 'rice', 'steak', 'cheese', 'susu', 
                'pizza', 'cream', 'ikan', 'sambal', 'udang', 'burger', 'matcha',
                'braga', 'sukajadi', 'cibeunying', 'coblong', 'lengkong', 
                'citarum', 'antapani', 'cicendo', 'dago', 'juanda', 'gegerkalong',
                'wifi', 'parkiran', 'toilet',
                'santai', 'nyaman', 'trendi', 'tenang', 'romantis'
            }
            self.priority_vocabulary.update(manual_priority)
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
            
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['metadata_tfidf_processed'])
            
            if self.tfidf_matrix.shape[0] == 0:
                raise ValueError("TF-IDF matrix kosong!")
                
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
        """Menerapkan normalisasi sinonim pada query.
        
        Menggunakan SYNONYM_MAP untuk mengganti variasi kata (e.g., 'chicken' -> 'ayam',
        'buah batu' -> 'buahbatu') agar query user konsisten dengan terminologi dataset.
        
        Args:
            query (str): Query yang sudah di-clean
            
        Returns:
            str: Query dengan sinonim yang sudah dinormalisasi
        """
        for synonym, replacement in SYNONYM_MAP.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            query = re.sub(pattern, replacement, query)
        return query
    
    def _apply_autocorrect(self, query):
        """Menerapkan auto-correct pada query menggunakan fuzzy matching.
        
        Menggunakan difflib.get_close_matches untuk mendeteksi typo dan mengoreksinya
        berdasarkan vocabulary yang dibangun dari dataset. Prioritas diberikan pada
        kata-kata di priority_vocabulary (kategori, menu populer, lokasi).
        
        Args:
            query (str): Query yang sudah dinormalisasi
            
        Returns:
            str: Query dengan typo yang sudah dikoreksi
        """
        corrected_words = []
        query_words = query.split()
        was_corrected = False
        
        for word in query_words:
            if word in self.vocabulary:
                corrected_words.append(word)
            else: # Removed 'elif word in COMMON_WORDS:'
                threshold = 0.82 if len(word) > 4 else 0.70
                matches = get_close_matches(word, self.vocabulary, n=3, cutoff=threshold)
                
                if matches:
                    suggestion = matches[0]
                    
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
        """Menerapkan ekspansi semantik pada query.
        
        Menambahkan kata kunci terkait berdasarkan SEMANTIC_EXPANSION. Contoh:
        - 'sushi' -> ditambah 'japanese food'
        - 'nugas' -> ditambah 'wi fi'
        Ini membantu mesin memahami konteks dan kategori dari query user.
        
        Args:
            query (str): Query asli (untuk deteksi term)
            processed_query (str): Query yang sudah diproses (untuk ditambahkan expansion)
            
        Returns:
            str: Query yang sudah di-expand dengan kata kunci semantik
        """
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
        """Ekstrak filter tambahan (lokasi, fasilitas, suasana) dari query.
        
        Menggunakan algoritma 'Longest Match Wins' untuk menghindari deteksi ganda.
        Contoh: Query 'buah batu' akan terdeteksi sebagai 1 filter ('buahbatu'),
        bukan 2 filter terpisah ('buah' dan 'batu').
        
        Args:
            query_normalized (str): Query yang sudah dinormalisasi
            
        Returns:
            list: Daftar filter yang terdeteksi (e.g., ['buahbatu', 'parkiran'])
        """
        additional_filters = set()
        
        try:
            alamat_words = ' '.join(self.df['alamat'].dropna().astype(str)).lower()
            location_keywords = set([w for w in alamat_words.split() if len(w) >= 4 and w.isalpha()])
            
            ignore_location_terms = {
                'cafe', 'dessert', 'chinese', 'food', 'japanese', 'korean', 
                'western', 'middle', 'eastern', 'masakan', 'indonesia', 'aneka',
                'chicken', 'kopi', 'coffee', 'latte', 'beef', 'bakar', 'ramen',
                'tahu', 'sate', 'katsu', 'rice', 'steak', 'cheese', 'pizza',
                'jalan', 'kota', 'bandung', 'kecamatan', 'kelurahan', 'nomor',
                'utara', 'selatan', 'barat', 'timur', 'tengah', 'jawa'
            }
            location_keywords = location_keywords - ignore_location_terms
            additional_filters.update(location_keywords)
            additional_filters.update(LOCATION_EXPANSION.keys()) # Tambahkan manual keys agar buahbatu dll terdeteksi
        except:
            pass
        
        price_keywords = {'murah', 'mahal', 'sedang', 'terjangkau', 'hemat', 'premium', 'mewah', 'budget', 'promo'}
        additional_filters.update(price_keywords)
        
        try:
            suasana_words = ' '.join(self.df['suasana'].dropna().astype(str)).lower()
            additional_filters.update([w.strip() for w in suasana_words.split(',') if len(w.strip()) >= 4])
        except:
            pass
        
        try:
            fasilitas_words = ' '.join(self.df['fasilitas'].dropna().astype(str)).lower()
            additional_filters.update([w.strip() for w in fasilitas_words.split(',') if len(w.strip()) >= 4])
        except:
            pass
        
        additional_filters = {w.strip() for w in additional_filters 
                             if len(w.strip()) >= 3 and w.strip() not in {'dan', 'yang', 'untuk', 'dari', 'dengan'}}
        
        price_terms = {'murah', 'mahal', 'sedang', 'terjangkau', 'ekonomis', 'hemat', 'premium', 'mewah', 'standar', 'menengah'}
        additional_filters = additional_filters - price_terms
        
        # 1. Deteksi dari keyword hasil ekstraksi dataset
        detected_raw = {kw for kw in additional_filters if kw in query_normalized.lower()}
        
        # 2. FORCE CHECK: Cek semua lokasi manual (LOCATION_EXPANSION)
        for loc_key in LOCATION_EXPANSION.keys():
            if loc_key in query_normalized.lower():
                detected_raw.add(loc_key)
        
        # 3. ALGORITMA "LONGEST MATCH WINS" (Pembersihan Filter)
        # Urutkan berdasarkan panjang string (terpanjang dulu)
        candidates = sorted(list(detected_raw), key=len, reverse=True)
        final_filters = []
        
        for cand in candidates:
            # Cek apakah kandidat ini adalah substring dari filter yang SUDAH diterima (yang lebih panjang)
            is_substring = False
            for accepted in final_filters:
                if cand in accepted:
                    is_substring = True
                    break
            
            if not is_substring:
                final_filters.append(cand)
        
        return final_filters
        
    # ========================================================================
    # METODE PEMBANTU UNTUK PENILAIAN SKOR
    # ========================================================================
    
    def _apply_category_matching(self, similarity_scores, query_normalized, has_additional_filter):
        """Menerapkan category matching logic dengan Strict Mode.
        
        Jika kategori makanan terdeteksi di query (e.g., 'japanese food', 'cafe & dessert'),
        sistem akan mengaktifkan Strict Mode: hanya restoran dengan kategori tersebut
        yang akan direkomendasikan. Ini mencegah hasil yang tidak relevan.
        
        Args:
            similarity_scores (np.array): Array skor similarity TF-IDF
            query_normalized (str): Query yang sudah di-expand (mengandung kategori)
            has_additional_filter (bool): Apakah ada filter lokasi/fasilitas aktif
            
        Returns:
            tuple: (similarity_scores, matched_category, strict_mode_activated)
        """
        all_categories = set(str(cat).lower() for cat in self.df['kategori'].dropna().unique())
        all_tipe_pengunjung = set()
        
        for val in self.df['tipe_pengunjung'].dropna():
            for item in str(val).split(','):
                cleaned = item.strip().lower()
                if len(cleaned) >= 4:
                    all_tipe_pengunjung.add(cleaned)
        
        matched_category = None
        sorted_categories = sorted(all_categories, key=len, reverse=True)
        for category in sorted_categories:
            if category in query_normalized:
                matched_category = category
                print(f"[DEBUG] MATCHED CATEGORY: '{category}'")
                break
        
        matched_tipe_pengunjung = None
        if not matched_category:
            sorted_tipe = sorted(all_tipe_pengunjung, key=len, reverse=True)
            for tipe in sorted_tipe:
                if tipe in query_normalized:
                    matched_tipe_pengunjung = tipe
                    print(f"[DEBUG] MATCHED TIPE: '{tipe}'")
                    break
        
        strict_mode_activated = False
        
        if matched_category:
            print(f"[STRICT MODE] Enforcing Category: '{matched_category}'")
            
            if any(x in matched_category for x in ['kopi', 'cafe', 'kafe', 'coffee', 'dessert']):
                 category_mask = self.df['kategori'].astype(str).str.lower().str.contains(
                    'kopi|cafe|kafe|coffee|dessert', na=False, regex=True
                )
            else:
                category_mask = self.df['kategori'].astype(str).str.lower() == matched_category
                
            similarity_scores[~category_mask] = -1000.0
            similarity_scores[category_mask] += 1.0
            strict_mode_activated = True
            
        elif matched_tipe_pengunjung and not has_additional_filter:
            print(f"[STRICT MODE] Tipe: '{matched_tipe_pengunjung}'")
            tipe_mask = self.df['tipe_pengunjung'].astype(str).str.lower().str.contains(
                matched_tipe_pengunjung, na=False, regex=False
            )
            similarity_scores[~tipe_mask] = -1000.0
            similarity_scores[tipe_mask] += 1.0
            strict_mode_activated = True
        
        else:
            if matched_tipe_pengunjung:
                tipe_mask = self.df['tipe_pengunjung'].astype(str).str.lower().str.contains(
                    matched_tipe_pengunjung, na=False, regex=False
                )
                similarity_scores[tipe_mask] += 5.0
            
            elif 'cafe' in query_normalized:
                print("[INFO] Cafe intent detected (manual fallback)")
                category_mask = self.df['kategori'].astype(str).str.lower().str.contains(
                    'kopi|cafe|kafe|coffee', na=False, regex=True
                )
                similarity_scores[~category_mask] = -1000.0
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
                
                similarity_scores[addr_mask] += 15.0
                
                if addr_mask.sum() > 0:
                    similarity_scores[~addr_mask] -= 50.0
                    print(f"[DEBUG] Applied Location Boost (+15.0) & Penalty (-50.0) for '{flt}' (Expanded: {search_terms})")
        
        return similarity_scores
    
    def _apply_content_boost(self, similarity_scores, query_lower):
        """Menerapkan boost untuk konten (nama/menu) berdasarkan keyword matching"""
        price_terms = {'murah', 'mahal', 'sedang', 'terjangkau', 'hemat', 'premium', 'mewah', 'budget', 'promo', 'murmer'}
        common_stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'atau', 'ini', 'itu', 'makan', 'minum', 'tempat', 'warung', 'resto', 'kafe', 'cafe'}
        ignore_terms = price_terms | common_stopwords
        
        core_words = [w for w in query_lower.split() if w not in ignore_terms and len(w) > 2]
        
        if core_words:
            all_search_terms = set(core_words)
            processed_concepts = set()
            
            # Simple Keyword Boosting
            for word in all_search_terms:
                if word in processed_concepts:
                    continue
                
                # Cek apakah ini lokasi (beri boost lebih kecil)
                is_loc = word in LOCATION_EXPANSION or word in ['dago', 'braga', 'riau', 'juanda']
                boost_val = 2.0 if is_loc else 10.0
                
                safe_word = re.escape(word)
                name_mask = self.df['nama_rumah_makan'].astype(str).str.lower().str.contains(safe_word, na=False)
                menu_mask = self.df['menu'].astype(str).str.lower().str.contains(safe_word, na=False)
                anywhere_mask = name_mask | menu_mask
                
                if anywhere_mask.any():
                    similarity_scores[anywhere_mask] += boost_val
                    processed_concepts.add(word)
            
            # Phrase Boosting (Urutan Kata)
            if len(core_words) >= 2:
                phrase = " ".join(core_words)
                safe_phrase = re.escape(phrase)
                
                phrase_name_mask = self.df['nama_rumah_makan'].astype(str).str.lower().str.contains(safe_phrase, na=False)
                phrase_menu_mask = self.df['menu'].astype(str).str.lower().str.contains(safe_phrase, na=False)
                phrase_mask = phrase_name_mask | phrase_menu_mask
                
                # Bonus besar untuk frasa utuh
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
            
            df_names_normalized = self.df['nama_rumah_makan'].apply(normalize_text)
            exact_matches = df_names_normalized == query_clean
            
            if exact_matches.any():
                exact_matches_array = exact_matches.values
                similarity_scores[exact_matches_array] += 2000.0
                
                matched_names = self.df.loc[exact_matches, 'nama_rumah_makan'].tolist()
                for name in matched_names:
                    print(f"[EXACT MATCH 100%] '{name}' matched query '{query}'")
            
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
    
    def _generate_warning_message(self, top_recommendations, is_murah, is_sedang, is_mahal, query, matched_category, active_filters):
        """Membuat pesan peringatan cerdas untuk user.
        
        Mendeteksi 3 jenis warning:
        1. Konflik Kategori: User mencari 2 kategori berbeda (e.g., 'sushi padang')
        2. Lokasi Mismatch: Hasil tidak ada di lokasi yang diminta user
        3. Harga Mismatch: Hasil tidak sesuai dengan filter harga user
        
        Args:
            top_recommendations (DataFrame): Top N hasil rekomendasi
            is_murah/is_sedang/is_mahal (bool): Flag filter harga aktif
            query (str): Query asli user
            matched_category (str): Kategori yang terdeteksi dari query
            active_filters (list): Daftar filter yang aktif
            
        Returns:
            str or None: Pesan warning jika ada kondisi yang perlu diperingatkan
        """
        if top_recommendations.empty:
            return None
            
        checked_recs = top_recommendations.head(5)
        
        # 0. CEK WARNING KONFLIK KATEGORI (Semantic Conflict)
        if matched_category:
            query_lower = query.lower()
            conflicting_term = None
            conflicting_cat = None
            
            # Daftar kategori makanan valid (untuk membedakan dengan fasilitas)
            food_categories = {
                'japanese food', 'western food', 'chinese food', 'korean food', 
                'masakan indonesia', 'middle eastern', 'cafe & dessert', 'masakan non halal'
            }
            
            if matched_category in food_categories:
                for term, mapped_cat in SEMANTIC_EXPANSION.items():
                    # Jika term ada di query TAPI kategorinya beda dengan matched_category
                    if term in query_lower and mapped_cat in food_categories and mapped_cat != matched_category:
                        conflicting_term = term
                        conflicting_cat = mapped_cat
                        break
                
                if conflicting_term:
                    return f"Sepertinya kamu mencari **'{matched_category.title()}'** sekaligus **'{conflicting_term}'**. Aku utamakan **{matched_category.title()}** dulu ya. Kalau kurang pas, coba cari dengan kata kunci yang lebih spesifik."

        # 0.5. CEK WARNING CONTENT + LOCATION MISMATCH
        # Deteksi jika user mencari keyword spesifik di lokasi tertentu, tapi keyword tidak ada di hasil
        location_filters = [f for f in active_filters if f in LOCATION_EXPANSION]
        
        if location_filters:
            # Ekstrak keyword utama dari query (bukan stopword, bukan lokasi, bukan kategori)
            query_lower = query.lower()
            stopwords_extended = {
                'enak', 'murah', 'bagus', 'recommended', 'rekomendasi', 'cari', 'mau', 'ingin',
                'di', 'daerah', 'sekitar', 'dekat', 'wilayah', 'area', 'yang', 'dong', 'sih'
            }
            
            # Hapus lokasi dan stopwords dari query
            query_words = query_lower.split()
            content_keywords = []
            
            # Buat set kata-kata yang merupakan bagian dari lokasi
            location_parts = set()
            for loc_filter in location_filters:
                # Pisahkan lokasi gabung jadi kata-kata (e.g., 'buahbatu' -> 'buah', 'batu')
                # Cek di LOCATION_EXPANSION untuk mendapatkan variasi dengan spasi
                loc_variants = LOCATION_EXPANSION.get(loc_filter, [loc_filter])
                for variant in loc_variants:
                    location_parts.update(variant.split())
            
            for word in query_words:
                # Skip jika stopword, lokasi, bagian dari lokasi, atau terlalu pendek
                if (word not in stopwords_extended and 
                    word not in location_filters and 
                    word not in location_parts and
                    len(word) >= 4 and
                    word not in ['cafe', 'kafe', 'resto', 'restoran', 'warung']):
                    content_keywords.append(word)
            
            # Jika ada keyword konten yang spesifik
            if content_keywords:
                # Cek apakah keyword ada di Top 5 results (nama atau menu)
                keyword_found_in_results = False
                for keyword in content_keywords:
                    for _, row in checked_recs.iterrows():
                        nama = str(row['nama_rumah_makan']).lower()
                        menu = str(row.get('menu', '')).lower()
                        if keyword in nama or keyword in menu:
                            keyword_found_in_results = True
                            break
                    if keyword_found_in_results:
                        break
                
                # Jika keyword TIDAK ditemukan di hasil, tapi lokasi cocok
                if not keyword_found_in_results:
                    target_loc = location_filters[0]
                    # Cek apakah hasil memang ada di lokasi yang diminta
                    loc_match_found = False
                    for _, row in checked_recs.iterrows():
                        addr_raw = str(row['alamat']).lower()
                        search_terms = LOCATION_EXPANSION.get(target_loc, [target_loc])
                        for term in search_terms:
                            if term in addr_raw or term.replace(" ", "") in addr_raw.replace(" ", ""):
                                loc_match_found = True
                                break
                        if loc_match_found:
                            break
                    
                    # Jika lokasi cocok tapi konten tidak cocok -> Warning
                    if loc_match_found:
                        keyword_str = "', '".join(content_keywords[:2])  # Ambil max 2 keyword
                        return f"Maaf, sepertinya **'{keyword_str}'** di daerah **{target_loc.replace('_', ' ').title()}** belum ada datanya. Tapi, coba cek rekomendasi kuliner lain di area tersebut ya!"

        # 1. CEK WARNING LOKASI (Prioritas Utama)
        # Pisahkan filter lokasi dari active_filters
        location_filters = [f for f in active_filters if f in LOCATION_EXPANSION]
        
        if location_filters:
            target_loc = location_filters[0] # Ambil satu lokasi utama
            search_terms = LOCATION_EXPANSION.get(target_loc, [target_loc])
            
            # Cek apakah ada satu pun hasil di top 5 yang alamatnya cocok
            loc_match_found = False
            for _, row in checked_recs.iterrows():
                addr_raw = str(row['alamat']).lower()
                # Normalisasi alamat (hapus spasi) agar "buahbatu" match dengan "buah batu"
                addr_clean = addr_raw.replace(" ", "")
                
                # Cek juga term yang sudah dinormalisasi
                match = False
                for term in search_terms:
                    # Cek term asli (dengan spasi jika ada)
                    if term in addr_raw:
                        match = True
                    # Cek term tanpa spasi
                    elif term.replace(" ", "") in addr_clean:
                        match = True
                    
                    if match: break
                
                if match:
                    loc_match_found = True
                    break
            
            if not loc_match_found:
                return f"Belum ada data kuliner di area **'{target_loc.title()}'** nih. Coba intip rekomendasi di daerah lain yang mungkin kamu suka."

        # 2. CEK WARNING HARGA
        target_price = None
        if is_murah:
            target_price = "Murah"
        elif is_mahal:
            target_price = "Mahal"
        elif is_sedang:
            target_price = "Sedang"
        
        if target_price:
            if 'kategori_harga' in checked_recs.columns:
                matched_count = checked_recs['kategori_harga'].astype(str).str.contains(target_price, case=False, na=False).sum()
                
                if matched_count == 0:
                    return f"Maaf, belum nemu rekomendasi yang pas untuk '{query}' dengan harga '{target_price}'. Tapi ini ada rekomendasi terbaik lainnya untukmu."
        
        return None
    
    # ========================================================================
    # METODE REKOMENDASI UTAMA
    # ========================================================================
    
    def get_recommendations(self, query, price_filter=None, top_n=5):
        """Mendapatkan rekomendasi UMKM berdasarkan query pengguna.
        
        Pipeline Lengkap:
        1. Preprocessing: Clean, Autocorrect, Synonym Normalization, Semantic Expansion
        2. TF-IDF Calculation: Menghitung similarity antara query dan dataset
        3. Boosting & Filtering: Category Matching, Location Boost, Content Boost, Price Boost
        4. Ranking: Mengurutkan hasil berdasarkan skor akhir
        5. Warning Generation: Mendeteksi konflik kategori/lokasi/harga
        
        Args:
            query (str): Query pencarian dari user (e.g., "sushi enak di dago")
            price_filter (str, optional): Filter harga ('Murah', 'Sedang', 'Mahal', atau None)
            top_n (int, optional): Jumlah rekomendasi yang dikembalikan (default: 5)
            
        Returns:
            tuple: (recommendations_df, warning_message, processed_query)
                - recommendations_df: DataFrame berisi top N rekomendasi
                - warning_message: Pesan warning jika ada (atau None)
                - processed_query: Query yang sudah diproses (untuk debugging)
                
        Raises:
            ValueError: Jika query kosong atau bukan string
        """
        
        if not query or not isinstance(query, str):
            raise ValueError("Query harus berupa string yang tidak kosong!")
        
        query = query.strip()
        if not query:
            raise ValueError("Query tidak boleh kosong atau hanya spasi!")
        
        raw_match_exists = self._check_exact_match(query)
        
        try:
            query_clean = self.preprocessor.clean_text(query)
            query_corrected = self._apply_autocorrect(query_clean)
            query_normalized = self._apply_synonym_normalization(query_corrected.lower())
            query_expanded = self._apply_semantic_expansion(query_normalized, query_normalized)
            processed_query = self.preprocessor.preprocess(query_expanded)
            
            if not raw_match_exists and len(processed_query.strip()) < 2:
                print(f"[INFO] Query '{processed_query}' diabaikan karena terlalu pendek.")
                return pd.DataFrame(), None, query_corrected
            
            if not processed_query.strip():
                return pd.DataFrame(), None, query_corrected
                
            query = query_corrected
            
        except Exception as e:
            raise Exception(f"Error preprocessing query pipeline: {str(e)}")
        
        try:
            query_vector = self.vectorizer.transform([processed_query])
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            query_lower = query_normalized
            active_filters = self._extract_filters(query_normalized)
            has_additional_filter = len(active_filters) > 0
            
            similarity_scores, matched_category, strict_mode = self._apply_category_matching(
                similarity_scores, query_expanded, has_additional_filter
            )
            
            similarity_scores = self._apply_location_boost(similarity_scores, active_filters)
            similarity_scores = self._apply_content_boost(similarity_scores, query_lower)
            
            similarity_scores, is_murah, is_sedang, is_mahal = self._apply_price_boost(
                similarity_scores, query_lower, price_filter
            )
            
            detected_price = "Murah" if is_murah else ("Sedang" if is_sedang else ("Mahal" if is_mahal else None))
            if matched_category and detected_price:
                self._apply_perfect_match_boost(similarity_scores, matched_category, active_filters, detected_price)
            
        except Exception as e:
            raise Exception(f"Error menghitung similarity: {str(e)}")
        
        similarity_scores = self._apply_exact_name_matching(similarity_scores, query)
        
        try:
            result_df = self.df.copy()
            result_df['similarity_score'] = similarity_scores
            
            top_recommendations = result_df.nlargest(top_n, 'similarity_score')
            top_recommendations = top_recommendations[top_recommendations['similarity_score'] > 0]
            
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
            
            # PENGGILAN UPDATE: Sertakan active_filters
            warning_msg = self._generate_warning_message(
                top_recommendations, is_murah, is_sedang, is_mahal, query, matched_category, active_filters
            )
            
            return top_recommendations, warning_msg, query
            
        except Exception as e:
            raise Exception(f"Error memproses hasil rekomendasi: {str(e)}")

    def _apply_perfect_match_boost(self, similarity_scores, matched_category, active_filters, price_filter):
        """Memberikan boost besar untuk perfect match"""
        cat_mask = pd.Series([False] * len(self.df), index=self.df.index)
        
        if any(x in matched_category for x in ['kopi', 'cafe', 'kafe', 'coffee', 'dessert']):
             cat_mask = self.df['kategori'].astype(str).str.lower().str.contains(
                'kopi|cafe|kafe|coffee|dessert', na=False, regex=True
            )
        else:
            cat_mask = self.df['kategori'].astype(str).str.lower() == matched_category
            
        price_mask = pd.Series([False] * len(self.df), index=self.df.index)
        query_lower = price_filter.lower()
        
        if any(k in query_lower for k in ['murah', 'terjangkau', 'hemat', 'low budget']):
            price_mask = self.df['kategori_harga'] == 'Murah'
        elif any(k in query_lower for k in ['sedang', 'standar', 'menengah']):
            price_mask = self.df['kategori_harga'] == 'Sedang'
        elif any(k in query_lower for k in ['mahal', 'premium', 'mewah']):
             price_mask = self.df['kategori_harga'] == 'Mahal'
             
        if active_filters and len(active_filters) > 0:
            loc_mask = pd.Series([False] * len(self.df), index=self.df.index)
            for flt in active_filters:
                search_terms = LOCATION_EXPANSION.get(flt, [flt])
                for term in search_terms:
                    term_mask = self.df['alamat'].astype(str).str.lower().str.contains(term, na=False)
                    loc_mask = loc_mask | term_mask
        else:
            loc_mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        perfect_mask = cat_mask & price_mask & loc_mask
        
        if perfect_mask.any():
            similarity_scores[perfect_mask] += 50.0
            
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
