# Logic Utama Chatbot: Menangani loading data, 
# perhitungan TF-IDF, Cosine Similarity, 
# dan logika rekomendasi (Strict/Flexible).

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import TextPreprocessor
from difflib import get_close_matches

class ChatbotEngine:
    """
    Mesin utama chatbot yang mengatur logika rekomendasi.
    Menggunakan metode TF-IDF Vectorization untuk mengubah teks menjadi angka,
    dan Cosine Similarity untuk mengukur kemiripan antar teks.
    """
    
    def __init__(self, csv_path):
        """Inisialisasi chatbot dengan memuat data dan membuat TF-IDF matrix"""
        import os

        # Memuat dataset
        try:
            # Load dataset utama
            try:
                self.df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(csv_path, encoding='ISO-8859-1')
            
            if self.df.empty:
                raise ValueError("Dataset kosong! Tidak ada data untuk diproses.")
            
            # Isi nilai kosong di kolom penting
            self.df['metadata_tfidf'] = self.df['metadata_tfidf'].fillna('')
            self.df['deskripsi'] = self.df['deskripsi'].fillna('')
            
            if self.df['metadata_tfidf'].str.strip().eq('').all():
                raise ValueError("[ERROR] Kolom metadata_tfidf kosong! Periksa dataset Anda.")

            # OTOMATISASI OPTIMASI: 
            # Jika kolom 'metadata_tfidf_processed' sudah ada, berarti file sudah ter-stemming.
            bypass_processing = 'metadata_tfidf_processed' in self.df.columns
            
        except Exception as e:
             raise Exception(f"Error loading dataset: {str(e)}")
        
        # Inisialisasi Preprocessor: Menyiapkan alat pembersih teks
        try:
            self.preprocessor = TextPreprocessor()
        except Exception as e:
            raise Exception(f"Error inisialisasi preprocessor: {str(e)}")
        
        # Preprocessing Data: Memproses data UMKM sebelum digunakan
        try:
            if bypass_processing:
                # Mode Cepat: Gunakan hasil stemming yang sudah ada di CSV
                self.df['metadata_tfidf_processed'] = self.df['metadata_tfidf_processed'].fillna('')
                print("[SUCCESS] Dataset teroptimasi ditemukan! Lewati stemming manual.")
            else:
                # Mode Normal/Lambat: Lakukan stemming manual (jika file baru ditambahkan)
                print("[INFO] Dataset belum teroptimasi. Melakukan preprocessing awal...")
                self.df['metadata_tfidf_original'] = self.df['metadata_tfidf'].copy()
                self.df['metadata_tfidf_processed'] = self.df['metadata_tfidf'].apply(
                    lambda x: self.preprocessor.preprocess(str(x))
                )
                print("[SUCCESS] Preprocessing dataset selesai!")
            
            # Membangun vocabulary untuk koreksi typo
            print("[INFO] Membangun vocabulary untuk koreksi otomatis...")
            all_text = " ".join(self.df['metadata_tfidf_processed'].astype(str).tolist())
            self.vocabulary = set(all_text.split())
            print(f"[INFO] Ukuran Vocabulary: {len(self.vocabulary)} kata unik")
            
        except Exception as e:
            raise Exception(f"Error preprocessing dataset: {str(e)}")
        
        # Konfigurasi TF-IDF: Membuat matriks pembobotan kata
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
        
        query = query.strip()
        if not query:
            raise ValueError("Query tidak boleh kosong atau hanya spasi!")
        
        try:
            # Preprocessing Query: Membersihkan input pengguna (huruf kecil, hapus simbol, stemming)
            processed_query = self.preprocessor.preprocess(query)
            
            # VALIDASI TAMBAHAN: Abaikan query 1 huruf (misal 'a')
            if len(processed_query.strip()) < 2:
                print(f"[INFO] Query '{processed_query}' diabaikan karena terlalu pendek.")
                return pd.DataFrame() # Kembalikan DataFrame kosong agar tidak ada hasil
                
            if not processed_query.strip():
                return pd.DataFrame()

            # --- NORMALISASI SINONIM (SEBELUM AUTO-CORRECT) ---
            # Mengubah kata gaul/singkatan menjadi istilah baku di database
            # Contoh: 'cina' -> 'chinese food', 'indo' -> 'masakan indonesia'
            # Mencegah kata seperti 'cina' dikoreksi menjadi 'cinta'
            synonym_map = {
                # Kategori Makanan Dasar
                'makanan': 'masakan', 'makan': 'masakan',
                
                # Tempat Makan (Variasi Nama)
                'kafe': 'kafe/kedai kopi', 'cafe': 'kafe/kedai kopi', 
                'kedai kopi': 'kafe/kedai kopi', 'coffee': 'kafe/kedai kopi',
                'coffe shop': 'kafe/kedai kopi', 'kopi': 'kafe/kedai kopi',
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
                    matches = get_close_matches(word, self.vocabulary, n=1, cutoff=0.90)
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
                'kafe': 'kafe/kedai kopi', 'cafe': 'kafe/kedai kopi', 
                'kedai kopi': 'kafe/kedai kopi', 'coffee': 'kafe/kedai kopi',
                'coffe shop': 'kafe/kedai kopi', 'kopi': 'kafe/kedai kopi',
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
                        category_mask = self.df['kategori'].astype(str).str.lower() == matched_category
                        similarity_scores[category_mask] += 0.5
                    
                    if matched_tipe_pengunjung:
                        tipe_mask = self.df['tipe_pengunjung'].astype(str).str.lower().str.contains(matched_tipe_pengunjung, na=False, regex=False)
                        similarity_scores[tipe_mask] += 0.5
                    
                    # Boosting Lokasi: Memberikan prioritas pada lokasi yang cocok
                    if len(final_active_filters) > 0:
                        for flt in final_active_filters:
                            addr_mask = self.df['alamat'].astype(str).str.lower().str.contains(flt, na=False)
                            if addr_mask.any():
                                print(f"[BOOST] Location match: '{flt}'")
                                similarity_scores[addr_mask] += 2.0
            
            # --- BOOSTING HARGA: Menyesuaikan skor berdasarkan preferensi harga ---
            BOOST_FACTOR = 2.0
            
            is_murah = any(k in query_lower for k in ['murah', 'terjangkau', 'hemat', 'low budget'])
            is_sedang = any(k in query_lower for k in ['sedang', 'standar', 'menengah', 'reasonable'])
            is_mahal = any(k in query_lower for k in ['mahal', 'premium', 'mewah', 'fancy'])
            
            if price_filter and price_filter != "Semua":
                if price_filter == "Murah": is_murah, is_sedang, is_mahal = True, False, False
                elif price_filter == "Menengah": is_murah, is_sedang, is_mahal = False, True, False
                elif price_filter == "Mahal": is_murah, is_sedang, is_mahal = False, False, True
            
            if not strict_mode_activated:
                if is_murah:
                    mask = self.df['kategori_harga'].astype(str).str.contains('Murah', case=False, na=False)
                    similarity_scores[mask] *= BOOST_FACTOR
                elif is_mahal:
                    mask = self.df['kategori_harga'].astype(str).str.contains('Mahal', case=False, na=False)
                    similarity_scores[mask] *= BOOST_FACTOR
                elif is_sedang:
                    mask = self.df['kategori_harga'].astype(str).str.contains('Sedang', case=False, na=False)
                    similarity_scores[mask] *= BOOST_FACTOR
            
        except Exception as e:
            raise Exception(f"Error menghitung similarity: {str(e)}")
        
        try:
            result_df = self.df.copy()
            result_df['similarity_score'] = similarity_scores
            
            top_recommendations = result_df.nlargest(top_n, 'similarity_score')
            top_recommendations = top_recommendations[top_recommendations['similarity_score'] > 0]
            
            # Mekanisme Cadangan (Fallback): Pencarian kata kunci manual jika TF-IDF gagal
            if top_recommendations.empty:
                print(f"[INFO] Fallback search for: {query}")
                keyword = query.lower()
                
                # Cegah pencarian fallback untuk kata kunci terlalu pendek (misal 'a')
                if len(keyword) < 3:
                    print(f"[INFO] Fallback dilewati: Kata kunci harus minimal 3 karakter.")
                    mask = pd.Series([False] * len(self.df))
                else:
                    mask = self.df['metadata_tfidf'].str.lower().str.contains(keyword, na=False)
                
                if mask.any():
                    fallback_df = self.df[mask].copy()
                    fallback_df['similarity_score'] = 0.5
                    top_recommendations = fallback_df.head(top_n)
                    print(f"[SUCCESS] Fallback found {len(top_recommendations)} results.")
            
            return top_recommendations
            
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