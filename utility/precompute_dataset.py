# Script Optimasi: Melakukan preprocessing berat (stemming) di awal dan menyimpan hasilnya agar aplikasi berjalan instan.

import pandas as pd
import os
import sys

# Tambahkan root directory ke path agar bisa import preprocessing.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from preprocessing import TextPreprocessor

def optimize_dataset():
    """Melakukan preprocessing berat (stemming) dan menyimpan ke CSV baru"""
    print("[INFO] Memulai optimasi dataset...")
    
    # Gunakan path absolut agar aman dijalankan dari folder manapun
    input_path = os.path.join(root_dir, 'dataset', 'data-test.csv')
    output_path = os.path.join(root_dir, 'dataset', 'data-test-optimized.csv')
    
    # 1. Load Dataset Original
    try:
        print(f"[INFO] Membaca file: {input_path}")
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"[ERROR] Gagal membaca dataset: {e}")
        return

    # 2. Inisialisasi Preprocessor
    try:
        preprocessor = TextPreprocessor()
        print("[INFO] Preprocessor siap.")
    except Exception as e:
        print(f"[ERROR] Gagal inisialisasi preprocessor: {e}")
        return

    # 3. Lakukan Preprocessing (Stemming Berat)
    print("[INFO] Sedang melakukan stemming pada 500+ data (Bisa memakan waktu 30-60 detik)...")
    
    # Pastikan kolom metadata_tfidf ada
    df['metadata_tfidf'] = df['metadata_tfidf'].fillna('').astype(str)
    
    # Proses kolom metadata
    df['metadata_tfidf_processed'] = df['metadata_tfidf'].apply(
        lambda x: preprocessor.preprocess(x)
    )
    
    # 4. Simpan ke File Baru
    print(f"[INFO] Menyimpan hasil optimasi ke: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print("-" * 50)
    print("[SUCCESS] Dataset berhasil dioptimasi!")
    print("Sekarang 'chatbot_engine.py' bisa memuat file '-optimized.csv' dengan instan.")
    print("-" * 50)

if __name__ == "__main__":
    optimize_dataset()
