import pandas as pd


# Path file
csv_path = r'C:\Users\Zidhan Maula Fatih\Kuliah\chatbot-kuliner\dataset\data-test.csv'

def generate_clean_metadata(row):
    """
    Menggabungkan kolom-kolom penting menjadi satu string metadata.
    Fokus pada data yang membedakan satu tempat dengan tempat lain.
    """
    
    # Ambil data dari kolom, ganti NaN dengan string kosong
    def get_val(col):
        val = row[col]
        if pd.isna(val) or str(val).lower() == 'nan':
            return ""
        return str(val).strip()

    # Komponen Metadata
    # 1. Nama Tempat (Penting untuk pencarian by name)
    nama = get_val('nama_rumah_makan')
    
    # 2. Kategori (Utama)
    kategori = get_val('kategori')
    
    # 3. Menu (Penting untuk pencarian makanan spesifik)
    menu = get_val('menu')
    
    # 4. Details (Suasana, Fasilitas, Pengunjung)
    suasana = get_val('suasana')
    fasilitas = get_val('fasilitas')
    tipe_pengunjung = get_val('tipe_pengunjung')
    
    # 5. Lokasi & Harga
    alamat = get_val('alamat')
    harga = get_val('kategori_harga') # Murah/Sedang/Mahal
    
    # 6. Deskripsi (Narasi tambahan)
    deskripsi = get_val('deskripsi')
    
    # Gabungkan semua (Space separated)
    # Urutan tidak terlalu berpengaruh untuk TF-IDF
    pieces = [nama, kategori, menu, suasana, fasilitas, tipe_pengunjung, alamat, harga, deskripsi]
    
    # Filter string kosong dan gabung
    full_text = " ".join([p for p in pieces if p])
    
    return full_text

def main():
    try:
        print("[INFO] Membaca dataset...")
        # Coba encoding berbeda jika default gagal
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='ISO-8859-1')
            
        print(f"[INFO] Dataset dimuat. Total baris: {len(df)}")
        
        # Simpan backup dulu jaga-jaga
        # df.to_csv(csv_path + ".bak", index=False)
        
        print("[INFO] Sedang men-generate ulang kolom 'metadata_tfidf'...")
        df['metadata_tfidf'] = df.apply(generate_clean_metadata, axis=1)
        
        # Validasi hasil kosong
        empty_count = df[df['metadata_tfidf'].str.strip() == ''].shape[0]
        if empty_count > 0:
            print(f"[WARNING] Ada {empty_count} baris yang metadatanya kosong!")
        
        print("[INFO] Menyimpan perubahan ke CSV...")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print("\n[SUCCESS] Selesai! Kolom metadata_tfidf telah diperbarui.")
        print("-" * 50)
        print("SAMPLE DATA BARU (Baris Pertama):")
        print(df['metadata_tfidf'].iloc[0])
        print("-" * 50)
        
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
