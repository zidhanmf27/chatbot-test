import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import urllib.parse
from chatbot_engine import ChatbotEngine


# --- Konfigurasi Awal Halaman ---
st.set_page_config(
    page_title="Chatbot Kuliner UMKM Kota Bandung",
    page_icon="style/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fungsi Loading Resource ---
@st.cache_resource(show_spinner=False)
def load_chatbot_v2(dataset_path):
    """Memuat instance chatbot engine dengan caching agar tidak di-reload setiap interaksi"""
    return ChatbotEngine(dataset_path)

def load_css(file_name):
    """Memuat file CSS eksternal untuk styling khusus"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Memuat Style
load_css('style/app.css')

# Memuat Font Awesome untuk Ikon
st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<div id="top-of-page"></div>
""", unsafe_allow_html=True)

# --- Inisialisasi State Aplikasi (Session Management) ---
if 'chatbot' not in st.session_state:
    dataset_path = os.path.join('dataset', 'data-kuliner-umkm-optimized.csv')
    try:
        if not os.path.exists(dataset_path):
            st.error("Dataset not found!")
            st.stop()
        st.session_state.chatbot = load_chatbot_v2(dataset_path)
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'show_scroll_btn' not in st.session_state:
            st.session_state.show_scroll_btn = False
    except Exception as e:
        st.error(f"Error loading chatbot: {e}")
        st.stop()

# Inisialisasi tema di session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'



# --- Tampilan Sidebar ---
with st.sidebar:
    try:
        total_umkm = len(st.session_state.chatbot.df)
    except:
        total_umkm = 0
        
    st.markdown(f"""
    <div class="sidebar-header"><i class="fas fa-database"></i> DATA SISTEM</div>
    <div class="data-system-card">
        <span class="stat-label">TOTAL UMKM</span>
        <span class="stat-value">{total_umkm}</span>
        <div class="stat-status">
            <div class="status-dot"></div>
            <span>Database Aktif</span>
        </div>
        <i class="fas fa-store" style="position: absolute; right: -10px; bottom: -10px; font-size: 5rem; opacity: 0.1; color: white;"></i>
    </div>
    """, unsafe_allow_html=True)

    # Toggle Tema
    st.markdown('<div class="sidebar-header"><i class="fas fa-palette"></i> TEMA</div>', unsafe_allow_html=True)
    
    # Toggle switch untuk tema
    theme_options = {"🌙 Dark Mode": "dark", "☀️ Light Mode": "light"}
    selected_theme_label = st.radio(
        "Pilih Tema",
        options=list(theme_options.keys()),
        index=0 if st.session_state.theme == 'dark' else 1,
        key="theme_selector",
        label_visibility="collapsed"
    )
    
    # Deteksi perubahan tema
    new_theme = theme_options[selected_theme_label]
    theme_changed = st.session_state.theme != new_theme
    
    # Update tema di session state
    st.session_state.theme = new_theme
    
    # Inject JavaScript untuk mengubah tema secara real-time
    import time
    timestamp = int(time.time() * 1000)  # milliseconds untuk uniqueness
    components.html(
        f"""
        <script>
            (function() {{
                const theme = '{st.session_state.theme}';
                const htmlElement = window.parent.document.documentElement;
                const currentTheme = htmlElement.getAttribute('data-theme');
                
                if (currentTheme !== theme) {{
                    htmlElement.setAttribute('data-theme', theme);
                    console.log('Theme changed from', currentTheme, 'to', theme, 'at', {timestamp});
                }}
            }})();
        </script>
        """,
        height=0
    )
    
    # Rerun jika tema berubah untuk memastikan semua elemen terupdate
    if theme_changed:
        st.rerun()


    # --- Bagian Preferensi Harga (Collapsible) ---
    st.markdown('<div class="sidebar-header"><i class="fas fa-dollar-sign"></i></div>', unsafe_allow_html=True)
    with st.expander("Preferensi Harga", expanded=True):
        selected_price = st.radio(
            "RANGE HARGA",
            ["Semua", "Murah", "Sedang", "Mahal"],
            index=0,
            key="price_filter"
        )

    # --- Bagian Kategori Kuliner (Collapsible) ---
    st.markdown('<div class="sidebar-header"><i class="fas fa-utensils"></i></div>', unsafe_allow_html=True)
    with st.expander("KATEGORI KULINER", expanded=True):
        if total_umkm > 0:
            top_cats = st.session_state.chatbot.df['kategori'].value_counts().head(10)
            st.markdown("<div class='sidebar-cat-list'>", unsafe_allow_html=True)
            for cat, count in top_cats.items():
                st.markdown(f"<div class='sidebar-cat-item'><span>{cat}</span><span>{count}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("""
    <div class="tips-card">
        <div class="tips-header">
            <i class="fas fa-lightbulb"></i> TIPS PENCARIAN
        </div>
        <ul class="tips-list">
            <li>Gunakan kata kunci spesifik.</li>
            <li>Contoh: "kopi murah di dago".</li>
            <li>Bot akan memberi top 5 hasil.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# # Hero Section
# import base64

# def get_img_as_base64(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# try:
#     icon_base64 = get_img_as_base64("style/icon.png")
#     icon_img_tag = f'<img src="data:image/png;base64,{icon_base64}" alt="Icon" class="hero-icon" style="width: 100px; height: 100px; object-fit: contain;">'
# except Exception:
#     icon_img_tag = '<i class="fas fa-utensils"></i>' # Fallback jika gambar gagal load

# # Hero Section
# st.markdown(f"""
# <div class="hero-container">
#     <div class="hero-icon">
#         {icon_img_tag}
#     </div>
#     <div class="hero-title">
#         Jelajahi Rasa<br>
#         <span class="hero-title-blue">Kota Bandung</span>
#     </div>
#     <div class="hero-subtitle">
#         Temukan kuliner terbaik dengan bantuan Chatbot.
#     </div>
# </div>
# """, unsafe_allow_html=True)

# --- Bagian Utama (Main Content) ---
# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">
        <i class="fas fa-utensils"></i>
    </div>
    <div class="hero-title">
        Jelajahi Rasa<br>
        <span class="hero-title-blue">Kota Bandung</span>
    </div>
    <div class="hero-subtitle">
        Temukan kuliner terbaik dengan bantuan Chatbot.
    </div>
</div>
""", unsafe_allow_html=True)

# Form Pencarian Utama
with st.form(key='search_form'):
    st.markdown("""
    <div class="form-header">
        <div class="bot-avatar"><i class="fas fa-robot"></i></div>
        <div class="form-title-group">
            <h3>Mulai Percakapan</h3>
            <p>Apa kuliner yang Anda cari?</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    user_input_val = st.text_input("Search", placeholder="Contoh: Kopi Murah di Dago", label_visibility="collapsed")
    st.markdown('<div class="kirim-container">', unsafe_allow_html=True)
    submitted = st.form_submit_button("Kirim", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 0.75rem; font-size: 0.9rem; font-weight: 600; color: var(--text-secondary); display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
        <i class="fas fa-bolt" style="color: #eab308;"></i> Pencarian Cepat
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="quick-search-container">', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        quick_kopi = st.form_submit_button("Kopi", type="secondary", use_container_width=True)
    with q2:
        quick_ramen = st.form_submit_button("Ramen", type="secondary", use_container_width=True)
    with q3:
        quick_sunda = st.form_submit_button("Masakan Sunda", type="secondary", use_container_width=True)
    with q4:
        quick_roti = st.form_submit_button("Roti", type="secondary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- Logika Pemrosesan Query ---
final_query = None

if 'temp_query' in st.session_state and st.session_state.temp_query:
    final_query = st.session_state.temp_query
    del st.session_state.temp_query
elif submitted and user_input_val:
    final_query = user_input_val
elif quick_kopi:
    final_query = "kopi murah"
elif quick_ramen:
    final_query = "ramen pedas"
elif quick_sunda:
    final_query = "masakan sunda"
elif quick_roti:
    final_query = "toko roti"

if final_query:
    st.session_state.show_scroll_btn = False # Reset tombol setiap kali searching baru
    st.session_state.messages.append({"role": "user", "content": final_query})
    
    price_map = {"Semua": "Semua", "Murah": "Murah", "Sedang": "Sedang", "Mahal": "Mahal"}
    backend_price = price_map.get(selected_price, "Semua")
    
    try:
        with st.spinner('Sedang mencari rekomendasi kuliner...'):
            recommendations = st.session_state.chatbot.get_recommendations(
                final_query, 
                price_filter=backend_price,
                top_n=50
            )
        
        st.session_state.messages.append({
            "role": "bot",
            "content": final_query,
            "full_recommendations": recommendations,
            "display_count": 5
        })
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

# --- Tampilan Hasil Rekomendasi ---
if len(st.session_state.messages) > 0:
    st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True) 
    
    for idx, message in enumerate(st.session_state.messages):
        if message['role'] == 'user':
            pass
        else:
            # Tambahkan garis pemisah sebelum hasil pencarian kedua dan seterusnya
            if idx > 1:  # idx > 1 karena idx 0 adalah user message pertama, idx 1 adalah bot response pertama
                st.markdown("""
                <div style="margin: 2rem 0; display: flex; align-items: center; gap: 1rem;">
                    <div style="flex: 1; height: 2px; background: linear-gradient(to right, transparent, var(--accent-blue), transparent); opacity: 0.5;"></div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.5rem; background: var(--card-bg); border: 2px solid var(--accent-blue); border-radius: 50px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);">
                        <i class="fas fa-search" style="color: var(--accent-blue); font-size: 1rem;"></i>
                        <span style="color: var(--text-primary); font-weight: 600; font-size: 0.9rem;">Pencarian Baru</span>
                    </div>
                    <div style="flex: 1; height: 2px; background: linear-gradient(to left, transparent, var(--accent-blue), transparent); opacity: 0.5;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="results-container">
                <div class="results-header-container">
                    <h3 class="results-header-title"><i class="fas fa-check-circle" style="color:var(--success);"></i> Berikut Rekomendasi untuk "{message['content']}"</h3>
                </div>
            """, unsafe_allow_html=True)
            
            full_recs = message.get('full_recommendations')
            display_count = message.get('display_count', 5)
            
            if full_recs is not None and not full_recs.empty:
                current_view = full_recs.iloc[:display_count]
                
                for _, row in current_view.iterrows():
                    similarity = row['similarity_score'] * 100
                    
                    def get_category_icon(cat_name):
                        cat_lower = str(cat_name).lower()
                        if any(x in cat_lower for x in ['kopi', 'cafe', 'kafe', 'coffee']): return "fa-mug-hot"
                        if any(x in cat_lower for x in ['jepang', 'sushi', 'ramen', 'udon']): return "fa-fish"
                        if any(x in cat_lower for x in ['sunda', 'khas', 'tradisional']): return "fa-leaf"
                        if any(x in cat_lower for x in ['western', 'steak', 'burger', 'pizza', 'pasta']): return "fa-burger"
                        if any(x in cat_lower for x in ['roti', 'bakery', 'kue', 'cake', 'donat']): return "fa-bread-slice"
                        if any(x in cat_lower for x in ['ayam', 'bebek', 'geprek', 'fried chicken']): return "fa-drumstick-bite"
                        if any(x in cat_lower for x in ['mie', 'bakso', 'soto', 'sop', 'kuah']): return "fa-bowl-food"
                        if any(x in cat_lower for x in ['minuman', 'jus', 'thai tea', 'bobba']): return "fa-glass-water"
                        if any(x in cat_lower for x in ['nasi', 'padang', 'warteg']): return "fa-utensils"
                        if any(x in cat_lower for x in ['pedas', 'sambal']): return "fa-fire"
                        if 'pedas' in cat_lower: return "fa-fire"
                        return "fa-utensils"

                    icon_class = get_category_icon(row['kategori'])
                    
                    # Generate Google Maps URL
                    maps_query = urllib.parse.quote(f"{row['nama_rumah_makan']} {row['alamat']}")
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query}"
                    
                    st.markdown(f"""
<div class="recommendation-card">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.5rem;">
        <h4 style="margin:0;"><i class="fas {icon_class}"></i> {row['nama_rumah_makan']}</h4>
        <span class="card-match-badge">{similarity:.0f}% Match</span>
    </div>
    <p><i class="fas fa-map-marker-alt icon-fixed-width"></i> <strong>Alamat:</strong> {row['alamat']}</p>
    <p><i class="fas fa-tag icon-fixed-width"></i> <strong>Kategori:</strong> {row['kategori']}</p>
    <p><i class="fas fa-money-bill icon-fixed-width"></i> <strong>Harga:</strong> {row['range_harga']} ({row['kategori_harga']})</p>
    <p><i class="fas fa-utensils icon-fixed-width"></i> <strong>Menu:</strong> {row['menu']}</p>
    <p><i class="fas fa-comments icon-fixed-width"></i> <strong>Deskripsi:</strong> {row['deskripsi']}</p>
    <div style="margin-top: 1rem;">
        <a href="{maps_url}" target="_blank" style="text-decoration: none; display: block;">
            <div style="background: linear-gradient(90deg, #2563eb, #3b82f6); color: white; padding: 0.6rem 1rem; border-radius: 10px; text-align: center; font-weight: 600; font-size: 0.9rem; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <i class="fas fa-location-dot"></i> Lihat Lokasi di Google Maps
            </div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)


                
                if len(full_recs) > display_count:
                    st.markdown('<div class="load-more-wrapper">', unsafe_allow_html=True)
                    if st.button(f"Lebih Banyak ({len(full_recs) - display_count})", key=f"more_{idx}"):
                        message['display_count'] += 5
                        st.session_state.show_scroll_btn = True
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True) 
            else:
                st.markdown("""
                <div style="max-width: 820px; width: 95%; margin: 0 auto; padding: 0.75rem 1.25rem; background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; color: #ef4444; display: flex; align-items: center; gap: 0.75rem; font-weight: 500;">
                    <i class="fas fa-circle-xmark" style="font-size: 1.1rem;"></i> Tidak ada rekomendasi yang ditemukan untuk kriteria ini.
                </div>
                """, unsafe_allow_html=True)

# --- Bagian Pencarian Ulang di Bawah ---
if len(st.session_state.messages) > 0:
    st.markdown("<div style='text-align:center; margin-top:1rem; margin-bottom:0.5rem; font-size:1rem; color:var(--text-secondary);'>Ingin mencari yang lain?</div>", unsafe_allow_html=True)
    st.markdown('<div class="bottom-search-wrapper">', unsafe_allow_html=True)
    with st.form(key='search_form_bottom'):
        c_b1, c_b2 = st.columns([4, 1], gap="medium")
        with c_b1:
            user_input_bottom = st.text_input("Search Bottom", placeholder="Cari kuliner lain...", label_visibility="collapsed")
        with c_b2:
            st.markdown('<div class="cari-container-bottom">', unsafe_allow_html=True)
            submit_bottom = st.form_submit_button("Cari", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_bottom and user_input_bottom:
        st.session_state.temp_query = user_input_bottom
        st.rerun()

# --- Fitur Hapus Riwayat ---
if len(st.session_state.messages) > 0:
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown('<div class="hapus-container">', unsafe_allow_html=True)
        if st.button("Hapus Riwayat Chat", type="primary", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- Scroll to Top Button (Conditional Render) ---
# Tombol HANYA muncul jika ada output rekomendasi yang DITAMPILKAN lebih dari 5
# (Artinya user sudah mengklik 'Lebih Banyak', sehingga display_count > 5)
# --- Scroll to Top Button (Conditional Render) ---
# Tombol hanya muncul jika user telah mengklik 'Lebih Banyak' (status disimpan di session_state)
if st.session_state.get('show_scroll_btn', False):
    st.markdown("""
    <a href="#top-of-page" class="scroll-to-top-btn" title="Kembali ke atas">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="19" x2="12" y2="5"></line>
            <polyline points="5 12 12 5 19 12"></polyline>
        </svg>
    </a>
    """, unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<div class="footer-text">
    <p><i class="fas fa-gear"></i> Metode TF-IDF dan Cosine Similarity</p>
    <p style="margin-top:0.25rem;"><i class="fas fa-database"></i> Sumber: <a href="https://opendata.bandung.go.id/dataset/data-rumah-makan-restoran-cafe-di-kota-bandung" target="_blank">Open Data Bandung</a> &nbsp;|&nbsp; Dinas Kebudayaan dan Pariwisata </p>
    <p style="margin-top:0.5rem; opacity:0.8;"><i class="fas fa-code"></i> Developed by <a href="https://zidhanmf-portofolio.vercel.app/" target="_blank" style="color: var(--accent-blue); font-weight: 600;">zidhanmf</a></p>
</div>
""", unsafe_allow_html=True)
