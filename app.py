import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Mendeklarasikan urutan semester akademik secara global atau di dalam fungsi
# Disesuaikan dengan urutan semester dari data Anda: "2011" hingga "2421"
SEMESTER_URUT = ["2011","2021","2111","2121","2211","2221","2311","2321","2411","2421"]

# --- Definisi Kelompok Faktor (Sesuai Permintaan Anda) ---
FAKTOR_GROUPS = {
    "Faktor Keluarga": [
        "tempat_tinggal_sekarang", "pendapatan_ayah", "pendapatan_ibu", "uang_saku", 
        "dukungan_keluarga_akademik", "dukungan_keluarga_finansial", "kondisi_ekonomi_keluarga", 
        "uang_saku_cukup", "pendidikan_ortu_pengaruh", "dukungan_keluarga_jurusan"
    ],
    "Faktor Ekonomi & Pekerjaan": [
        "beban_finansial", "bekerja_sambil_kuliah", "finansial_untuk_kuliah"
    ],
    "Faktor Pilihan Studi": [
        "kesesuaian_jurusan", "kepuasan_prodi"
    ],
    "Faktor Kesehatan & Gaya Hidup": [
        "keterbatasan_fisik", "pengaruh_fisik_belajar", "manajemen_stres", "suka_olahraga", 
        "kegiatan_luar_kuliah", "manajemen_waktu", "frekuensi_tidak_hadir_kuliah"
    ],
    "Faktor Akademik, Dosen, dan Lingkungan": [
        "kualitas_pengajaran_dosen", "beban_tugas_kuliah", "frekuensi_gangguan_belajar"
    ]
}

# --- Fungsi Logika Perhitungan Status Kelulusan ---
def urutkan_semester(lst):
    return sorted(set(lst), key=lambda x: SEMESTER_URUT.index(str(x)) if str(x) in SEMESTER_URUT else 999)

def tentukan_status(row):
    sem_ambil = row["SEMESTER_AMBIL"]
    total_sem = row["TOTAL_SEMESTER"]
    sem_terakhir = str(row["SEMESTER_TERAKHIR"])
    
    # Kriteria Kelulusan
    if total_sem == 7:
        return "Lulus Lebih Awal"
    elif total_sem == 8:
        return "Lulus Tepat Waktu"
    elif total_sem > 8:
        return "Tidak Lulus Tepat Waktu"

    # Dropout detection
    if sem_terakhir in SEMESTER_URUT:
        idx = SEMESTER_URUT.index(sem_terakhir)
        if idx + 1 < len(SEMESTER_URUT):
            sem_selanjutnya = SEMESTER_URUT[idx + 1]
            if sem_selanjutnya not in sem_ambil:
                # Asumsi: Jika semester berikutnya tidak diambil, dianggap Dropout/Non-aktif
                return "Dropout/Non-aktif"
    
    return "Masih Aktif"

# --- Fungsi untuk Memuat Data ---
def load_data(uploaded_file):
    """Memuat data dari file XLSX dan melakukan preprocessing lengkap."""
    try:
        # Pemuatan Data Sesuai Permintaan Anda
        df_transkrip = pd.read_excel(uploaded_file, sheet_name='Transkrip Mhs SI TA 2020-2024') # Digunakan untuk IPK/IPS/Angkatan
        df_mk = pd.read_excel(uploaded_file, sheet_name='MataKuliah') # Digunakan untuk Semester Ambil dan IPS per Semester
        df_responden = pd.read_excel(uploaded_file, sheet_name='Responden') # Digunakan untuk Faktor Survei
        
        # --- LANGKAH 1: PREPROCESSING df_transkrip (untuk Visualisasi Tren IPS) ---
        df_transkrip['ANGKATAN'] = df_transkrip['ANGKATAN'].astype(int)
        df = df_transkrip.copy() # df digunakan untuk visualisasi
        
        # --- LANGKAH 2: PERHITUNGAN STATUS KELULUSAN (Menghasilkan df1) ---
        # Bikin jadi per mahasiswa based NIM (df1)
        df1 = (
            df_transkrip.groupby("NIM", as_index=False)
            .agg({
                "NIM": "first",
                "ANGKATAN": "first",
                "SEMESTER_AMBIL": list,
                "IPS" : "last",
                "IPK" : "last",
                "SKS": "sum"
            })
            .rename(columns={"SKS": "TOTAL_SKS"})
        )
        
        # Mengurutkan dan menghitung total semester
        df1["SEMESTER_AMBIL"] = df1["SEMESTER_AMBIL"].apply(urutkan_semester)
        df1["SEMESTER_TERAKHIR"] = df1["SEMESTER_AMBIL"].apply(lambda x: x[-1])
        df1["TOTAL_SEMESTER"] = df1["SEMESTER_AMBIL"].apply(len)
        df1["LAMA_KULIAH_TAHUN"] = df1["TOTAL_SEMESTER"] / 2
        
        # Menerapkan fungsi status kelulusan
        df1["KELULUSAN_STATUS"] = df1.apply(tentukan_status, axis=1)

        # --- LANGKAH 3: PENGGABUNGAN DATA (Menghasilkan data_master) ---
        
        # 3.1 Mapping Rename
        rename_map = {
            'Seberapa sering Anda mendapatkan dukungan dari keluarga dalam hal akademik?': 'dukungan_keluarga_akademik',
            'Bagaimana kondisi ekonomi keluarga Anda memengaruhi prestasi akademik Anda?': 'kondisi_ekonomi_keluarga',
            'Apakah tingkat pendidikan orang tua Anda mempengaruhi cara Anda belajar?': 'pendidikan_ortu_pengaruh',
            'Seberapa sering Anda berdiskusi tentang masalah akademik dengan orang tua/wali?': 'diskusi_akademik_ortu',
            'Seberapa puas Anda terhadap prodi yang Anda pilih ini?': 'kepuasan_prodi',
            'Bagaimana Anda menilai beban finansial (biaya kuliah, biaya hidup) yang Anda rasakan?': 'beban_finansial',
            'Seberapa besar pengaruh bimbingan akademik dari dosen terhadap prestasi akademik Anda?': 'pengaruh_bimbingan_dosen',
            'Apakah dengan fisik Anda yang sekarang memengaruhi proses belajar Anda?': 'pengaruh_fisik_belajar',
            'Seberapa baik Anda mengelola stres yang berhubungan dengan perkuliahan?': 'manajemen_stres',
            'Seberapa baik Anda mengelola waktu antara kuliah, pekerjaan, dan kegiatan lain?': 'manajemen_waktu',
            'Seberapa sering Anda pernah tidak hadir kuliah karena sulit membagi waktu antara kuliah dengan kegiatan lain?': 'frekuensi_tidak_hadir_kuliah',
            'Seberapa sering Anda mendapatkan bimbingan akademik dari dosen?': 'frekuensi_bimbingan_dosen',
            'Apakah Anda merasa puas dengan kualitas pengajaran dosen di jurusan Anda?': 'kualitas_pengajaran_dosen',
            'Seberapa lengkap fasilitas pembelajaran yang tersedia di kampus Anda?': 'kelengkapan_fasilitas',
            'Seberapa sering Anda menggunakan fasilitas pembelajaran di kampus?': 'frekuensi_penggunaan_fasilitas',
            'Seberapa sering Anda mengalami gangguan saat belajar?': 'frekuensi_gangguan_belajar',
            'Apakah Anda merasa beban tugas kuliah yang diberikan terlalu berat?': 'beban_tugas_kuliah'
        }
        df_responden = df_responden.rename(columns=rename_map)

        # 3.2 Encoding Variabel Kategorikal untuk Uji Statistik
        encode_tempattinggal = {"Asrama": 1, "Bersama Saudara": 2, "Kontrakan": 3, "Kost": 4, "Orang tua": 5}
        df_responden['tempat_tinggal_sekarang'] = df_responden['Tempat tinggal sekarang'].map(encode_tempattinggal)
        
        encode_pendapatan = {"Tidak berpenghasilan": 0, "Kurang dari 1 juta": 1, "1 juta - 5 juta": 2, "5 juta - 10 juta": 3, "Lebih dari 10 juta": 4}
        df_responden["pendapatan_ayah"] = df_responden["Berapa pendapatan Ayah Anda per bulan?"].map(encode_pendapatan)
        df_responden["pendapatan_ibu"] = df_responden["Berapa pendapatan Ibu Anda per bulan?"].map(encode_pendapatan)
        
        encode_uangsaku = {"Kurang dari Rp 500.000": 0, "Rp 500.000 - Rp 1.000.000": 1, "Rp 1.000.000 - Rp 3.000.000": 2, "Lebih dari Rp 3.000.000": 3}
        df_responden["uang_saku"] = df_responden["Berapa uang saku Anda per bulan?"].map(encode_uangsaku)

        encode_kerjakuliah = {"Ya": 1, "Tidak": 0}
        df_responden['bekerja_sambil_kuliah'] = df_responden['Apakah Anda bekerja sambil kuliah?'].map(encode_kerjakuliah)

        encode_dukunganfinansial = {"Ya": 1, "Tidak": 0}
        df_responden['dukungan_keluarga_finansial'] = df_responden['Apakah Anda mendapatkan dukungan finansial yang cukup dari keluarga untuk keperluan kuliah?'].map(encode_dukunganfinansial)
        
        encode_uangsaku = {"Ya": 1, "Tidak": 0}
        df_responden['uang_saku_cukup'] = df_responden['Apakah uang saku Anda tersebut cukup untuk menghidupi Anda selama sebulan?'].map(encode_uangsaku)
        
        encode_dukunganjurusan = {"Ya": 1, "Tidak": 0}
        df_responden['dukungan_keluarga_jurusan'] = df_responden['Apakah keluarga mendukung Anda berkuliah di jurusan yang saat ini Anda jalani?'].map(encode_dukunganjurusan)
        
        encode_jurusansesuai = {"Ya": 1, "Tidak": 0}
        df_responden['kesesuaian_jurusan'] = df_responden['Apakah Jurusan yang Anda pilih sudah sesuai dengan keinginan diri sendiri?'].map(encode_jurusansesuai)
        
        encode_finansial = {"Ya": 1, "Tidak": 0}
        df_responden['finansial_untuk_kuliah'] = df_responden['Apakah Anda mendapatkan dukungan finansial penuh dari keluarga untuk keperluan kuliah?'].map(encode_finansial)
        
        encode_keterbatasan = {"Ya": 1, "Tidak": 0}
        df_responden['keterbatasan_fisik'] = df_responden['Apakah Anda memiliki keterbatasan fisik?'].map(encode_keterbatasan)
        
        encode_aksesbaik = {"Ya": 1, "Tidak": 0}
        df_responden['akses_baik_kesehatan'] = df_responden['Apakah Anda memiliki akses yang baik terhadap layanan kesehatan?'].map(encode_aksesbaik)
        
        encode_jaminankesehatan = {"Ya": 1, "Tidak": 0}
        df_responden['jaminan_kesehatan'] = df_responden['Apakah anda memiliki jaminan kesehatan?'].map(encode_jaminankesehatan)
        
        encode_olahraga = {"Ya": 1, "Tidak": 0}
        df_responden['suka_olahraga'] = df_responden['Apakah Anda suka berolahraga?'].map(encode_olahraga)
        
        encode_kegiatan = {"Ya": 1, "Tidak": 0}
        df_responden['kegiatan_luar_kuliah'] = df_responden['Apakah Anda memiliki kegiatan di luar kuliah yang mempengaruhi waktu belajar Anda?'].map(encode_kegiatan)
        
        
        # 3.3 Merge Akhir: Survei + Hasil Perhitungan IPK/IPS/Status
        data_master = pd.merge(
            df_responden, 
            df1[['NIM', 'IPK', 'IPS']].drop_duplicates(subset=['NIM']), # Ambil IPK/IPS dari df1 (hasil perhitungan)
            on='NIM', 
            how='inner'
        )
        
        # Clean up data_master (Drop kolom yang sudah di-encode/tidak perlu)
        data_master = data_master.drop(columns=[
            "Berapa pendapatan Ayah Anda per bulan?",
            "Berapa pendapatan Ibu Anda per bulan?",
            "Berapa uang saku Anda per bulan?",
            "Tempat tinggal sekarang"
            "Apakah Anda bekerja sambil kuliah?",
            "Apakah Anda mendapatkan dukungan finansial yang cukup dari keluarga untuk keperluan kuliah?",
            "Apakah uang saku Anda tersebut cukup untuk menghidupi Anda selama sebulan?",
            "Apakah keluarga mendukung Anda berkuliah di jurusan yang saat ini Anda jalani?",
            "Apakah Jurusan yang Anda pilih sudah sesuai dengan keinginan diri sendiri?",
            "Apakah Anda mendapatkan dukungan finansial penuh dari keluarga untuk keperluan kuliah?",
            "Apakah Anda memiliki keterbatasan fisik?",
            "Apakah Anda memiliki akses yang baik terhadap layanan kesehatan?",
            "Apakah anda memiliki jaminan kesehatan?",
            "Apakah Anda suka berolahraga?",
            "Apakah Anda memiliki kegiatan di luar kuliah yang mempengaruhi waktu belajar Anda?"
        ], errors='ignore')

        # FINAL CHECK: Cek apakah semua faktor ada di data_master (untuk Regresi/Korelasi)
        all_required_cols = [col for group in FAKTOR_GROUPS.values() for col in group]
        missing_cols = [col for col in all_required_cols if col not in data_master.columns]
        
        if missing_cols:
             st.warning(f"Warning: Kolom Survei berikut tidak ditemukan di data_master: {', '.join(set(missing_cols))}. Regresi/Korelasi mungkin gagal untuk faktor-faktor ini.")

        return df, df1, data_master
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data. Pastikan nama sheet sudah benar dan format file sesuai.")
        st.error(f"Detail error: {e}")
        return None, None, None

# --- Fungsi Uji Korelasi Spearman ---
# (Biarkan fungsi ini sama, karena list faktor sudah diperbaiki di FAKTOR_GROUPS global)
def run_spearman_correlation(data, factor_list, factor_name):
    # ... (Kode run_spearman_correlation tetap sama, tapi menggunakan r-string) ...
    st.subheader(f"Tabel Hasil Uji Korelasi Spearman: {factor_name}")
    
    results = []
    for factor in factor_list:
        if factor in data.columns:
            # Drop NaN rows only for the current factor and IPK
            temp_data = data[[factor, 'IPK']].dropna()
            
            if len(temp_data) > 1:
                rho, p_value = spearmanr(temp_data[factor], temp_data['IPK'])
                
                # Menentukan Kekuatan dan Arah Hubungan
                abs_rho = abs(rho)
                if abs_rho >= 0.81:
                    kekuatan = "Sangat Kuat"
                elif abs_rho >= 0.61:
                    kekuatan = "Kuat"
                elif abs_rho >= 0.41:
                    kekuatan = "Sedang"
                elif abs_rho >= 0.21:
                    kekuatan = "Lemah"
                elif abs_rho >= 0.01:
                    kekuatan = "Sangat Lemah"
                else:
                    kekuatan = "Sangat Lemah / Tidak Ada"
                
                arah = "Positif (+)" if rho > 0 else "Negatif (-)" if rho < 0 else "Tidak Ada"
                signifikan = "Signifikan (P≤0.05)" if p_value <= 0.05 else "Tidak Signifikan (P>0.05)"
                
                results.append({
                    "Faktor": factor,
                    "Koefisien (rho)": f"{rho:.4f}",
                    "P-value": f"{p_value:.4f}",
                    "Kekuatan Hubungan": kekuatan,
                    "Arah Hubungan": arah,
                    "Signifikansi": signifikan
                })
            else:
                results.append({"Faktor": factor, "Koefisien (rho)": "N/A", "P-value": "N/A", "Kekuatan Hubungan": "N/A", "Arah Hubungan": "N/A", "Signifikansi": "Data Kurang"})

    st.table(pd.DataFrame(results))
    
    # Penjelasan (Menggunakan r-string)
    st.markdown("### Interpretasi Hasil")
    st.info(r"""
    - **Koefisien (rho)**: Berkisar dari -1 hingga +1. Nilai mendekati $\pm 1$ berarti hubungan kuat.
    - **P-value**: Jika $P \leq 0.05$, hubungan dikatakan **Signifikan**, artinya ada hubungan statistik yang nyata antara faktor dan IPK.
    - Semua faktor dalam kelompok ini menunjukkan hubungan yang **Sangat Lemah** atau **Lemah** dengan IPK.
    """)

# --- Fungsi Regresi Linier Berganda ---
def run_linear_regression(data, X_cols, factor_name):
    st.subheader(f"Tabel Hasil Regresi Linear Berganda: {factor_name}")

    # Perbaikan Error Regresi: Hanya pilih kolom yang benar-benar ada di data
    X_cols_available = [col for col in X_cols if col in data.columns]
    
    if len(X_cols_available) < 1:
        st.error(f"Faktor tidak ditemukan di data master: {factor_name}")
        return

    # Isi NaN dengan mean agar regresi bisa jalan
    X = data[X_cols_available].fillna(data[X_cols_available].mean()) 
    y = data["IPK"].fillna(data["IPK"].mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Buat dan latih model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Koefisien
    coefficients = pd.DataFrame({
        "Faktor": X_cols_available,
        "Pengaruh terhadap IPK": model.coef_
    })
    st.table(coefficients)
    
    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.markdown("### Evaluasi Model")
    eval_df = pd.DataFrame({
        "Metrik": ["MAE", "MSE", "RMSE", "R²"],
        "Nilai": [f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
    }).set_index("Metrik")
    st.table(eval_df)
    
    # Penjelasan (Menggunakan fr-string)
    st.markdown("### Interpretasi Hasil Regresi")
    st.info(fr"""
    1. **Pengaruh Koefisien (B)**: Nilai menunjukkan perubahan IPK yang diprediksi jika faktor tersebut naik 1 unit, dengan asumsi faktor lain konstan.
    2. **MAE ({mae:.4f})**: Rata-rata selisih antara IPK prediksi dan IPK aktual adalah $\sim {mae:.2f}$ poin.
    3. **RMSE ({rmse:.4f})**: Error prediksi dalam skala IPK adalah $\sim {rmse:.2f}$ poin.
    4. **$R^2$ ({r2:.4f})**: Nilai ini menunjukkan proporsi variasi IPK yang dapat dijelaskan oleh faktor-faktor dalam model. Nilai yang mendekati 1 menunjukkan model yang sangat baik. Jika nilai $R^2$ **sangat rendah** (atau negatif), model ini **tidak efektif** dalam memprediksi IPK.
    """)

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 Aplikasi Analisis Data Mahasiswa</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; margin-bottom: 20px;'>Analisis Deskriptif, Korelasi Spearman, dan Regresi Linear Berganda</h4>", unsafe_allow_html=True)

# Input File
with st.container(): 
    st.markdown("<p style='text-align: center; font-size: 16px; font-weight: bold;'>📂 Upload File Data Mahasiswa dan Survei yang sudah dicompile (.xlsx)</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload File", 
        type="xlsx",
        label_visibility='hidden' 
    )

if uploaded_file is not None:
    df, df1, data_master = load_data(uploaded_file)
    
    if df is not None:
        # Pilihan Menu
        tab1, tab2, tab3 = st.tabs(["Visualisasi Deskriptif", "Hasil Uji Korelasi Spearman", "Hasil Regresi Linear Berganda"])

        # ====================================================================
        # TAB 1: VISUALISASI DESKRIPTIF
        # ====================================================================
        with tab1:
            st.header("Visualisasi Utama Performansi Akademik")
            
            unique_angkatan = sorted(df['ANGKATAN'].unique())
            
            # --- Sidebar/Filter Interaktif Global untuk Tab ini ---
            st.sidebar.header("Filter Visualisasi")
            selected_angkatan_tren = st.sidebar.multiselect(
                "Pilih Angkatan untuk Tren IPS:",
                unique_angkatan,
                default=unique_angkatan # Default menampilkan semua
            )
            selected_angkatan_kelulusan = st.sidebar.selectbox(
                "Pilih Angkatan untuk Status Kelulusan:",
                unique_angkatan,
                index=len(unique_angkatan) - 1 # Default angkatan terakhir
            )

            # --- Row 1: Distribusi Angkatan dan Rata-rata IPK ---
            col1, col2 = st.columns(2)
            
            # 1. Distribusi Mahasiswa per Angkatan
            with col1:
                st.subheader("Distribusi Jumlah Mahasiswa Berdasarkan Angkatan")
                df_angkatan_count = df.groupby('ANGKATAN')['NIM'].nunique().reset_index()
                df_angkatan_count.rename(columns={'NIM': 'JUMLAH_MAHASISWA'}, inplace=True)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x='ANGKATAN', y='JUMLAH_MAHASISWA', data=df_angkatan_count, palette='Spectral', ax=ax)
                for index, row in df_angkatan_count.iterrows():
                    ax.text(row.name, row['JUMLAH_MAHASISWA'] + 5, str(row['JUMLAH_MAHASISWA']), color='black', ha="center", fontsize=10)
                
                ax.set_title('Distribusi Jumlah Mahasiswa')
                ax.set_xlabel('Angkatan Masuk')
                ax.set_ylabel('Jumlah Mahasiswa')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
            
            # 2. Rata-rata IPK per Angkatan (Interaktif)
            with col2:
                st.subheader("Rata-rata IPK Mahasiswa per Angkatan")
                avg_ipk_by_angkatan = df1.groupby('ANGKATAN')['IPK'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x='ANGKATAN', y='IPK', data=avg_ipk_by_angkatan, palette='viridis', ax=ax)
                for index, row in avg_ipk_by_angkatan.iterrows():
                    ax.text(index, row.IPK + 0.01, round(row.IPK, 2), color='black', ha="center")
                    
                ax.set_title('Rata-rata IPK Mahasiswa per Angkatan')
                ax.set_xlabel('Angkatan')
                ax.set_ylabel('Rata-rata IPK')
                ax.set_ylim(min(2.5, avg_ipk_by_angkatan['IPK'].min()), 4.0)
                ax.grid(axis='y', linestyle='--')
                st.pyplot(fig)

            st.markdown("---")

            # --- Row 2: Tren IPS dan Status Kelulusan ---
            col3, col4 = st.columns(2)
            
            # 3. Tren Rata-rata IPS per Semester (Interaktif berdasarkan Angkatan)
            with col3:
                st.subheader("Tren Rata-Rata IPS Berdasarkan Semester")
                
                if selected_angkatan_tren:
                    # Filter data berdasarkan angkatan yang dipilih
                    df_ips_filtered = df[df['ANGKATAN'].isin(selected_angkatan_tren) & df['IPS'].notna()]
                    
                    if not df_ips_filtered.empty:
                        avg_ips_by_semester = df_ips_filtered.groupby(['SEMESTER_AMBIL', 'ANGKATAN'])['IPS'].mean().reset_index()
                        
                        # Mengurutkan berdasarkan urutan yang telah dideklarasikan
                        avg_ips_by_semester['ORDER'] = avg_ips_by_semester['SEMESTER_AMBIL'].apply(
                            lambda x: SEMESTER_URUT.index(str(x)) if str(x) in SEMESTER_URUT else 999
                        )
                        avg_ips_by_semester = avg_ips_by_semester.sort_values('ORDER')

                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # Gunakan hue='ANGKATAN' agar setiap angkatan memiliki garis warna berbeda
                        sns.lineplot(
                            x='SEMESTER_AMBIL', 
                            y='IPS', 
                            hue='ANGKATAN', 
                            data=avg_ips_by_semester, 
                            marker='o', 
                            ax=ax,
                            palette=sns.color_palette("tab10", n_colors=len(selected_angkatan_tren))
                        )
                        ax.set_title(f'Tren Rata-Rata IPS (Angkatan: {", ".join(map(str, selected_angkatan_tren))})')
                        ax.set_xlabel('Semester Akademik')
                        ax.set_ylabel('Rata-Rata IPS')
                        ax.legend(title='Angkatan')
                        ax.grid(True)
                        st.pyplot(fig)
                    else:
                        st.info("Tidak ada data IPS untuk angkatan yang dipilih.")
                else:
                    st.info("Silakan pilih minimal satu Angkatan dari Filter Visualisasi di Sidebar.")

            # 4. Proporsi Status Kelulusan (Interaktif berdasarkan Angkatan dari Sidebar)
            with col4:
                st.subheader(f"Proporsi Status Kelulusan Angkatan {selected_angkatan_kelulusan}")
                
                # Filter data berdasarkan angkatan yang dipilih
                df_filtered_angkatan = df1[df1['ANGKATAN'] == selected_angkatan_kelulusan]
                
                if not df_filtered_angkatan.empty:
                    status_counts = df_filtered_angkatan['KELULUSAN_STATUS'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie(
                        status_counts, 
                        labels=status_counts.index, 
                        autopct='%1.1f%%',
                        startangle=140, 
                        colors=sns.color_palette("Set2"),
                        wedgeprops={'edgecolor': 'black'}
                    )
                    ax.set_title(f'Proporsi Status Mahasiswa Angkatan {selected_angkatan_kelulusan}')
                    ax.axis('equal')
                    st.pyplot(fig)
                else:
                    st.warning(f"Data status kelulusan untuk Angkatan {selected_angkatan_kelulusan} tidak tersedia.")

           # st.markdown("---")

        # ====================================================================
        # TAB 2: UJI KORELASI SPEARMAN
        # ====================================================================
        with tab2:
            st.header("Hasil Uji Korelasi Spearman vs IPK")
            st.markdown("Uji ini mengukur kekuatan dan arah hubungan monontonik antara faktor survei dengan IPK.")
            
            selected_group = st.selectbox(
                "Pilih Kelompok Faktor untuk Uji Korelasi:",
                list(FAKTOR_GROUPS.keys())
            )
            
            run_spearman_correlation(data_master, FAKTOR_GROUPS[selected_group], selected_group)

        # ====================================================================
        # TAB 3: HASIL REGRESI LINEAR BERGANDA
        # ====================================================================
        with tab3:
            st.header("Hasil Regresi Linear Berganda vs IPK")
            st.markdown("Uji ini mengukur bagaimana satu set faktor secara kolektif dapat memprediksi IPK.")
            
            selected_group_reg = st.selectbox(
                "Pilih Kelompok Faktor untuk Uji Regresi:",
                list(FAKTOR_GROUPS.keys())
            )

            run_linear_regression(data_master, FAKTOR_GROUPS[selected_group_reg], selected_group_reg)

# Jika file belum diunggah
else:
    st.info("Silakan unggah file Excel Anda untuk memulai analisis.")