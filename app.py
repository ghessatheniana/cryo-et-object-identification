import streamlit as st
import os
import pandas as pd
import torch # Diperlukan untuk sample_transformed.image.unsqueeze(0).to(device)
from czii_helper import *
from dataset import *

# Impor dari file modular
from config import (
    DEFAULT_MODEL_FILENAME, MODEL_CHANNELS_OPTIONS, DEFAULT_MODEL_CHANNELS_KEY,
    DEFAULT_DATASET_LOCATION, DEFAULT_COPICK_CONFIG_CONTENT,
    CLASSES, ID_TO_NAME
)
from model_utils import load_model
from copick_utils import get_copick_root, load_tomogram_data
from inference import get_inference_transforms, run_inference, extract_particles
from visualization import dict_to_df, plot_3d_scatter
from evaluation_utils import plot_evaluation_results, do_one_eval, compute_lb

# --- Konfigurasi Aplikasi ---
st.set_page_config(page_title="CryoET Protein Object Identification", layout="wide")
st.title("🔬 CryoET Protein Object Identification Inference")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# --- Inisialisasi Session State ---
# Kunci untuk mempertahankan data di antara eksekusi ulang skrip
if 'prediction_complete' not in st.session_state:
    st.session_state.prediction_complete = False
if 'all_results_df' not in st.session_state:
    st.session_state.all_results_df = pd.DataFrame()
if 'lb_gb_df' not in st.session_state:
    st.session_state.lb_gb_df = pd.DataFrame()
if 'lb_score' not in st.session_state:
    st.session_state.lb_score = None
if 'global_evaluation_complete' not in st.session_state:
    st.session_state.global_evaluation_complete = False

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("📁 Lokasi File & Data")
dataset_location = st.sidebar.text_input("Lokasi Direktori Dataset (static_root)", value=(DEFAULT_DATASET_LOCATION + "/static"))
model_location = st.sidebar.text_input("Lokasi File Model (.pth)", value=DEFAULT_MODEL_FILENAME)
copick_config_str = DEFAULT_COPICK_CONFIG_CONTENT

st.sidebar.header("⚙️ Pengaturan Inferensi")
selected_model = st.sidebar.selectbox("Arsitektur", ["3DUnet", "3DUnetAG", "3DUnetSE", "3DUnetCBAM"], index=0)
selected_model_channels_key = st.sidebar.selectbox(
    "Arsitektur Channel Model",
    options=list(MODEL_CHANNELS_OPTIONS.keys()),
    index=list(MODEL_CHANNELS_OPTIONS.keys()).index(DEFAULT_MODEL_CHANNELS_KEY)
)
model_channels_tuple_input = MODEL_CHANNELS_OPTIONS[selected_model_channels_key]

voxel_size_input = 10.0
tomo_type_input = "denoised"

# st.sidebar.header("🔬 Threshold Partikel")
class_thresholds_input = {
    1: {"blob": 90,  "certainty": 0.5},
    2: {"blob": 100, "certainty": 0.7},
    3: {"blob": 200, "certainty": 0.95},
    4: {"blob": 100, "certainty": 0.4},
    5: {"blob": 500, "certainty": 0.95},
    6: {"blob": 500, "certainty": 0.5},
}
# Opsi untuk input manual (dinonaktifkan sementara)
# for c_id, name in ID_TO_NAME.items():
#     cols = st.sidebar.columns(2)
#     with cols[0]:
#         blob_thresh = st.number_input(f"Min Blob ({name})", value=100, min_value=1, step=10, key=f"blob_{c_id}")
#     with cols[1]:
#         certainty_thresh = st.slider(f"Certainty ({name})", 0.0, 1.0, 0.5, 0.01, key=f"cert_{c_id}")
#     class_thresholds_input[c_id] = {"blob": blob_thresh, "certainty": certainty_thresh}

# --- Load Model dan Copick Root ---
model, device = None, None
copick_root = None

if dataset_location and model_location and copick_config_str:
    if os.path.exists(model_location) and os.path.isdir(dataset_location):
        try:
            model, device = load_model(model_location, model_channels_tuple_input, selected_model)
            copick_root = get_copick_root(copick_config_str, dataset_location)
        except Exception as e:
            st.sidebar.error(f"Gagal memuat model atau copick: {e}")
    else:
        if not os.path.exists(model_location):
            st.sidebar.warning("Path model tidak valid.")
        if not os.path.isdir(dataset_location):
            st.sidebar.warning("Path dataset (static_root) tidak valid.")
else:
    st.sidebar.warning("Harap isi semua path yang diperlukan.")

# --- Logika Utama Aplikasi ---
if model and copick_root:
    try:
        run_objects = copick_root.runs
        run_names = [run.name for run in run_objects if run]
        
        if not run_names:
            st.warning(f"Tidak ada 'runs' yang ditemukan di {dataset_location}.")
        else:
            selected_run_names = st.multiselect("Pilih Run untuk Inferensi:", run_names, default=run_names[0] if run_names else None)

            if st.button("🚀 Lakukan Prediksi", use_container_width=True) and selected_run_names:
                # Blok ini hanya berjalan saat tombol ditekan
                all_results_df_local = pd.DataFrame()
                progress_bar = st.progress(0, "Memulai proses inferensi...")
                total_runs = len(selected_run_names)
                inference_transforms = get_inference_transforms()

                with st.spinner("Melakukan inferensi pada semua run yang dipilih..."):
                    for i, run_name in enumerate(selected_run_names):
                        progress_bar.progress((i) / total_runs, f"Memproses: {run_name}")
                        run = copick_root.get_run(run_name)
                        if not run:
                            st.error(f"Run {run_name} tidak ditemukan.")
                            continue

                        tomo_data, voxel_size_actual = load_tomogram_data(run, voxel_size_input, tomo_type_input)
                        if tomo_data is None:
                            continue

                        sample = {"image": tomo_data}
                        try:
                            sample_transformed = inference_transforms(sample)
                            input_tensor = sample_transformed["image"].unsqueeze(0).to(device)
                        except Exception as e:
                            st.error(f"Error saat transformasi data untuk {run_name}: {e}")
                            continue
                        
                        probs = run_inference(model, device, input_tensor)
                        location_run = extract_particles(probs, class_thresholds_input, voxel_size_actual)
                        
                        df_output_run = dict_to_df(location_run, run.name)
                        all_results_df_local = pd.concat([all_results_df_local, df_output_run], ignore_index=True)
                    
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache() # Membersihkan cache memori GPU

                    progress_bar.progress(1.0, "Inferensi Selesai!")

                # Simpan hasil ke session state
                st.session_state.all_results_df = all_results_df_local
                st.session_state.prediction_complete = True
                # Reset state evaluasi sebelumnya
                st.session_state.lb_score = None
                st.session_state.lb_gb_df = pd.DataFrame()
                
                st.session_state.global_evaluation_complete = False 


                # Paksa Streamlit untuk menjalankan ulang skrip dengan state yang baru
                st.rerun()
                
    except Exception as e:
        st.error(f"Terjadi error pada level aplikasi: {e}")
        st.error("Pastikan semua konfigurasi dan path sudah benar.")

# --- Tampilkan Hasil, Visualisasi, dan Evaluasi dari Session State ---
# Blok ini akan berjalan jika prediksi telah selesai, bahkan setelah interaksi widget lain
if st.session_state.prediction_complete and not st.session_state.all_results_df.empty:
    st.markdown("--- \n ### 📊 Hasil Inferensi Keseluruhan")
    st.dataframe(st.session_state.all_results_df)
    
    csv_all = st.session_state.all_results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Unduh semua hasil sebagai CSV",
        data=csv_all,
        file_name='all_predictions.csv',
        mime='text/csv',
    )

    # st.markdown("--- \n ###  Visualisasi 3D Scatter Plot")
    # particle_types_in_results = st.session_state.all_results_df['particle_type'].unique()
    # if len(particle_types_in_results) > 0:
    #     selected_particle_type_vis = st.selectbox(
    #         "Pilih Tipe Partikel untuk Visualisasi:", 
    #         particle_types_in_results,
    #         key="vis_3d_selector" # Key unik untuk widget
    #     )
    #     if selected_particle_type_vis:
    #         df_vis = st.session_state.all_results_df[st.session_state.all_results_df['particle_type'] == selected_particle_type_vis]
    #         plot_3d_scatter(df_vis, selected_particle_type_vis)
    # else:
    #     st.warning("Tidak ada tipe partikel yang terdeteksi untuk divisualisasikan.")
    
    # --- GANTI BAGIAN EVALUASI LAMA ANDA DENGAN BLOK KOMPREHENSIF INI ---

    # --- Bagian Evaluasi ---
    st.header("📈 Evaluasi Kinerja Model")
    eval_data_dir = DEFAULT_DATASET_LOCATION

    if eval_data_dir:
        try:
            submission_df = st.session_state.all_results_df
            eval_run_ids = list(submission_df['experiment'].unique())

            if st.session_state.prediction_complete and not eval_run_ids:
                st.warning("Inferensi selesai tetapi tidak ada partikel yang terdeteksi untuk dievaluasi.")
            
            elif eval_run_ids:
                # ===================================================================
                # BAGIAN 1: PERHITUNGAN SKOR GLOBAL OTOMATIS (DIJALANKAN SEKALI)
                # ===================================================================
                # Cek jika prediksi sudah selesai TAPI evaluasi global untuk set data ini BELUM.
                # Ini adalah "gerbang" yang mencegah perulangan tak terbatas.
                if st.session_state.prediction_complete and not st.session_state.get('global_evaluation_complete', False):
                    with st.spinner("Menghitung Skor Evaluasi Global secara otomatis..."):
                        lb_gb_df, lb_score = compute_lb(
                            submission_df,
                            f'{eval_data_dir}/overlay/ExperimentRuns',
                            valid_ids_for_eval=eval_run_ids
                        )
                        # Simpan hasil global ke session state
                        st.session_state.lb_gb_df = lb_gb_df
                        st.session_state.lb_score = lb_score
                        
                        # [PENTING] Tutup gerbangnya agar perhitungan ini tidak dijalankan lagi
                        st.session_state.global_evaluation_complete = True
                        # Lakukan satu kali rerun untuk segera menampilkan hasil yang baru dihitung
                        st.rerun()

                # ===================================================================
                # BAGIAN 2: TAMPILKAN SEMUA HASIL (GLOBAL DAN PER-RUN)
                # ===================================================================

                # Tampilkan hasil global jika sudah ada di session state
                if st.session_state.get('lb_score') is not None:
                    st.subheader("Evaluasi Keseluruhan")
                    st.metric(label="🏆 F-beta Score Keseluruhan", value=f"{st.session_state.lb_score:.4f}")
                    st.write("Rincian Skor per Tipe Partikel (di semua run):")
                    st.dataframe(st.session_state.lb_gb_df)
                    st.markdown("---")

                # Bagian untuk analisis detail per run (tetap sama)
                st.subheader("Analisis Detail per Run")
                selected_eval_id_for_vis = st.selectbox(
                    "Pilih Run untuk dievaluasi dan divisualisasikan secara detail:",
                    eval_run_ids,
                    key="vis_eval_selector"
                )

                if selected_eval_id_for_vis:
                    with st.spinner(f"Membuat analisis detail untuk {selected_eval_id_for_vis}..."):
                        eval_fig, metric_df_for_run = plot_evaluation_results(
                            submission_df,
                            f'{eval_data_dir}/overlay/ExperimentRuns',
                            selected_eval_id_for_vis
                        )
                        st.write(f"**Ringkasan Evaluasi untuk Run: `{selected_eval_id_for_vis}`**")
                        st.dataframe(metric_df_for_run)
                        st.write(f"**Visualisasi 3D untuk Run: `{selected_eval_id_for_vis}`**")
                        st.pyplot(eval_fig)

        except Exception as e:
            st.error(f"Terjadi error saat melakukan evaluasi: {e}")
    else:
        st.info("Atur `Lokasi Direktori Data Evaluasi` untuk melihat hasil evaluasi.")

# Kondisi jika belum ada model yang dimuat atau belum ada prediksi
elif not model or not copick_root:
    st.info("Harap periksa konfigurasi di sidebar untuk memuat model dan data.")
else:
    st.info("Pilih run dari daftar dan klik 'Lakukan Prediksi' untuk memulai.")