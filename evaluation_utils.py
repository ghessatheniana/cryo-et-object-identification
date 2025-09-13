import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment # Import ini
from czii_helper import *
from dataset import *

# --- Fungsi do_one_eval ---
def do_one_eval(truth, predict, threshold):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)]
    return hit, fp, miss, metric

# --- Fungsi compute_lb ---
def compute_lb(submit_df, overlay_dir, valid_ids_for_eval=None):
    if valid_ids_for_eval is None:
        valid_ids_for_eval = list(submit_df['experiment'].unique())
    
    print(f"Mengevaluasi ID: {valid_ids_for_eval}")

    eval_df = []
    for id_name in valid_ids_for_eval: # Menggunakan id_name untuk menghindari konflik dengan 'id' pandas
        truth = read_one_truth(id_name, overlay_dir)
        id_df = submit_df[submit_df['experiment'] == id_name]
        for p in PARTICLE:
            p = dotdict(p)
            # print('\r', id_name, p.name, end='', flush=True) # Nonaktifkan untuk Streamlit
            xyz_truth = truth.get(p.name, np.array([])) # Gunakan .get() untuk menghindari KeyError
            xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius * 0.5)
            eval_df.append(dotdict(
                id=id_name, particle_type=p.name,
                P=metric[0], T=metric[1], hit=metric[2], miss=metric[3], fp=metric[4],
            ))
    print('')
    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby('particle_type').agg('sum').drop(columns=['id'])
    gb.loc[:, 'precision'] = gb['hit'] / gb['P']
    gb.loc[:, 'precision'] = gb['precision'].fillna(0)
    gb.loc[:, 'recall'] = gb['hit'] / gb['T']
    gb.loc[:, 'recall'] = gb['recall'].fillna(0)
    gb.loc[:, 'f-beta4'] = 17 * gb['precision'] * gb['recall'] / (16 * gb['precision'] + gb['recall'])
    gb.loc[:, 'f-beta4'] = gb['f-beta4'].fillna(0)

    gb = gb.sort_values('particle_type').reset_index(drop=False)
    gb.loc[:, 'weight'] = [1, 0, 2, 1, 2, 1] # Pastikan urutan partikel di sini sama dengan sort_values
    lb_score = (gb['f-beta4'] * gb['weight']).sum() / gb['weight'].sum()
    return gb, lb_score

# --- Fungsi untuk Visualisasi Hasil Evaluasi (Baru) ---
def plot_evaluation_results(submission_df, overlay_dir, run_id):
    """
    Fungsi ini melakukan tiga hal untuk satu 'run_id':
    1. Menghitung metrik evaluasi (Hit, Miss, FP) untuk setiap jenis partikel.
    2. Membuat plot 3D yang memvisualisasikan hasil evaluasi.
    3. Mengembalikan DataFrame metrik dan figure plot.
    """
    
    # Baca data ground truth untuk run_id yang dipilih
    truth = read_one_truth(run_id, overlay_dir=overlay_dir)
    
    # Ambil data prediksi hanya untuk run_id yang dipilih
    submit_df_id = submission_df[submission_df['experiment'] == run_id]
    
    fig = plt.figure(figsize=(20, 10))
    particle_metrics = [] # List untuk menyimpan kamus metrik

    # Loop melalui setiap jenis partikel yang telah didefinisikan
    for p in PARTICLE:
        p = dotdict(p)
        xyz_truth = truth.get(p.name, []) # Gunakan .get untuk menghindari error jika partikel tidak ada di ground truth
        xyz_predict = submit_df_id[submit_df_id['particle_type'] == p.name][['x', 'y', 'z']].values

        # Lakukan evaluasi untuk partikel ini
        # Pastikan do_one_eval dapat menangani xyz_truth yang kosong
        if len(xyz_truth) == 0 and len(xyz_predict) > 0:
            hit, fp, miss, _ = ([], list(range(len(xyz_predict))), [], [])
        elif len(xyz_truth) > 0:
            hit, fp, miss, _ = do_one_eval(xyz_truth, xyz_predict, p.radius)
        else: # Keduanya kosong
            hit, fp, miss, _ = ([], [], [], [])
            
        # Kumpulkan data metrik ke dalam list
        particle_metrics.append({
            'Particle': p.name,
            'Truth Count': len(xyz_truth),
            'Predicted Count': len(xyz_predict),
            'Hit': len(hit[0]) if hit else 0,
            'False Positive': len(fp),
            'Missed': len(miss),
        })

        # --- Bagian Plotting ---
        ax = fig.add_subplot(2, 3, p.label, projection='3d')
        ax.set_title(f'{p.name} ({p.difficulty})', fontsize=12)

        # Hits: Prediksi yang benar
        if hit and len(hit[0]) > 0:
            pt_pred = xyz_predict[hit[0]]
            pt_gt = xyz_truth[hit[1]]
            ax.scatter(pt_pred[:, 0], pt_pred[:, 1], pt_pred[:, 2], alpha=0.6, color='g', label='Hit (Pred)')
            ax.scatter(pt_gt[:, 0], pt_gt[:, 1], pt_gt[:, 2], s=80, facecolors='none', edgecolors='g', label='Hit (GT)')

        # False Positives: Prediksi yang salah (tidak ada ground truth yang cocok)
        if len(fp) > 0:
            pt_fp = xyz_predict[fp]
            ax.scatter(pt_fp[:, 0], pt_fp[:, 1], pt_fp[:, 2], alpha=0.8, color='k', label='False Positive')

        # Misses: Ground truth yang tidak terdeteksi
        if len(miss) > 0:
            pt_miss = xyz_truth[miss]
            ax.scatter(pt_miss[:, 0], pt_miss[:, 1], pt_miss[:, 2], s=160, facecolors='none', edgecolors='r', label='Missed')

        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"Evaluation for Volume: {run_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Konversi list metrik menjadi DataFrame
    metric_df = pd.DataFrame(particle_metrics)
    
    # Kembalikan figure plot dan dataframe metrik
    return fig, metric_df