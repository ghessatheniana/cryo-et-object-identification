# visualization.py
import streamlit as st
import pandas as pd
import plotly.express as px

def dict_to_df(location_dict, run_name):
    rows = []
    for particle_type, coords_list in location_dict.items():
        if coords_list.size > 0: # Pastikan tidak kosong
            for coords_certainty in coords_list:
                x, y, z, certainty = coords_certainty
                rows.append({
                    "experiment": run_name,
                    "particle_type": particle_type,
                    "x": x,
                    "y": y,
                    "z": z,
                    "certainty": certainty
                })
    return pd.DataFrame(rows)

def plot_3d_scatter(df_vis, particle_type_name):
    if not df_vis.empty:
        fig = px.scatter_3d(df_vis, x='x', y='y', z='z',
                            color='experiment',
                            size='certainty',
                            opacity=0.7,
                            title=f"Visualisasi Partikel: {particle_type_name}")
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Tidak ada data untuk partikel {particle_type_name} untuk divisualisasikan.")