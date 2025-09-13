# copick_utils.py
import streamlit as st
import copick
import json
import os

@st.cache_resource
def get_copick_root(copick_config_content_str, static_root_path_str):
    try:
        temp_config_path = "temp_copick_config.json"
        config_data = json.loads(copick_config_content_str)
        config_data["static_root"] = static_root_path_str

        overlay_root = config_data.get("overlay_root", "overlay_output_streamlit")
        if not os.path.isabs(overlay_root):
             overlay_root = os.path.join(os.getcwd(), overlay_root)
        config_data["overlay_root"] = overlay_root
        os.makedirs(overlay_root, exist_ok=True)

        with open(temp_config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        root = copick.from_file(temp_config_path)
        os.remove(temp_config_path)
        return root
    except Exception as e:
        st.sidebar.error(f"Error saat memproses copick config: {e}")
        return None

def load_tomogram_data(run, voxel_spacing_value, tomo_type_value):
    voxel_spacing_spec = run.get_voxel_spacing(voxel_spacing_value)
    if voxel_spacing_spec is None:
        st.warning(f"Skipping {run.name}: Tidak ada voxel spacing untuk voxel_size={voxel_spacing_value}")
        return None, None
    
    try:
        tomo_obj = voxel_spacing_spec.get_tomogram(tomo_type_value)
        if tomo_obj is None:
                st.error(f"Tomogram '{tomo_type_value}' tidak ditemukan untuk VoxelSpacing {voxel_spacing_value}Å di run '{run.name}'.")
                return None, None
        tomo_data = tomo_obj.numpy()
        return tomo_data, voxel_spacing_spec.voxel_size
    except Exception as e:
        st.error(f"Error saat me-load tomogram untuk {run.name}: {e}")
        return None, None