# config.py
import os

# Model
DEFAULT_MODEL_FILENAME = "D:/KULIAH POLBAN/TA/streamlit/se_128_128.pth"
MODEL_CHANNELS_OPTIONS = {
    "32-64-128-256": (32, 64, 128, 256),
    "16-32-64-128": (16, 32, 64, 128),
}
DEFAULT_MODEL_CHANNELS_KEY = "16-32-64-128"

# Copick
DEFAULT_DATASET_LOCATION = os.path.join("D:/KULIAH POLBAN/TA/dataset/archive/test")
DEFAULT_COPICK_CONFIG_CONTENT = """
{
    "name": "czii_cryoet_mlchallenge_2024_streamlit",
    "description": "Streamlit app for CryoET ML Challenge.",
    "version": "1.0.0",
    "pickable_objects": [
        {"name": "apo-ferritin", "is_particle": true, "pdb_id": "4V1W", "label": 1, "color": [0, 117, 220, 128], "radius": 60, "map_threshold": 0.0418},
        {"name": "beta-amylase", "is_particle": true, "pdb_id": "1FA2", "label": 2, "color": [153, 63, 0, 128], "radius": 65, "map_threshold": 0.035},
        {"name": "beta-galactosidase", "is_particle": true, "pdb_id": "6X1Q", "label": 3, "color": [76, 0, 92, 128], "radius": 90, "map_threshold": 0.0578},
        {"name": "ribosome", "is_particle": true, "pdb_id": "6EK0", "label": 4, "color": [0, 92, 49, 128], "radius": 150, "map_threshold": 0.0374},
        {"name": "thyroglobulin", "is_particle": true, "pdb_id": "6SCJ", "label": 5, "color": [43, 206, 72, 128], "radius": 130, "map_threshold": 0.0278},
        {"name": "virus-like-particle", "is_particle": true, "label": 6, "color": [255, 204, 153, 128], "radius": 135, "map_threshold": 0.201}
    ],
    "overlay_root": "overlay_output_streamlit",
    "overlay_fs_args": {"auto_mkdir": true},
    "static_root": ""
}
"""

# Inferensi
CLASSES = [1, 2, 3, 4, 5, 6]
ID_TO_NAME = {
    1: "apo-ferritin",
    2: "beta-amylase",
    3: "beta-galactosidase",
    4: "ribosome",
    5: "thyroglobulin",
    6: "virus-like-particle"
}

# Parameter Sliding Window
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 8
OVERLAP = 0.25