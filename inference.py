# inference.py
import torch
import numpy as np
from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, NormalizeIntensityd
from monai.inferers import sliding_window_inference
from skimage.measure import label, regionprops
from config import CLASSES, ID_TO_NAME, ROI_SIZE, SW_BATCH_SIZE, OVERLAP

def get_inference_transforms():
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS")
    ])

def run_inference(model, device, input_tensor):
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP
        )[0]
    probs = torch.softmax(output, dim=0)
    return probs

def extract_particles(probs, class_thresholds_config, voxel_size_actual):
    location_run = {}
    for c_class_id in CLASSES:
        thresholds_current_class = class_thresholds_config.get(c_class_id, {"blob": 100, "certainty": 0.5})
        blob_thresh = thresholds_current_class["blob"]
        certainty_thresh = thresholds_current_class["certainty"]

        binary_mask = (probs[c_class_id] > certainty_thresh).cpu().numpy().astype(np.uint8)
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask, intensity_image=probs[c_class_id].cpu().numpy())

        centroids = []
        for region in regions:
            if region.area > blob_thresh:
                z, y, x = region.centroid
                particle_certainty = region.mean_intensity
                centroids.append([
                    x * voxel_size_actual,
                    y * voxel_size_actual,
                    z * voxel_size_actual,
                    particle_certainty
                ])
        location_run[ID_TO_NAME[c_class_id]] = np.array(centroids)
    return location_run