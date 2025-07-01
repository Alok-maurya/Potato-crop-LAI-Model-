# -*- coding: utf-8 -*-
"""

@author: ALOK KUMAR MAURYA

This script consist of trained model on the full crop growth period dataset:
"""

import time
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
# from sklearn.externals import joblib 
import joblib # For loading the trained regression model

# Load the trained regression model
model_path = r"E:\Phd work\DATA_CALCULATION\Module01-CanopyCover_Paper\Paper2_LAI\Model\XGBOOST\MLR_FULL_POOL_best_model.pkl"  # Update with the correct path
regression_model = joblib.load(model_path)


def adjust_pixel_count(canopy_cover):
    """Adjust the pixel count based on canopy cover percentage."""
    if canopy_cover < 5:
        return 1000
    elif canopy_cover < 10:
        return 2000
    elif canopy_cover < 30:
        return 3000
    else:
        return 500

def compute_average_canopy_cover(ExG_O, CGI_O, MExGI_O, ExG_T, CGI_T, MExGI_T):
    """Compute average canopy cover values."""
    average_canopy_cover = (ExG_O + CGI_O + MExGI_O) / 3
    average_canopy_cover_T = (ExG_T + CGI_T + MExGI_T) / 3
    return average_canopy_cover, average_canopy_cover_T


def process_and_predict(image_path):
    """Process the image and predict LAI using the regression model."""
    im = Image.open(image_path)
    im_array = np.array(im)
    height, width, _ = im_array.shape
    numpixels = height * width

    # Normalize RGB channels
    r = im_array[:, :, 0] / 255.0
    g = im_array[:, :, 1] / 255.0
    b = im_array[:, :, 2] / 255.0

    # Apply median filter
    mr = median_filter(r, size=3)
    mg = median_filter(g, size=3)
    mb = median_filter(b, size=3)

    # MExGI Calculation
    MExGI = 1.262 * mg - 0.884 * mr - 0.311 * mb
    T1 = threshold_otsu(MExGI)
    BW1 = MExGI > T1
    canopypx1 = np.sum(BW1)
    canopy_cover_MExGI_O = 100 * (canopypx1 / numpixels)

    # Adjust minPixelCount based on canopy_cover
    minPixelCount1 = adjust_pixel_count(canopy_cover_MExGI_O)

    # Remove small objects
    cleaned_binary_image1 = remove_small_objects(BW1, min_size=minPixelCount1)
    canopyp1 = np.sum(cleaned_binary_image1)
    canopy_cover_MExGI_T = 100 * (canopyp1 / numpixels)

    SDscaleT = (canopy_cover_MExGI_T - 45.901) / 14.410

    # CGI Calculation
    ExG = 2 * mg - (mr + mb)
    CGI = ExG + MExGI
    T2 = threshold_otsu(CGI)
    BW2 = CGI > T2
    canopypx2 = np.sum(BW2)
    canopy_cover_CGI_O = 100 * (canopypx2 / numpixels)
    minPixelCount2 = adjust_pixel_count(canopy_cover_CGI_O)

    cleaned_binary_image2 = remove_small_objects(BW2, min_size=minPixelCount2)
    canopyp2 = np.sum(cleaned_binary_image2)
    canopy_cover_CGI_T = 100 * (canopyp2 / numpixels)
    SDscaleT1 = (canopy_cover_CGI_T - 45.261) / 14.841

    # ExG Calculation
    T3 = threshold_otsu(ExG)
    BW3 = ExG > T3
    canopypx3 = np.sum(BW3)
    canopy_cover_ExG_O = 100 * (canopypx3 / numpixels)
    minPixelCount3 = adjust_pixel_count(canopy_cover_ExG_O)

    cleaned_binary_image3 = remove_small_objects(BW3, min_size=minPixelCount3)
    canopyp3 = np.sum(cleaned_binary_image3)
    canopy_cover_ExG_T = 100 * (canopyp3 / numpixels)
    SDscaleT2 = (canopy_cover_ExG_T - 45.07) / 13.308

    # Compute average canopy cover
    average_canopy_cover, average_canopy_cover_T = compute_average_canopy_cover(
        canopy_cover_ExG_O, canopy_cover_CGI_O, canopy_cover_MExGI_O,
        canopy_cover_ExG_T, canopy_cover_CGI_T, canopy_cover_MExGI_T
    )

    # Prepare input for the model
    input_features = np.array([[SDscaleT, SDscaleT1, SDscaleT2]])
    predicted_LAI = regression_model.predict(input_features)[0]

    # Save data to Excel
    update_excel_data(image_path, canopy_cover_MExGI_T, canopy_cover_ExG_T, canopy_cover_CGI_T, predicted_LAI)

    # Plotting
    plot_temporal_data()

    return average_canopy_cover, predicted_LAI


def update_excel_data(image_name, canopy_cover,canopy_cover1,canopy_cover2, predicted_LAI):
    """Update the Excel file with new data."""
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data = {
        "Image Name": os.path.basename(image_name),
        "Date/Time": current_date,
        "canopy_cover_MExGI_T (%)": canopy_cover,
        "canopy_cover_ExG_T (%)": canopy_cover2,
        "canopy_cover_CGI_T (%)": canopy_cover1,
        "Predicted LAI": predicted_LAI,
    }
    output_file_path = r"E:\Phd work\DATA_CALCULATION\Module01-CanopyCover_Paper\Paper2_LAI\Paper_revision\Result\canopy_data_calibartion.xlsx"

    try:
        existing_data = pd.read_excel(output_file_path)
        existing_data = pd.concat([existing_data, pd.DataFrame([new_data])], ignore_index=True)
        existing_data.to_excel(output_file_path, index=False)
    except FileNotFoundError:
        pd.DataFrame([new_data]).to_excel(output_file_path, index=False)


import pandas as pd
import matplotlib.pyplot as plt

def plot_temporal_data():
    """Generate a temporal plot with subplots for Canopy Cover and Predicted LAI."""
    output_file_path = r"E:\Phd work\DATA_CALCULATION\Module01-CanopyCover_Paper\Paper2_LAI\Paper_revision\Result\canopy_data_calibartion.xlsx"
    data = pd.read_excel(output_file_path)

    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot for Predicted LAI
    axs[0].plot(data["Date/Time"], data["Predicted LAI"], marker="o", label="Predicted LAI", color="blue")
    axs[0].set_xlabel("Date/Time",fontweight='bold')
    axs[0].set_ylabel("Predicted LAI",fontweight='bold')
    axs[0].set_title("Temporal variation Predicted LAI",fontweight='bold', fontsize='14')
    axs[0].tick_params(axis="x", rotation=30)
    axs[0].legend()

    # Second subplot for Canopy Cover
    canopy_cover_avgf = (data["canopy_cover_MExGI_T (%)"] + data["canopy_cover_ExG_T (%)"] + data["canopy_cover_CGI_T (%)"]) / 3
    axs[1].plot(data["Date/Time"], canopy_cover_avgf, marker="o", label="Canopy Cover", color="green")
    axs[1].set_xlabel("Date/Time",fontweight='bold')
    axs[1].set_ylabel("Canopy Cover (%)",fontweight='bold')
    axs[1].set_title("Temporal variation of Canopy Cover (%)",fontweight='bold', fontsize='14')
    axs[1].tick_params(axis="x", rotation=30)
    axs[1].legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot to a file
    plot_path = r"E:\Phd work\DATA_CALCULATION\Module01-CanopyCover_Paper\Paper2_LAI\Paper_revision\Result\temporal_plotFull.jpg"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {plot_path}")

# Example usage
plot_temporal_data()



# RUN
image_path = r"E:\Phd work\DATA_CALCULATION\Module01-CanopyCover_Paper\Paper2_LAI\Data\Calibration dataF25\F25_3\IMG20240128122251.jpg"  # Replace with actual image path
canopy_cover, predicted_LAI = process_and_predict(image_path)
print(f"Canopy Cover: {canopy_cover}, Predicted LAI: {predicted_LAI}")
