# Indoor Localization System (Wi-Fi Fingerprinting)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Status](https://img.shields.io/badge/University_Project-orange?style=flat-square)

This repository contains the source code and documentation for an **Indoor Positioning System (IPS)** I developed as a university project. The solution utilizes **Wi-Fi Fingerprinting** and Machine Learning algorithms to locate a user inside a building (Building ID and Floor) where GPS signals are unreliable.

## 📄 Project Overview

The project focuses on a **virtual solution** approach, analyzing pre-collected sensor data to train a predictive model. By using Received Signal Strength Indication (RSSI) from various Wireless Access Points (WAPs), the system generates a "fingerprint" for specific locations.

### Key Features
* **Data Analysis:** Utilization of the UJIIndoorLoc dataset (multi-building, multi-floor environment).
* **Machine Learning:** Implementation of the **k-Nearest Neighbors (k-NN)** algorithm for classification.
* **Positioning:** Simultaneous prediction of Building ID and Floor number.
* **Performance:** Evaluation of model accuracy using standard metrics.
* **Visualization:** Automatic generation of confusion matrix and data distribution plots for analysis.

## 🛠️ Technologies Used

* **Language:** Python 3.12
* **Libraries:**
    * `pandas` (Data manipulation and analysis)
    * `scikit-learn` (Machine Learning models and metrics)
    * `numpy` (Numerical computing)
    * `matplotlib` (Data visualization and plot generation)
    * `seaborn` (Statistical data visualization)

## 📂 Dataset

This project uses the **UJIIndoorLoc** dataset, available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).

* **Inputs:** Intensity of signals (RSSI) from 520 WAPs.
* **Outputs:** Longitude, Latitude, Floor, Building ID.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/indoor-localization-project.git](https://github.com/yourusername/indoor-localization-project.git)
    cd indoor-localization-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis script:**
    ```bash
    python main.py
    ```

## 📊 Methodology (Virtual Solution)

1.  **Data Preprocessing:** Cleaning the dataset and mapping Wifi signals to location labels (BUILDINGID-FLOOR format).
2.  **Model Training:** Training a k-NN classifier (k=5) on the training set (19,937 fingerprints from 520 WAPs).
3.  **Evaluation:** Predicting locations on the validation set (1,111 samples) and calculating accuracy metrics.
4.  **Visualization:** Generating data distribution plots and confusion matrix heatmaps.
5.  **Demo:** Simulating real-time tracking with a random validation sample.

## 📈 Outputs

When you run the script, it generates:

* **Console Output:**
    * Model training progress
    * Validation accuracy (~90%)
    * Detailed classification report (precision, recall, F1-score per location)
    * Real-time demo prediction result

* **Generated Files:**
    * `distribution_training_set.png` - Bar chart showing fingerprint distribution across locations
    * `confusion_matrix.png` - Heatmap showing prediction accuracy per Building-Floor combination
