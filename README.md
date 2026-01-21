<<<<<<< HEAD
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

## 🛠️ Technologies Used

* **Language:** Python 3.12
* **Libraries:**
    * `pandas` (Data manipulation and analysis)
    * `scikit-learn` (Machine Learning models and metrics)
    * `numpy` (Numerical computing)
    * `matplotlib` (Data visualization - optional)

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
    pip install pandas scikit-learn numpy
    ```

3.  **Run the analysis script:**
    ```bash
    python main.py
    ```

## 📊 Methodology (Virtual Solution)

1.  **Data Preprocessing:** Cleaning the dataset and mapping Wifi signals to location labels.
2.  **Model Training:** Training a k-NN classifier on the training set (Wifi Fingerprints).
3.  **Testing:** Simulating real-time localization by predicting locations on the validation set.
4.  **Results:** The system outputs the classification report and overall accuracy score.
=======
# Wi-fi-Indoor-Localization
Indoor Positioning System (IPS) based on Wi-Fi Fingerprinting and Machine Learning (k-NN), implemented in Python.
>>>>>>> 6ad8c9ef82c2a553275a5c47920c7ca6d84f0315
