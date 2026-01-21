import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def load_data(filepath):
    """
    Loads a dataset from a CSV file.
    """
    try:
        print(f"[INFO] Loading dataset from {filepath}...")
        data = pd.read_csv(filepath)
        print(f"       -> Loaded {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        print(f"[ERROR] File {filepath} not found. Make sure it is in the same folder.")
        return None


def preprocess_data(df):
    """
    Prepares features (X) and targets (y) for the model.
    """
    # 1. Features: Select the first 520 columns (WAPs)
    # The UJIIndoorLoc dataset stores WAP intensity in columns 0 to 519.
    X = df.iloc[:, 0:520]

    # 2. Handling 'No Signal': 
    # The dataset uses +100 for no signal. We replace it with -105 (a very weak signal in dBm)
    # so the distance calculation in k-NN works correctly.
    X = X.replace(100, -105)

    # 3. Target: Combine BuildingID and Floor (e.g., "2-3")
    y = df['BUILDINGID'].astype(str) + "-" + df['FLOOR'].astype(str)

    return X, y


def plot_confusion_matrix(y_true, y_pred):
    """
    Generates and saves a Confusion Matrix heatmap.
    """
    print("[INFO] Generating Confusion Matrix plot...")

    # Get unique labels from the data to ensure correct order
    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Predicted vs Actual Location')
    plt.xlabel('Predicted Location')
    plt.ylabel('Actual Location')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save instead of show, so you can use it in your PDF
    plt.savefig('confusion_matrix.png')
    print("[SUCCESS] Saved 'confusion_matrix.png'")
    # plt.show() # Uncomment if you want to see it immediately


def plot_data_distribution(y_data, title="Data Distribution"):
    """
    Plots a bar chart showing samples per location.
    """
    print(f"[INFO] Generating distribution plot for {title}...")

    plt.figure(figsize=(12, 6))
    y_data.value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title(f'Number of Fingerprints per Location ({title})')
    plt.xlabel('Location (Building-Floor)')
    plt.ylabel('Count of Samples')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f'distribution_{title.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    print(f"[SUCCESS] Saved '{filename}'")


def main():
    print("--- INDOOR LOCALIZATION SYSTEM (Virtual Solution) ---\n")

    # 1. Load Training Data
    train_df = load_data('UJIndoorLoc/trainingData.csv')

    # 2. Load Validation Data
    val_df = load_data('UJIndoorLoc/validationData.csv')

    if train_df is not None and val_df is not None:
        print("\n[INFO] Preprocessing data...")
        X_train, y_train = preprocess_data(train_df)
        X_val, y_val = preprocess_data(val_df)

        # 3. Train the Model
        # k=5 is a standard starting point for k-NN
        print("[INFO] Training k-NN model (this may take a few seconds)...")
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        model.fit(X_train, y_train)
        print("[INFO] Model training complete.")

        # 4. Evaluate on Independent Validation Data
        print("[INFO] Predicting locations for Validation Data...")
        y_pred = model.predict(X_val)

        # Calculate Accuracy
        acc = accuracy_score(y_val, y_pred)

        print("\n" + "="*50)
        print(f" FINAL ACCURACY ON VALIDATION DATA: {acc * 100:.2f}%")
        print("="*50 + "\n")

        # Grafic 1: Distribuția datelor de antrenare (Bun pentru documentație - arată volumul de muncă)
        plot_data_distribution(y_train, title="Training Set")

        # Grafic 2: Performanța modelului (Bun pentru prezentare - arată rezultatele)
        plot_confusion_matrix(y_val, y_pred)

        # 5. Save results to a text file (Optional, useful for documentation)
        print("[INFO] Generating classification report...")
        report = classification_report(y_val, y_pred)
        print(report)
        
        # 6. Real-time Simulation Example
        print("\n[DEMO] Simulating a random user tracking request:")
        random_idx = np.random.randint(0, len(X_val))
        # Keep feature names to avoid sklearn warning about missing names
        sample = X_val.iloc[[random_idx]]

        predicted = model.predict(sample)[0]
        actual = y_val.iloc[random_idx]

        print(f" -> User Signal Fingerprint ID: {random_idx}")
        print(f" -> System Prediction: Building-Floor {predicted}")
        print(f" -> Actual Location:   Building-Floor {actual}")

        if predicted == actual:
            print(" -> RESULT: SUCCESS ✅")
        else:
            print(" -> RESULT: FAILURE ❌")


if __name__ == "__main__":
    main()