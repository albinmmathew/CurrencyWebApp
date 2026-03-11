# Indian Currency Recognition Web App

This is a Flask-based web application that uses deep learning models (MobileNetV2 and ResNet50) to recognize and classify Indian currency notes.

## Features
- Upload an image of an Indian currency note.
- Real-time classification using pre-trained models.
- Support for multiple denominations (10, 20, 50, 100, 200, 500, 2000).

## Setup & Installation

> [!IMPORTANT]
> **Python Version Requirement:** This project requires **Python 3.9 to 3.12**. (TensorFlow is currently not fully compatible with Python 3.13+).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/albinmmathew/CurrencyWebApp.git
   cd CurrencyWebApp
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

## Training New Models

If you want to retrain the models or create new ones, follow these steps using the scripts in the root directory or the `training/` folder.

### 1. Dataset Preparation
- Download the dataset from Kaggle: [Indian Currency Note Images Dataset (2020)](https://www.kaggle.com/datasets/vishalmane109/indian-currency-note-images-dataset-2020)
- Extract the downloaded zip file into your root **Project** folder.
- **IMPORTANT:** Rename the extracted folder to simply `dataset`.
- Your **Local Project Structure** should look like this:

```text
Project/ (Local Root Folder)
├── dataset/                     # Kaggle Dataset (Download & Rename)
│   └── Indian currency dataset v1/
├── class_indices.txt            # Class Mapping
├── train_mobilenet.py           # Training script (MobileNetV2)
├── train_resnet.py              # Training script (ResNet50)
├── compare_models.py            # Comparison & Confusion Matrices
├── requirements.txt             # Dependencies for training
└── CurrencyWebApp/              # Git Repository
    ├── training/                # Scripts (Copy to root folder to retrain)
    │   ├── train_mobilenet.py
    │   ├── train_resnet.py
    │   ├── compare_models.py
    │   └── requirements.txt     # Dependencies for training
    ├── models/                  # Latest .h5 model files
    ├── static/                  # UI Assets (CSS, JS)
    ├── templates/               # HTML Templates
    ├── app.py                   # Flask Application
    ├── requirements.txt         # Dependencies
    └── class_indices.txt        # Mapping file copy
```

### 2. Retraining & Evaluation
Retraining should be done in the **Root Folder** where the `dataset/` is located. 

**Steps:**
1. **Copy Necessary Files:** Copy the following files from `CurrencyWebApp/training/` to your root folder:
   - `train_mobilenet.py`
   - `train_resnet.py`
   - `compare_models.py`
   - `requirements.txt`
   - `class_indices.txt`

2. **Activate Virtual Environment:**
   Ensure you are in the project folder and your venv is active:
   ```bash
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Training:**
   ```bash
   python train_mobilenet.py  # For MobileNetV2
   # OR
   python train_resnet.py     # For ResNet50
   ```

4. **Evaluate & Compare:**
   ```bash
   python compare_models.py
   ```
   This will generate visual reports (`.png` files) and classification reports (`.txt` files) in the root folder.

### 3. Deploying New Models
- Copy the newly generated `.h5` files into the `models/` directory.
- Restart the Flask server to use the updated models.

## Project Structure
- `app.py`: Main Flask application.
- `models/`: Contains pre-trained Keras model files.
- `static/`: CSS and client-side JavaScript.
- `templates/`: HTML templates.
- `uploads/`: Temporary storage for uploaded images.
