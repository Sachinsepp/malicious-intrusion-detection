# 🚨 Real-Time Malicious Intrusion Detection

## 🔍 Overview
This project presents an advanced, intelligent intrusion detection and attack mitigation framework tailored for IoT-empowered cybersecurity systems. Utilizing the UNSW-NB15 dataset and deep learning methodologies—notably Generative Adversarial Networks (GANs)—the system is engineered to detect, classify, and respond to real-time network threats and anomalies across smart infrastructures.

A Tkinter-based GUI streamlines user interactions, supporting dataset upload, preprocessing, training, and prediction—enabling seamless integration into real-world cybersecurity operations.

## 🎯 Objectives
- Leverage GANs to identify and simulate cyberattacks in IoT networks.
- Enhance cybersecurity through real-time detection and classification of intrusions.
- Provide visual feedback and intelligent recommendations based on attack patterns.
- Offer a modular, extensible framework suitable for academic research, industry, or internships.

## 🌐 Key Features

- 📁 **Dataset Upload**: User-friendly interface to upload UNSW-NB15 in CSV format.
- ⚙️ **Automated Preprocessing**: Feature normalization, label encoding, and shuffling.
- 📊 **Train-Test Split**: Robust data partitioning using `train_test_split` (80/20 split).
- 🧠 **Deep Learning with GANs**: Anomaly detection and classification using advanced adversarial learning.
- 🔍 **Attack Prediction**: Real-time prediction of malicious intrusions.
- 📈 **Metrics & Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and visual graphs.
- 📊 **Visualization Tools**: Interactive plots for performance analysis and comparison.
- 💾 **Model Persistence**: Model weights and training history saved via pickle.

## 🧪 Dataset - UNSW-NB15
A benchmark dataset developed by the Australian Centre for Cyber Security (ACCS), containing 49 features and 9 attack types, including:

- DoS
- DDoS
- Backdoors
- Exploits
- Worms
- Reconnaissance
- Shellcode
- Fuzzers
- Generic

Core Columns:
- `proto`: Protocol used
- `service`: Type of network service
- `state`: Connection status
- `attack_cat`: Attack classification
- `label`: Binary label (0 = Normal, 1 = Attack)

## 🧰 Technologies Used

| Area                | Technology                         |
|---------------------|-------------------------------------|
| Programming         | Python 3.x                          |
| GUI                 | Tkinter                             |
| Deep Learning       | Keras, TensorFlow                   |
| ML & Preprocessing  | Scikit-learn, Pandas, Numpy         |
| Visualization       | Matplotlib, Seaborn                 |
| Model Persistence   | Pickle                              |

 <section>
    <h2>📁 File Structure</h2>
    <pre>
.
├── dataset/
│   └── UNSW-NB15.csv
├── model/
│   ├── gan_weights.hdf5
│   └── gan_history.pckl
├── main.py
├── requirements.txt
└── README.md
    </pre>
  </section>

## ⚙️ Installation & Setup

### Step 1: Clone Repository

git clone https://github.com/your-username/RealTimeMaliciousDetection.git
cd RealTimeMaliciousDetection

### Step 2: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
### Step 3: Run the Application
bash
Copy
Edit
python main.py
### 🔄 Application Workflow
Upload Dataset: Load the CSV file using the GUI.

Preprocess Data: Normalize features, encode labels, and clean missing data.

Split Data: 80/20 training-testing split for model validation.

Train GAN: Train adversarial models to learn attack representations.

Predict Attacks: Use trained models to classify new data as normal or malicious.

Performance Metrics: Display Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

Graphical Output: Visual comparisons between metrics and across models.

### 🔧 Code Highlights
python
Copy
Edit
def uploadDataset():
    # Loads and verifies the CSV file integrity, displays summary stats.
    pass

def preprocessing():
    # Applies normalization (MinMaxScaler), categorical encoding, and data cleaning.
    pass

def dataSplit():
    # Splits the dataset with a fixed random seed for reproducibility.
    pass

def runGAN():
    # Constructs and trains GAN model using convolutional layers, dropout, and batch norm.
    pass

def attackPrediction():
    # Predicts attack type from test set and displays classification results.
    pass

def calculateMetrics():
    # Computes confusion matrix, accuracy, precision, recall, and F1-score.
    pass
### 📊 Performance Metrics
Metric	Description
Accuracy	Correct predictions over total predictions
Precision	TP / (TP + FP) – How many predicted attacks were correct?
Recall	TP / (TP + FN) – How many actual attacks were detected?
F1-Score	Harmonic mean of precision and recall

Where:

TP = True Positives

FP = False Positives

FN = False Negatives

### 🚀 Future Enhancements
✅ Real-time Data Integration: Connect with IoT sensors for live monitoring.

🔄 Model Optimization: Hyperparameter tuning, architecture enhancements.

🔁 Cross-Validation: K-Fold CV for more robust evaluation.

🧩 Multi-dataset Support: Support for NSL-KDD, CICIDS2017, etc.

🧠 Explainable AI: Add SHAP/ELI5 visual explanations for predictions.

### 🌐 Use Cases
Smart city infrastructure security.

Industrial IoT (IIoT) attack detection.

University/college research in cybersecurity & AI.

Internship projects in ML & network defense.

### 📬 Contact
Maintainer: Shabreen Taj

🔗 GitHub: https://github.com/taj-shabreen
