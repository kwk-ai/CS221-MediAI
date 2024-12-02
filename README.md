# CS221-MediAI: A Health Assistant Project

This repository provides tools and workflows for working with the Kvasir dataset and a private tubular data, cleaning data, training machine learning models, and deploying a Streamlit-based interface for patient symptom analysis. Below is an overview of the repository structure and its usage.

---

## **Repository Structure**

```plaintext
.
├── build_docker.sh              # Script to build the main Docker container
├── Clean_data_3000_with_gas.ipynb # Notebook for tabular data analysis and multiple-algorithms training
├── Dockerfile                   # Dockerfile for the main container
├── kvasir-dataset/              # Folder for Kvasir dataset operations
│   ├── data_preperation.py      # Script to clean and prepare Kvasir dataset
│   ├── inference_test.py        # Script to run inference on test data
│   ├── run_docker.sh            # Script to run Kvasir-related tasks in Docker
│   └── train_model.py           # Script to train the ResNet-50 model
├── run_docker_it.sh             # Script to run the container interactively
├── run_docker.sh                # Script to run the container and start a jupyter notebook instance
└── streamlit/                   # Folder for Streamlit-based interface
    ├── build.sh                 # Script to build Streamlit Docker container
    ├── dockerfile               # Dockerfile for Streamlit app
    ├── run_docker.sh            # Script to run the Streamlit app in Docker
    └── streamlit_code.py        # Streamlit code for the frontend
```

---

## **1. Kvasir Dataset**

### **Description**
The `kvasir-dataset` folder contains tools for handling the [Kvasir Dataset](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset), including data preparation, model training, and inference testing.

### **Steps to Use**
1. **Download the Dataset:**
   - Visit the [Kvasir Dataset Kaggle page](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset).
   - Download the dataset and place it in the `kvasir-dataset` folder.
  
2. **Prepare the Data:**
   - Start the docker by running `run_docker.sh` inside `kvasir-dataset`.
     ```bash
     cd kvasir-dataset
     ./run_docker.sh
     ```

3. **Prepare the Data:**
   - Run `data_preperation.py` to clean and preprocess the dataset:
     ```bash
     python data_preperation.py
     ```

4. **Train the Model:**
   - Use `train_model.py` to train a ResNet-50 model with transfer learning(only update last two layers):
     ```bash
     python train_model.py
     ```
   - The model unfreezes the last two layers for fine-tuning.

5. **Test Inference:**
   - Run `inference_test.py` to test the trained model on new images:
     ```bash
     python inference_test.py
     ```

---

## **2. Main Notebook**

### **Description**
The `Clean_data_3000_with_gas.ipynb` notebook processes private tabular data, trains multiple models, and identifies the best-performing one.

### **Steps to Use**
1. Build Docker and Open the notebook:
   ```bash
   ./build_docker.sh
   ./run_docker.sh
   ```

2. Load your private tabular dataset into the notebook.

3. Train and evaluate models, including XGBoost, which has shown to provide the best results.

---

## **3. Streamlit Application**

### **Description**
The `streamlit` folder contains a Streamlit-based frontend for patient symptom analysis, powered by:
- **Llama 3.1 8B**: Large Language Model (LLM) backend.
- **Streamlit**: Interactive user interface.

### **Features**
- **Image Analysis:** Upload an image (e.g., from the Kvasir dataset) to analyze symptoms using the ResNet-50 model.
- **Tabular Data Analysis:** Upload tabular data to receive insights using the XGBoost model.
- **Symptom Insights:** Provides detailed analysis by integrating both image and tabular data processing.

### **Steps to Use**
1. **Build the Streamlit Docker Container:**
   - Navigate to the `streamlit` folder and run:
     ```bash
     ./build.sh
     ```

2. **Run the Streamlit App:**
   - Launch the Streamlit app in Docker:
     ```bash
     ./run_streamlit_docker.sh
     ```

3. **Access the App:**
   - Open your browser and go to `http://localhost:8501` to interact with the interface.

## **Contributors**
- Yang Liu
- Tong Yin
