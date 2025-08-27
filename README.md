# Antarya Stamatics YOLO Assignment 3

This project implements a **Convolutional Neural Network (CNN)** for handwritten digit recognition (similar to MNIST classification).  
The notebook walks through **data preprocessing, model building, training, evaluation, and prediction submission**.

---

## 🚀 Project Overview

1. **Introduction**  
   The task involves classifying grayscale digit images into one of 10 classes (0–9).  

2. **Data Preparation**  
   - Imported required Python libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras).  
   - Loaded dataset (`train.csv` and `test.csv`).  
   - Preprocessed the dataset: normalization, reshaping, and one-hot encoding of labels.  
   - Split training data into training and validation sets (90% train, 10% validation).  
   - Performed visualization of sample digits.  

3. **Model Architecture (CNN)**  
   - Built a Convolutional Neural Network (CNN) using **Keras/TensorFlow**.  
   - Layers include:
     - Convolutional layers
     - MaxPooling layers
     - Dropout for regularization
     - Fully connected Dense layers  
   - Compiled with `categorical_crossentropy` loss and `adam` optimizer.  

4. **Training**  
   - Trained the CNN model on the training dataset.  
   - Validated performance on a held-out validation dataset.  

5. **Evaluation**  
   - Used metrics such as **accuracy** and **confusion matrix**.  

6. **Prediction and Submission**  
   - Generated predictions for test dataset.  
   - Created a submission file (`submission.csv`).  

---

## 🛠️ Installation & Setup

### Requirements
Make sure you have the following installed:
- Python 3.8+
- Jupyter Notebook / Jupyter Lab
- Required libraries (install via pip):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

---

## ▶️ How to Run the Project

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. Place the dataset files:
   - `train.csv`
   - `test.csv`

   in the project directory.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Antarya_Stamatics_YOLO_Assignment_3.ipynb
   ```

4. Run all cells **sequentially**:
   - Data preprocessing
   - Model definition
   - Training
   - Evaluation
   - Prediction

5. The final submission file (`submission.csv`) will be generated in the working directory.

---

## 📊 Results

- Achieved high accuracy on validation dataset.
- Confusion matrix provides insights into model misclassifications.
- Predictions are saved for submission.

---

## 📂 File Structure

```
├── Antarya_Stamatics_YOLO_Assignment_3.ipynb   # Main notebook
├── train.csv                                   # Training dataset
├── test.csv                                    # Testing dataset
├── submission.csv                              # Final output file (after running notebook)
└── README.md                                   # Documentation
```

---

## 📌 Notes
- You can adjust **epochs, batch size, and learning rate** for better performance.
- Additional data augmentation techniques can be applied to improve accuracy.

---

## 👨‍💻 Author
Developed as part of **Antarya Stamatics YOLO Assignment 3**.  
