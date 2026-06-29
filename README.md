# Image Classification Project

This repository contains an image classification project built using a Convolutional Neural Network (CNN). The model was trained to classify multiple object categories and includes a simple UI for testing predictions.

## Project Structure

```bash
├── EvaluationResults/
│   ├── ClassificationReport.png
│   ├── MisclassifiedSamples.png
│   ├── ModelResults.png
│   └── ModelsConfusionMatrix.png
│
├── ExeProject/
│   ├── CNN_Model.keras
│   ├── class_names.pkl
│
├── ModelDevelopment/
│   ├── CNN_Development.py
│   ├── DataAugmentation.py
│   └── UI.py
│
├── ValidationImage/
│   ├── sample images for testing
│
├── requirements.txt
```

---

## About the Project

This project uses a CNN model to classify images into different categories.  

The dataset used for training was gathered using a custom scraper from the **Divar** website.

The repository includes:

- **Model training scripts**
- **Data augmentation pipeline**
- **Pre-trained model**
- **Class labels**
- **Evaluation metrics and visualizations**
- **Validation images for testing**
- **User Interface for predictions**

---

## How to Use

### 1. Clone the repository

```bash
git clone https://github.com/Adnan-Akbari04/CNN_Model.git
cd Adnan-Akbari04/CNN_Model
pip install -r requirements.txt
```

---

### 2. Install required libraries

Make sure you have Python 3.9+ installed.

Install dependencies:

```bash
pip install tensorflow keras numpy matplotlib pillow scikit-learn pickle-mixin
```
or

```bash
pip install -r requirements.txt
```

---

### 3. Run the UI module

To start the image classification interface:

```bash
python ModelDevelopment/UI.py
```

This will launch the UI where you can upload or test images for prediction.

---

## Model Files

The application depends on these files:

- `ExeProject/CNN_Model.keras` → trained CNN model
- `ExeProject/class_names.pkl` → class labels
- `ExeProject/icon.png` → UI icon

Do not move or rename these files unless you update the paths in `UI.py`.

---

## Validation

You can test the model using images inside:

```bash
ValidationImage/
```

Or use your own images.

---

## Results

Model evaluation outputs can be found in:

```bash
EvaluationResults/
```

Including:

- Classification report
- Confusion matrix
- Misclassified samples
- Performance graphs

---

## Requirements

Main libraries used:

- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

---

## Reusing the Data Augmentation Module

The `DataAugmentation.py` module is not limited to this project. You can reuse it in your own image classification projects with any dataset.

To use it:

1. Place your dataset in any folder structure you want.
2. Open `ModelDevelopment/DataAugmentation.py`.
3. Update the dataset path variable to point to your own dataset directory.
4. Run:

```bash
python ModelDevelopment/DataAugmentation.py
```

This will apply the augmentation pipeline to your dataset and generate augmented images based on the transformations defined in the script.

This makes it easy to:
- expand small datasets
- improve model generalization
- reduce overfitting
- prepare custom datasets for training

**Important:**  
Make sure your dataset follows a class-based folder structure like:

```bash
dataset/
├── class_1/
├── class_2/
├── class_3/
```
so the augmentation pipeline can process the data correctly.

---

## Contact

**Email:** adnanakbari@outlook.com   
**GitHub:** [GitHub profile](https://github.com/Adnan-Akbari04)  
**LinkedIn:** [Linkedin profile](https://www.linkedin.com/in/adnan-akbari/)

---

## Notes

- Make sure all dependencies are installed before running.
- Keep the folder structure intact for the UI to work properly.
- Empty folders are not tracked by GitHub.
