# **ML Pipeline Projects 🚀**

Welcome to the **ML_Pipeline** repository, a collection of multiple machine learning projects and implementations. This repository includes a variety of end-to-end machine learning pipelines focused on solving real-world problems such as disease detection, sentiment analysis, recommendation systems, and more.

Each project follows a structured approach, with data preprocessing, feature engineering, model training, evaluation, and prediction. You'll find Jupyter notebooks, datasets, and scripts to understand and reproduce each project.

---

## **📁 Project Structure**
```
ML_Pipeline/
├── Book Recommendations System/
│   ├── Book Recommendation System with Python.ipynb
│   ├── books_data.csv
│
├── Breast Cancer Prediction Using ML/
│   ├── Advance Project Breast Cancer Prediction Using ML.ipynb
│   ├── breast_cancer.csv
│   ├── brest_cancer.pkl
│   ├── PE_breast_cancer.jpeg
│   └── roc_breast_cancer.jpeg
│
├── Heart Disease Prediction/
│   ├── Heart Disease Prediction With Machine Learning.ipynb
│   └── heart.csv
│
├── Hybrid Recommendation System using Python/
│   ├── Hybrid Recommendation System using Python.ipynb
│   └── fashion_products.csv
│
├── Liver Disease Prediction Using ML/
│   ├── Advance Project Liver Disease Prediction Using ML.ipynb
│   ├── liver.csv
│   ├── liver.pkl
│   ├── PE_liver.jpeg
│   └── roc_liver.jpeg
│
├── Movie Recommender System Using Python/
│   ├── Movie Recommendation System.ipynb
│   ├── movies.csv
│   └── ratings.csv
│
├── Online Payments Fraud Detection/
│   └── Online Payments Fraud Detection.ipynb
│
├── Parkinson's Disease Detection/
│   ├── Parkinson's Disease Detection.ipynb
│   └── dataset.csv
│
├── Text Emotions Classification/
│   ├── Text Emotions Classification using Python.ipynb
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
├── Tiktok Reviews Sentimental Analysis/
│   └── TikTok Reviews Sentiment Analysis using Python.ipynb
│
├── energy_consumption_yt-master/
│   ├── 0_yt_EDA.ipynb
│   ├── README.md
│   └── dataset/
│       ├── LoureiroDataset.zip
│       ├── LoureiroDataset/loureiro_energy.csv
│       └── LoureiroDataset/weather_aveiro_final.csv
│
└── house-prices-advanced-regression-techniques/
    └── house-prices-advanced-regression-techniques.zip
```

---

## **📚 Projects Overview**
Here is a brief description of each project included in this repository.

| **Project**                     | **Description**                                               |
|---------------------------------|-------------------------------------------------------------|
| **Book Recommendation System**  | Collaborative filtering-based system to recommend books.    |
| **Breast Cancer Prediction**    | Predicts if a tumor is benign or malignant.                 |
| **Heart Disease Prediction**    | Predicts heart disease risk using ML models.                |
| **Hybrid Recommendation System**| Combines collaborative and content-based recommendations.    |
| **Liver Disease Prediction**    | Detects liver disease risk based on clinical data.          |
| **Movie Recommender System**    | Recommends movies using collaborative filtering.            |
| **Online Payments Fraud Detection** | Detects fraudulent payment transactions.                  |
| **Parkinson's Disease Detection** | Classifies Parkinson's disease using biomedical data.      |
| **Text Emotions Classification**| Classifies emotions from text into categories.             |
| **TikTok Sentiment Analysis**    | Analyzes TikTok reviews for positive/negative sentiment.  |
| **Energy Consumption Analysis**  | Analyzes energy consumption trends for forecasting.        |
| **House Prices Regression**      | Predicts house prices using regression techniques.         |

---

## **🔧 Installation Instructions**
To run the projects locally, you will need to install the required libraries.

1. **Clone the repository**
   ```bash
   git clone git@github.com:Amiche02/ML_Pipeline.git
   cd ML_Pipeline
   ```

2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

> **Note**: Each subdirectory may have its own set of dependencies specified within the notebooks. You may need to install additional libraries as required (e.g., `scikit-learn`, `pandas`, `numpy`, `matplotlib`, etc.).

---

## **💻 Usage**
To explore the projects, follow these steps:

1. Open the project folder you're interested in (e.g., **Breast Cancer Prediction Using ML**).
2. Launch the Jupyter Notebook file (`.ipynb`).
3. Follow the steps inside the notebook, from data loading to model evaluation.

Each notebook is self-explanatory, with comments and explanations to guide you through the entire machine learning pipeline.

---

## **📈 Machine Learning Techniques**
This repository demonstrates several machine learning concepts and techniques, including:

- **Data Preprocessing**: Handling missing values, encoding categorical data, and feature scaling.
- **Data Visualization**: Using Matplotlib, Seaborn, and other libraries to visualize data distributions.
- **Model Building**: Regression, classification, and clustering models.
- **Evaluation Metrics**: Confusion matrices, accuracy, precision, recall, F1-score, and ROC curves.
- **Feature Engineering**: Feature selection and dimensionality reduction.
- **Deployment**: Preparing models for deployment (exporting to `.pkl` files).

---

## **📂 Datasets**
Each project contains its own dataset, found within its corresponding folder. The datasets have been sourced from public repositories like **Kaggle** and other public sources. Here are some of the datasets used:

- **books_data.csv** - Data on books for the recommendation system.
- **breast_cancer.csv** - Clinical data to predict breast cancer.
- **liver.csv** - Liver disease prediction dataset.
- **heart.csv** - Heart disease prediction dataset.
- **fashion_products.csv** - E-commerce product data for recommendations.
- **tiktok_reviews.csv** - Sentiment analysis on TikTok reviews.
- **energy_consumption.csv** - Energy consumption dataset for trend analysis.

> **Note**: Some datasets may require unpacking (`.zip` files). Follow the instructions inside each notebook.

---

## **📦 Dependencies**
These are the key Python libraries used in this repository. Each project may have additional dependencies.

- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models
- **XGBoost & LightGBM**: Advanced machine learning models
- **NLTK & TextBlob**: Natural Language Processing (NLP)
- **TensorFlow & Keras**: Deep learning models
- **Jupyter Notebook**: Interactive Python notebooks

To install all required dependencies:
```bash
pip install -r requirements.txt
```

---

## **🛠️ Features**
- Multiple machine learning projects in one place.
- End-to-end pipelines with data cleaning, modeling, and evaluation.
- Easy to understand, reproducible Jupyter Notebooks.
- Pre-trained models and serialized files (`.pkl` files) to save time.
- Datasets included in each project directory.

---

## **🤝 Contribution**
If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make changes and commit (`git commit -m 'Add a new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## **📜 License**
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it as you like.

---

## **🎉 Acknowledgments**
A big thank you to all the contributors and public datasets from **Kaggle** and other sources.

If this project has helped you, please give it a ⭐️ on GitHub.

---

## **📞 Contact**
If you have any questions, issues, or feedback, feel free to reach out.

**Author**: [Stéphane Kpoviessi](https://github.com/Amiche02)  
**Email**: oastephaneamiche@gmail.com  
