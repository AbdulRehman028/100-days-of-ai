# 📊 Advanced Dataset Cleaner & Explorer

This Streamlit web application provides a user-friendly interface for exploring and cleaning datasets. It allows users to upload a CSV file, view its statistics, apply various data cleaning operations, and then download the cleaned dataset.

---

## ✨ Features

* **CSV Upload**: Easily upload your dataset in CSV format.
* **Dataset Preview**: View the first few rows of your original and cleaned datasets.
* **Initial Statistics**: Get a quick overview of your dataset's shape, missing values, column types, and basic descriptive statistics before cleaning.
* **Missing Data Handling**:
    * Drop rows with any missing values.
    * Impute missing values using Mean, Median, Mode, KNN, Forward Fill (`ffill`), or Backward Fill (`bfill`).
* **Duplicate Data Handling**:
    * Remove exact duplicate rows from your dataset.
* **Outlier Detection & Treatment**:
    * Identify and remove outliers using the Interquartile Range (IQR) method.
    * Cap outliers (Winsorization) using the IQR method.
* **Data Type Corrections**:
    * Convert selected columns to `datetime` format.
* **Standardization & Normalization**:
    * Scale numerical data using `StandardScaler` (Z-score normalization).
    * Scale numerical data using `MinMaxScaler` (normalize to \[0, 1] range).
* **Post-Cleaning Statistics**: View statistics (missing values, types, descriptive stats) for the dataset **after** applying cleaning operations to assess their impact.
* **Download Cleaned Data**: Download the processed dataset as a new CSV file.

---

## 🚀 How to Run the App

### Prerequisites

Make sure you have Python installed (version 3.7 or higher recommended).

### Installation

1.  **Clone this repository** (or copy the code into a file named `app.py`):
    ```bash
    git clone git clone https://github.com/AbdulRehman028/100-days-of-ai.git
    cd Day-02-Dataset Explorer and Cleaner
    ```
  

2.  **Install the required Python packages**:
    ```bash
    pip install streamlit pandas numpy scikit-learn
    ```

### Running the Application

1.  **Navigate to the directory** where you saved `app.py` in your terminal or command prompt.

2.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

3.  Your web browser should automatically open a new tab with the Streamlit application running. If not, open your browser and go to `http://localhost:8501`.

---

## 📖 Usage

1.  **Upload your CSV file** using the "Upload CSV" button in the sidebar.
2.  **Explore Initial Stats**: Check the "Show Initial Dataset Stats" checkbox in the sidebar to view detailed statistics of your original data.
3.  **Apply Cleaning Steps**: Use the expanders in the sidebar (e.g., "1. Handle Missing Values", "2. Handle Duplicates") to select and apply various cleaning operations. Click the "Apply" button within each expander to perform the selected action.
4.  **Review Cleaned Data**: After applying cleaning steps, the "Cleaned Dataset Preview" will update.
5.  **Check Post-Cleaning Stats**: Select the "Show Post-Cleaning Statistics" checkbox to see how the cleaning operations have altered the dataset's characteristics.
6.  **Download Cleaned CSV**: Once satisfied, click the "Download Cleaned CSV" button to save your processed dataset.

---

## 🛠️ Technologies Used

* **Python 3**
* **Streamlit**: For creating the interactive web application.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **scikit-learn**: For advanced imputation (KNNImputer) and scaling (StandardScaler, MinMaxScaler).

---