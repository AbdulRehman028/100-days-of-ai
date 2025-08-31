## 🚀 Overview  
The **Dataset Splitter and Visualizer** is a **Streamlit-based web app** that lets you:  
- Upload any CSV dataset  
- Select a target column  
- Choose a custom train/test split ratio  
- Enable optional **stratification** by target  
- Visualize target distribution with histograms  
- Download split datasets as CSV files  

Perfect for **ML dataset preparation**, **class distribution validation**, or **data exploration** before modeling.

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)  
![Python](https://img.shields.io/badge/Language-Python-blue?logo=python)  
![Pandas](https://img.shields.io/badge/Library-Pandas-yellowgreen)  
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange)  
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-lightgrey)

![App Screenshot](your-app-screenshot.png) <!-- Replace with actual image -->


## ✨ Features  
- 📁 Upload CSV datasets  
- 🎯 Select target column for ML  
- 🎚️ Adjustable train/test split  
- ⚖️ Optional **stratified sampling**  
- 📊 Side-by-side histograms for label distribution  
- 🧾 Preview train and test DataFrames  
- 📥 Download split CSV files  


## 🛠️ Tech Stack  
- **Python** 🐍  
- **Streamlit** 🎨 – Interactive web interface  
- **Pandas** 📊 – Data manipulation  
- **Scikit-learn** 🔀 – Dataset splitting  
- **Matplotlib** 📈 – Visualization  
- **BytesIO** 💾 – File download handler  


## 📂 Workflow  
1. **Upload** → Load your CSV file  
2. **Configure** → Choose target column, test size, and stratification  
3. **Split** → Automatically divide into train and test sets  
4. **Visualize** → Check target distribution in both sets  
5. **Download** → Save train/test data as `.csv` files  


## 🎯 Use Cases  
- 🧠 Train/Test splitting for machine learning  
- 📉 Verifying label distribution after stratification  
- 🛠️ Data preprocessing pipelines  
- 👩‍🏫 Teaching/train-test split concepts  


## ⚡ Installation & Usage

```bash
# Clone the repository
git clone git clone https://github.com/AbdulRehman028/100-days-of-ai.git
cd Day-05-Dataset-Splitter

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

## 👨‍💻 Developer
Developed by **M.AbdulRehman Baig** ❤️

---

⭐ **If you found this project helpful, please give it a star!** ⭐