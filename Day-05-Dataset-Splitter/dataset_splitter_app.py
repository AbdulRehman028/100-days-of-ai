import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO

st.title("Dataset Splitter and Visualizer")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Options
    target_col = st.selectbox("Select target column", df.columns)
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
    stratify = st.checkbox("Stratify by target", value=False)
    
    if st.button("Split Dataset"):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        stratify_option = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_option, random_state=42
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        st.write("Training Set Preview:")
        st.dataframe(train_df.head())
        st.write("Test Set Preview:")
        st.dataframe(test_df.head())
        
        # Visualize target distribution
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        train_df[target_col].hist(ax=ax[0])
        ax[0].set_title("Train Set Distribution")
        test_df[target_col].hist(ax=ax[1])
        ax[1].set_title("Test Set Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download
        train_csv = train_df.to_csv(index=False)
        test_csv = test_df.to_csv(index=False)
        st.download_button(
            label="Download Train CSV",
            data=BytesIO(train_csv.encode()),
            file_name="train_data.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Test CSV",
            data=BytesIO(test_csv.encode()),
            file_name="test_data.csv",
            mime="text/csv"
        )