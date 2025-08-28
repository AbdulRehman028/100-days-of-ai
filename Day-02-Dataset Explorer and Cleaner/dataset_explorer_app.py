import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

# Main app title
st.set_page_config(page_title="Advanced Dataset Cleaner", layout="wide")
st.title("Advanced Dataset Cleaner & Explorer")

# File uploader section
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Original Dataset Preview")
        st.dataframe(df.head())

        # --- Initial Data Stats ---
        if st.sidebar.checkbox("Show Initial Dataset Stats"):
            st.write("### Initial Dataset Statistics")
            st.write("Missing Values:")
            st.write(df.isnull().sum())
            st.write("Column Types:")
            st.write(df.dtypes)
            st.write("Basic Stats:")
            st.write(df.describe(include='all'))

        df_clean = df.copy() # Use a copy for all cleaning operations
        st.sidebar.subheader("Cleaning Steps")

        # --- Missing Data Handling ---
        with st.sidebar.expander("1. Handle Missing Values"):
            missing_method = st.radio("Select method", ["Do nothing", "Drop rows", "Impute"])
            if missing_method == "Impute":
                impute_type = st.selectbox("Imputation type", ["Mean", "Median", "Mode", "KNN", "ffill", "bfill"])
            
            if st.button("Apply Missing Data Handling"):
                if missing_method == "Drop rows":
                    df_clean.dropna(inplace=True)
                    st.success("Successfully dropped all rows with missing values.")
                elif missing_method == "Impute":
                    try:
                        if impute_type in ["Mean", "Median", "Mode"]:
                            for col in df_clean.columns:
                                if df_clean[col].isnull().sum() > 0:
                                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                                        fill_value = df_clean[col].mean() if impute_type == "Mean" else df_clean[col].median()
                                        df_clean[col].fillna(fill_value, inplace=True)
                                    else:
                                        # Use a try-except block for mode as it can fail on empty data
                                        try:
                                            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True) # Mode for non-numeric
                                        except IndexError:
                                            st.warning(f"No mode found for column '{col}'. Skipping.")

                        elif impute_type == "KNN":
                            # KNN only works on numeric columns
                            numeric_cols = df_clean.select_dtypes(include=np.number).columns
                            imputer = KNNImputer(n_neighbors=5)
                            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                        elif impute_type == "ffill":
                            df_clean.fillna(method='ffill', inplace=True)
                        elif impute_type == "bfill":
                            df_clean.fillna(method='bfill', inplace=True)
                        st.success(f"Successfully imputed missing values using **{impute_type}**.")
                    except Exception as e:
                        st.error(f"Error during imputation: {e}")
                
        # --- Duplicate Data Handling ---
        with st.sidebar.expander("2. Handle Duplicates"):
            remove_duplicates = st.checkbox("Remove exact duplicate rows")
            if st.button("Apply Duplicate Handling"):
                if remove_duplicates:
                    initial_rows = len(df_clean)
                    df_clean.drop_duplicates(inplace=True)
                    rows_removed = initial_rows - len(df_clean)
                    st.success(f"Successfully removed **{rows_removed}** duplicate rows.")

        # --- Outlier Detection & Treatment ---
        with st.sidebar.expander("3. Treat Outliers"):
            outlier_method = st.selectbox("Method", ["Do nothing", "IQR (Remove)", "IQR (Cap)"])
            if st.button("Apply Outlier Treatment"):
                if outlier_method != "Do nothing":
                    try:
                        numeric_cols = df_clean.select_dtypes(include=np.number).columns
                        for col in numeric_cols:
                            Q1 = df_clean[col].quantile(0.25)
                            Q3 = df_clean[col].quantile(0.75)
                            IQR = Q3 - Q1
                            if IQR > 0:
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                if outlier_method == "IQR (Remove)":
                                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                                    st.success(f"Removed outliers in **{col}**.")
                                elif outlier_method == "IQR (Cap)":
                                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                                    st.success(f"Capped outliers in **{col}**.")
                    except Exception as e:
                        st.error(f"Error during outlier treatment: {e}")

        # --- Data Type Corrections ---
        with st.sidebar.expander("4. Correct Data Types"):
            convert_to_datetime = st.checkbox("Convert to Datetime")
            if convert_to_datetime:
                date_cols = st.multiselect("Select date columns", df_clean.columns)
            
            if st.button("Apply Data Type Corrections"):
                if convert_to_datetime and date_cols:
                    for col in date_cols:
                        try:
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                            st.success(f"Successfully converted '{col}' to datetime format.")
                        except Exception as e:
                            st.error(f"Could not convert '{col}' to datetime: {e}")

        # --- Standardization & Normalization ---
        with st.sidebar.expander("5. Scale Numerical Data"):
            scaling_method = st.selectbox("Scaling Method", ["Do nothing", "Standard Scaler", "Min-Max Scaler"])
            if st.button("Apply Scaling"):
                if scaling_method != "Do nothing":
                    try:
                        numeric_cols = df_clean.select_dtypes(include=np.number).columns
                        if scaling_method == "Standard Scaler":
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                        
                        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
                        st.success(f"Successfully applied **{scaling_method}** to numeric columns.")
                    except Exception as e:
                        st.error(f"Error during scaling: {e}")
        
        # --- Display Cleaned Dataset ---
        st.write("---")
        st.write("### Cleaned Dataset Preview")
        st.write(f"Final shape of the dataset: **{df_clean.shape}**")
        st.dataframe(df_clean.head())
        
        # --- Add new feature: Show cleaned stats ---
        if st.checkbox("Show Post-Cleaning Statistics"):
            st.write("### Post-Cleaning Statistics")
            st.write("Missing Values:")
            st.write(df_clean.isnull().sum())
            st.write("Column Types:")
            st.write(df_clean.dtypes)
            st.write("Basic Stats:")
            st.write(df_clean.describe(include='all'))

        # --- Download Button ---
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")