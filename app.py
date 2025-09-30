import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from shap_exp import explain_model_predictions

st.title("Batch Medical Report Integrity Checker")
st.write(
    "Upload multiple candidate medical reports (CSV or Excel) to automatically detect potential alterations. "
    "The system flags reports where lab values are abnormal or contradict expected physiological patterns. "
    "This helps identify suspicious or tampered reports, ensuring reliability of medical certifications."
)
#Load saved artifacts
with open("Best_Model/preproc.pkl", "rb") as f:
    preproc = pickle.load(f)

with open("Best_Model/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Best_Model/required_columns.pkl", "rb") as f:
    required_columns = pickle.load(f)


uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# if uploaded_file:
#     if uploaded_file.name.endswith(".csv"):
#         data = pd.read_csv(uploaded_file)
#     else:
#         data = pd.read_excel(uploaded_file)

#     st.write("### Original Data Preview:")
#     st.dataframe(data.head(10))

#  #Ensure all required columns are present
#     for col in required_columns:
#         if col not in data.columns:
#             data[col] = 0  
#     df = data[required_columns]  

#     #Data preprocessing
#     X_proc = preproc.transform(df)

#     #Model Prediction
#     predictions = model.predict(X_proc)
#     prediction_prob = model.predict_proba(X_proc)[:, 1]

#     #Append predictions to original DataFrame
#     data['Predicted_Target_Label'] = predictions
#     data['Probability_Altered'] = prediction_prob

#     st.write("### Prediction Results:")
#     st.dataframe(data.head(10))

#     #Evaluation Metrics (if actual labels are available)
#     if 'Target_Label' in data.columns:
#         actual_labels = np.where(data['Target_Label'] == 'Altered', 1, 0)

#         metrics = {
#         "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
#         "Score": [
#             accuracy_score(actual_labels, predictions),
#             precision_score(actual_labels, predictions),
#             recall_score(actual_labels, predictions),
#             f1_score(actual_labels, predictions)
#             ]
#             }
        
#         metrics_df = pd.DataFrame(metrics)
#         st.write("### Evaluation Metrics")
#         st.dataframe(metrics_df)

#         # Confusion matrix as separate table
#         conf_mat = confusion_matrix(actual_labels, predictions)
#         conf_df = pd.DataFrame(conf_mat, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])

#         st.write("#### Confusion Matrix")
#         st.dataframe(conf_df)

#     # --- Download Predictions ---
#     st.download_button(
#         label="Download Predictions as CSV",
#         data=df.to_csv(index=False),
#         file_name="predicted_results.csv",
#         mime="text/csv"
#     )

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("### Original Data Preview:")
    st.dataframe(data.head(10))

    #Ensure all required columns are present
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0  
    df = data[required_columns]  

    # Identify numerical and categorical columns from the uploaded dataset
    num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()


    # Data preprocessing
    X_proc = preproc.transform(df)

    # Model Prediction
    predictions = model.predict(X_proc)
    prediction_prob = model.predict_proba(X_proc)[:, 1]

    # Append predictions to original DataFrame
    data['Predicted_Target_Label'] = predictions
    data['Probability_Altered'] = prediction_prob

    #Compute SHAP values and top feature contributions
    recordwise_reasoning_df = explain_model_predictions(
        model=model,
        X_data=X_proc,
        num_cols=num_cols,
        cat_cols=cat_cols,
        preproc=preproc,
        top_n=3,
        threshold=0.5
    )

    # Merge 'feature_shap_values' after 'Probability_Altered'
    data.insert(
        loc=data.columns.get_loc('Probability_Altered') + 1,  # after Probability_Altered
        column='feature_shap_values',
        value=recordwise_reasoning_df['feature_shap_values']
    )

    data.rename(columns={'feature_shap_values': 'Top_Feature_Contributions'}, inplace=True)

    st.write("### Prediction Results with Feature Contributions:")
    st.dataframe(data.head(10))

    # Evaluation Metrics (if actual labels are available)
    if 'Target_Label' in data.columns:
        actual_labels = np.where(data['Target_Label'] == 'Altered', 1, 0)
        metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Score": [
                accuracy_score(actual_labels, predictions),
                precision_score(actual_labels, predictions),
                recall_score(actual_labels, predictions),
                f1_score(actual_labels, predictions)
            ]
        }
        metrics_df = pd.DataFrame(metrics)
        st.write("### Evaluation Metrics")
        st.dataframe(metrics_df)

        # Confusion matrix
        conf_mat = confusion_matrix(actual_labels, predictions)
        conf_df = pd.DataFrame(conf_mat, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
        st.write("#### Confusion Matrix")
        st.dataframe(conf_df)

    # --- Download Predictions ---
    st.download_button(
        label="Download Predictions as CSV",
        data=data.to_csv(index=False),
        file_name="predicted_results.csv",
        mime="text/csv"
    )
