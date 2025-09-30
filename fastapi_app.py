# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse, StreamingResponse
# import pandas as pd
# import numpy as np
# import pickle
# import io
# from fastapi.middleware.cors import CORSMiddleware
# from shap_exp import explain_model_predictions

# # Initialize FastAPI app
# app = FastAPI(title="Medical Report Integrity Checker API")

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load saved artifacts once at startup
# with open(r"Best_Model/preproc.pkl", "rb") as f:
#     preproc = pickle.load(f)

# with open(r"Best_Model/xgb_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open(r"Best_Model/required_columns.pkl", "rb") as f:
#     required_columns = pickle.load(f)


# @app.post("/predict")
# async def predict(file: UploadFile = File(...), return_csv: bool = False, top_n: int = 3):
#     """
#     Upload a CSV or Excel file containing medical reports.
#     Returns predictions and probability of 'Altered' reports.
#     Set return_csv=True to download a CSV instead of JSON.
#     """
#     try:
#         # Read uploaded file
#         if file.filename.endswith(".csv"):
#             data = pd.read_csv(file.file)
#         elif file.filename.endswith((".xls", ".xlsx")):
#             data = pd.read_excel(file.file)
#         else:
#             raise HTTPException(status_code=400, detail="Invalid file type. Use CSV or Excel.")


#         # num_cols = ['Hemoglobin', 'Hematocrit', 'WBC', 'Neutrophils', 'Lymphocytes', 'Platelets', 'ALT', 'AST', 'Bilirubin', 'TotalProtein', 'Albumin', 'HbA1c', 'FastingGlucose', 'Creatinine', 'BUN', 'Sodium', 'Potassium', 'ALP', 'Cholesterol', 'LDL', 'HDL', 'Triglycerides', 'Eosinophils', 'RandomGlucose']
#         # cat_cols = ['HBsAg', 'HBV_DNA', 'HAV_IgM', 'RPR', 'Treponemal', 'Urine_RBC', 'Urine_Blood', 'Urine_Protein', 'ABO_BloodGrp', 'Rh_Factor', 'Parasites', 'SampleType']

#         # Ensure all required columns exist
#         for col in required_columns:
#             if col not in data.columns:
#                 data[col] = 0
#         df = data[required_columns]

#         # Identify numerical and categorical columns from the uploaded dataset
#         num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
#         cat_cols = df.select_dtypes(include=['object']).columns.tolist()

#         # Preprocess data
#         X_proc = preproc.transform(df)

#         # Predict
#         predictions = model.predict(X_proc)
#         prediction_prob = model.predict_proba(X_proc)[:, 1]

#         # Append results
#         data['Predicted_Target_Label'] = predictions
#         data['Probability_Altered'] = prediction_prob

#         #Compute SHAP values and top feature contributions
#         recordwise_reasoning_df = explain_model_predictions(
#             model=model,
#             X_data=X_proc,
#             num_cols=num_cols,
#             cat_cols=cat_cols,
#             preproc=preproc,
#             top_n=top_n,
#             threshold=0.5
#         )

#         # Merge 'feature_shap_values' after 'Probability_Altered'
#         data.insert(
#             loc=data.columns.get_loc('Probability_Altered') + 1,  # after Probability_Altered
#             column='feature_shap_values',
#             value=recordwise_reasoning_df['feature_shap_values']
#         )

#         data.rename(columns={'feature_shap_values': 'Top_Feature_Contributions'}, inplace=True)

#         # Return CSV if requested
#         if return_csv:
#             stream = io.StringIO()
#             data.to_csv(stream, index=False)
#             response = StreamingResponse(
#                 iter([stream.getvalue()]),
#                 media_type="text/csv"
#             )
#             response.headers["Content-Disposition"] = "attachment; filename=predicted_results.csv"
#             return response

#         # Convert datetime columns to string before JSON serialization
#         data_for_json = data.copy()
#         for col in data_for_json.select_dtypes(include=['datetime64[ns]']).columns:
#             data_for_json[col] = data_for_json[col].astype(str)

#         # Replace NaN and infinite values with None
#         # data_for_json = data_for_json.replace([np.inf, -np.inf], np.nan)
#         data_for_json = data_for_json.where(pd.notnull(data_for_json), None)

#         # Return JSON
#         result = data_for_json.to_dict(orient="records")
#         return JSONResponse(content={"predictions": result})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import pickle
import io
from fastapi.middleware.cors import CORSMiddleware
from shap_exp import explain_model_predictions
from logger_setup import setup_logger

# Initialize FastAPI app
app = FastAPI(title="Medical Report Integrity Checker API")

logger = setup_logger()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved artifacts once at startup
try:
    with open(r"Best_Model/preproc.pkl", "rb") as f:
        preproc = pickle.load(f)
    with open(r"Best_Model/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(r"Best_Model/required_columns.pkl", "rb") as f:
        required_columns = pickle.load(f)
    logger.info("Model and preprocessing artifacts loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model artifacts.")
    raise RuntimeError(f"Model artifacts loading failed: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...), return_csv: bool = False, top_n: int = 3):
    """
    Upload a CSV or Excel file containing medical reports.
    Returns predictions and probability of 'Altered' reports.
    Set return_csv=True to download a CSV instead of JSON.
    """
    try:
        # Read uploaded file
        if file.filename.endswith(".csv"):
            data = pd.read_csv(file.file)
        elif file.filename.endswith((".xls", ".xlsx")):
            data = pd.read_excel(file.file)
        else:
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid file type. Use CSV or Excel.")
        logger.info(f"File {file.filename} loaded successfully with shape {data.shape}.")

        # Ensure all required columns exist
        for col in required_columns:
            if col not in data.columns:
                data[col] = 0
        df = data[required_columns]

        # Identify numerical and categorical columns
        num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Preprocess data
        try:
            X_proc = preproc.transform(df)
        except Exception as e:
            logger.exception("Error during preprocessing.")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

        # Predict
        try:
            predictions = model.predict(X_proc)
            prediction_prob = model.predict_proba(X_proc)[:, 1]
        except Exception as e:
            logger.exception("Error during model prediction.")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        # Append results
        data['Predicted_Target_Label'] = predictions
        data['Probability_Altered'] = prediction_prob

        num_cols = ['Hemoglobin', 'Hematocrit', 'WBC', 'Neutrophils', 'Lymphocytes', 'Platelets', 'ALT', 'AST', 'Bilirubin', 'TotalProtein', 'Albumin', 'HbA1c', 'FastingGlucose', 'Creatinine', 'BUN', 'Sodium', 'Potassium', 'ALP', 'Cholesterol', 'LDL', 'HDL', 'Triglycerides', 'Eosinophils', 'RandomGlucose']
        cat_cols = ['HBsAg', 'HBV_DNA', 'HAV_IgM', 'RPR', 'Treponemal', 'Urine_RBC', 'Urine_Blood', 'Urine_Protein', 'ABO_BloodGrp', 'Rh_Factor', 'Parasites', 'SampleType']


        # Compute SHAP values and top feature contributions
        try:
            recordwise_reasoning_df = explain_model_predictions(
                model=model,
                X_data=X_proc,
                num_cols=num_cols,
                cat_cols=cat_cols,
                preproc=preproc,
                top_n=top_n,
                threshold=0.5
            )
        except Exception as e:
            logger.exception("Error during SHAP explanation computation.")
            raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")

        # Merge 'feature_shap_values' after 'Probability_Altered'
        data.insert(
            loc=data.columns.get_loc('Probability_Altered') + 1,
            column='feature_shap_values',
            value=recordwise_reasoning_df['feature_shap_values']
        )
        data.rename(columns={'feature_shap_values': 'Top_Feature_Contributions'}, inplace=True)


        # Return CSV if requested
        if return_csv:
            stream = io.StringIO()
            data.to_csv(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv"
            )
            response.headers["Content-Disposition"] = "attachment; filename=predicted_results.csv"
            logger.info(f"Prediction CSV prepared for download for file {file.filename}.")
            return response

        # Convert datetime columns to string before JSON serialization
        data_for_json = data.copy()
        for col in data_for_json.select_dtypes(include=['datetime64[ns]']).columns:
            data_for_json[col] = data_for_json[col].astype(str)

        
        # COMPREHENSIVE NaN and infinite values handling
        def clean_json_serializable(value):
            """Convert non-serializable values to JSON-compatible ones"""
            if pd.isna(value) or value is None:
                return None
            elif isinstance(value, (float, np.floating)):
                if np.isinf(value) or np.isnan(value):
                    return None
                # Handle very large or very small floats
                if abs(value) > 1e10 or (abs(value) < 1e-10 and value != 0):
                    return float(value)  # Let JSON handle it, but this might still fail
            elif isinstance(value, (int, np.integer)):
                return int(value)
            return value

        # Apply cleaning to all columns
        for col in data_for_json.columns:
            data_for_json[col] = data_for_json[col].apply(
                lambda x: clean_json_serializable(x) if not isinstance(x, (dict, list)) else x
            )

        # Replace NaN and infinite values with None
        data_for_json = data_for_json.replace([np.inf, -np.inf], np.nan)
        data_for_json = data_for_json.where(pd.notnull(data_for_json), None)

        result = data_for_json.to_dict(orient="records")
        logger.info(f"Prediction JSON prepared successfully for file {file.filename}.")
        return JSONResponse(content={"predictions": result})

    except HTTPException as he:
        # Already handled exceptions
        raise he
    except Exception as e:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(status_code=500, detail=str(e))
