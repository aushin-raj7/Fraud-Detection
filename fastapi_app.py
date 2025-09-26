from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import pickle
import io
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Medical Report Integrity Checker API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved artifacts once at startup
with open(r"Best_Model/preproc.pkl", "rb") as f:
    preproc = pickle.load(f)

with open(r"Best_Model/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"Best_Model/required_columns.pkl", "rb") as f:
    required_columns = pickle.load(f)


@app.post("/predict")
async def predict(file: UploadFile = File(...), return_csv: bool = False):
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
            raise HTTPException(status_code=400, detail="Invalid file type. Use CSV or Excel.")

        # Ensure all required columns exist
        for col in required_columns:
            if col not in data.columns:
                data[col] = 0
        df = data[required_columns]

        # Preprocess data
        X_proc = preproc.transform(df)

        # Predict
        predictions = model.predict(X_proc)
        prediction_prob = model.predict_proba(X_proc)[:, 1]

        # Append results
        data['Predicted_Target_Label'] = predictions
        data['Probability_Altered'] = np.round(prediction_prob, 4) 

        # Return CSV if requested
        if return_csv:
            stream = io.StringIO()
            data.to_csv(stream, index=False)
            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv"
            )
            response.headers["Content-Disposition"] = "attachment; filename=predicted_results.csv"
            return response

        # Convert datetime columns to string before JSON serialization
        data_for_json = data.copy()
        for col in data_for_json.select_dtypes(include=['datetime64[ns]']).columns:
            data_for_json[col] = data_for_json[col].astype(str)

        # Replace NaN and infinite values with None
        # data_for_json = data_for_json.replace([np.inf, -np.inf], np.nan)
        data_for_json = data_for_json.where(pd.notnull(data_for_json), None)

        # Return JSON
        result = data_for_json.to_dict(orient="records")
        return JSONResponse(content={"predictions": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
