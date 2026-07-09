import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig

# 1. Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DIRECT CSV HEATMAP ROUTE
# ==========================================
@app.post("/get_heatmap")
async def get_heatmap_endpoint(request: Request):
    try:
        # Load the CSV directly
        # Ensure 'your_data.csv' is in the same directory as main.py
        df = pd.read_csv(
            'your_data.csv', 
            skiprows=1, 
            header=None,
            sep=None,
            encoding='latin1',
            engine='python', 
            on_bad_lines='skip'
        )
        
        # Extract values (Assuming Column 0 is Position, Column 1 is Force)
        # Note: You may need to normalize these values to be between 0 and 1 for the heatmap
        raw_values = df.iloc[:, 1].values
        normalized_values = (raw_values - np.min(raw_values)) / (np.max(raw_values) - np.min(raw_values))
        
        # Return the data as a list
        return {"heatmap_array": normalized_values.tolist()}
        
    except Exception as e:
        print(f"Heatmap Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ... (rest of your existing search_sysml/extract_hdf5_data/chat logic)