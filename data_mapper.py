import json
import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, Request # <-- ADD 'Request' HERE
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig

# NEW: Import your mapper function!
from data_mapper import map_nist_data_to_stl

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Configure CORS so your GitHub Pages site can talk to this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client (Ensure your GEMINI_API_KEY is set in your environment variables)
# We will pass the API key from the frontend for this demo, though normally it stays hidden here.
class ChatRequest(BaseModel):
    message: str
    api_key: str

# ==========================================
# AI TOOLS (Function Calling)
# ==========================================

def search_sysml(component_name: str) -> str:
    """
    Searches the SysML model for a given component and returns its HDF5 data location.
    Args:
        component_name: The name of the physical component (e.g., 'primary_beam').
    """
    print(f"\n[TOOL CALLED] Searching SysML for: {component_name}")
    try:
        # Point this to your updated JSON map
        with open("real_sysml_map.json", "r") as f:
            sysml_data = json.load(f)
            
        components = sysml_data.get("components", {})
        if component_name in components:
            file_name = components[component_name].get("file_name")
            dataset_path = components[component_name].get("dataset_path")
            
            print(f"--> Found location: File={file_name}, Path={dataset_path}")
            # We now return BOTH the file name and the path to the AI
            return f"File: {file_name}, Dataset Path: {dataset_path}"
        return "Component not found in SysML model."
    except Exception as e:
        return f"Error reading SysML: {str(e)}"

def extract_hdf5_data(file_name: str, dataset_path: str, operation: str = "max") -> str:
    """
    Extracts array data from a specific HDF5 file and internal path.
    Args:
        file_name: The actual .h5 file on your hard drive (e.g., 'run_04_stress_results.h5').
        dataset_path: The exact internal path (e.g., '/Results/Nodes/StressTensor').
        operation: The math operation to perform ('max', 'min', or 'average').
    """
    print(f"\n[TOOL CALLED] Extracting from {file_name} at: {dataset_path} with operation: {operation}")
    try:
        # Open the real file dynamically
        with h5py.File(file_name, "r") as f:
            if dataset_path not in f:
                return f"Path {dataset_path} not found in {file_name}."
            
            # Read the real data array
            data_array = f[dataset_path][:]
            
            if operation == "max":
                result = np.max(data_array)
            elif operation == "min":
                result = np.min(data_array)
            elif operation == "average":
                result = np.mean(data_array)
            else:
                return "Unknown operation."
                
            print(f"--> Result: {result}")
            return f"The {operation} value at {dataset_path} in {file_name} is {result}"
    except Exception as e:
        return f"Error reading HDF5: {str(e)}"

# ==========================================
# API ROUTES
# ==========================================

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Initialize client with the key passed from the frontend
        client = genai.Client(api_key=request.api_key)
        
        # We pass the actual Python functions to the tools list!
        # Gemini 2.0 automatically reads the docstrings to figure out how to use them.
        config = GenerateContentConfig(
            tools=[search_sysml, extract_hdf5_data],
            temperature=0.2,
        )
        
        print(f"\n[USER QUERY] {request.message}")
        
        # Start a chat session and send the message with automatic tool calling enabled
        chat = client.chats.create(model="gemini-2.5-flash", config=config)
        response = chat.send_message(request.message)
        
        return {"response": response.text}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# NEW: HEATMAP ROUTE
# ==========================================
@app.post("/get_heatmap")
async def get_heatmap_endpoint(request: Request):
    try:
        data = await request.json()
        stl_vertices = data.get("vertices")
        
        if not stl_vertices:
            raise HTTPException(status_code=400, detail="No vertices provided")
            
        print(f"Received request to map {len(stl_vertices)//3} vertices...")
        
        # Call the function from data_mapper.py!
        # IMPORTANT: Change "demo_data.h5" to your actual NIST file name once you have it.
        # Change 'temperature' to the actual dataset path inside the NIST file.
        mapped_array = map_nist_data_to_stl(
            stl_vertices, 
            "demo_data.h5", 
            "temperature" 
        )
        
        return {"heatmap_array": mapped_array}
        
    except Exception as e:
        print(f"Heatmap Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))