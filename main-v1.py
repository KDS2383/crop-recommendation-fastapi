# --- START OF FILE main.py ---

import fastapi
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import warnings
import time
import os
from typing import List, Optional, Dict, Any
import threading
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import gdown # For model download

# Import Pydantic BaseModel
from pydantic import BaseModel, Field

# Import MongoDB related components
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
MODEL_DIR = 'models'
DATA_DIR = 'data'

# --- ML Model Paths ---
MODEL_LOAD_NAME = os.path.join(MODEL_DIR, 'crop_recommender_model_reduced.joblib')
SCALER_LOAD_NAME = os.path.join(MODEL_DIR, 'crop_data_scaler_reduced.joblib')
ENCODER_LOAD_NAME = os.path.join(MODEL_DIR, 'crop_label_encoder_reduced.joblib')

# --- Model Download URL or ID (from Environment Variable) ---
# SET THIS IN RENDER. Use the standard Google Drive sharing link OR just the file ID.
MODEL_DOWNLOAD_URL_OR_ID = os.getenv("MODEL_DOWNLOAD_URL_OR_ID")

# --- Local CSV File Path (USING THE NEW STRUCTURE) ---
CROP_CONDITIONS_FILE = os.path.join(DATA_DIR, 'crop_ideal_conditions_new.csv') # <-- POINTING TO NEW CSV

# --- MongoDB Configuration ---
ATLAS_USERNAME = os.getenv("ATLAS_USERNAME", "precisionfarming")
ATLAS_PASSWORD = os.getenv("ATLAS_PASSWORD", "finalyear%40909") # Ensure password is URL encoded if needed
ATLAS_CLUSTER_URL = os.getenv("ATLAS_CLUSTER_URL", "cluster0.ogil0jz.mongodb.net")
DB_NAME_CROP_INFO = os.getenv("MONGO_DB_CROP_INFO", "Crop_Information")
DB_NAME_CROP_IMG = os.getenv("MONGO_DB_CROP_IMG", "Crop_Img")
DB_NAME_DISEASE_INFO = os.getenv("MONGO_DB_DISEASE_INFO", "Disease_Info")
CROP_DETAILS_COLLECTION = os.getenv("MONGO_CROP_DETAILS_COLLECTION", "crop_details")
CROP_IMAGES_COLLECTION = os.getenv("MONGO_CROP_IMAGES_COLLECTION", "crop_images")
DISEASE_COLLECTION = os.getenv("MONGO_DISEASE_COLLECTION", "diseases_info_std_3") # Use the standardized collection

ATLAS_CONNECTION_STRING_BASE = f"mongodb+srv://{ATLAS_USERNAME}:{ATLAS_PASSWORD}@{ATLAS_CLUSTER_URL}/?retryWrites=true&w=majority"

# --- Pydantic Models (Keep UserInput simple for now) ---
class UserInput(BaseModel):
    latitude: float; longitude: float; elevation: float; soil_ph: float; soil_nitrogen: int
    soil_phosphorus: int; soil_potassium: int; soil_moisture: int; soil_cec: int
    avg_temperature: float; min_temperature: float; avg_humidity: float; min_humidity: float
    avg_wind_speed: float; total_rainfall: float; historical_crops: List[str] = []
class GrowingInfo(BaseModel): growing_season: Optional[str]=None; water_needs: Optional[str]=None; soil_preference: Optional[str]=None; harvest_time: Optional[str]=None
class ExpectedYield(BaseModel): in_tons_per_acre: Optional[float]=None
class DiseaseDetail(BaseModel): name: str; symptoms: Optional[str]=None; treatment: Optional[str]=None
class CropInfoDetail(BaseModel):
    crop_name: str; image_url: Optional[str]=None; crop_info: Optional[str]=None
    growing_info: Optional[GrowingInfo]=None; expected_yield: Optional[ExpectedYield]=None
    predicted_diseases: Optional[List[str]]=Field(default_factory=list)
    diseases: Optional[List[DiseaseDetail]]=Field(default_factory=list)
class RecommendationResponse(BaseModel): recommendations: List[CropInfoDetail]; processing_time_seconds: float


app = FastAPI()

# --- CORS Middleware Configuration ---
origins = ["*"]
app.add_middleware( CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )

# --- Feature Columns for ML Model (Unchanged) ---
MODEL_FEATURE_COLS = [ 'elevation', 'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'soil_moisture', 'soil_cec', 'avg_temperature', 'min_temperature', 'avg_humidity', 'min_humidity', 'avg_wind_speed', 'total_rainfall' ]

# --- Recommendation Logic Parameters ---
TOP_N_FROM_MODEL = 40 # Increased to give suitability ranking more options

# --- PARAMETER WEIGHTS V2 (FOR NEW SUITABILITY FUNCTION - NEEDS TUNING!) ---
# These weights amplify the penalty calculated by the suitability function.
# Higher value means the factor is more important in the overall score.
PARAMETER_WEIGHTS_V2 = {
    # Critical Factors - High Weights
    'Temperature_Score': 25.0,       # Penalty for deviation from optimal temp range
    'Rainfall_Score': 30.0,          # Penalty for deviation from optimal rain range (esp. deficit)
    'pH_Score': 20.0,                # Penalty for deviation from optimal pH range
    'Drainage_Score': 25.0,          # Penalty for drainage mismatch (needs loc_drainage input)
    'Absolute_Temp_Cutoff': 1.0,     # Multiplier for the massive cutoff penalty
    'Absolute_Rainfall_Cutoff': 1.0, # Multiplier for the massive cutoff penalty (Added for consistency)
    'Absolute_pH_Cutoff': 1.0,       # Multiplier for the massive cutoff penalty

    # Important Factors - Moderate Weights
    'Humidity_Score': 5.0,
    'Moisture_Score': 10.0,          # Penalty for deviation from optimal moisture (if implemented)
    'Texture_Score': 10.0,           # Penalty for texture mismatch (needs loc_texture input)
    'Nitrogen_Score': 5.0,           # Penalty for deviation from target N
    'Phosphorus_Score': 5.0,         # Penalty for deviation from target P
    'Potassium_Score': 5.0,          # Penalty for deviation from target K
    'Elevation_Score': 3.0,          # Penalty for elevation mismatch

    # Less Critical/Optional - Lower Weights
    'CEC_Score': 2.0,                # Penalty for deviation from target CEC (if implemented)
    'Wind_Score': 1.0,               # Penalty for exceeding max wind (if implemented)
    'Frost_Score': 15.0,             # Multiplier for massive cutoff penalty if frost sensitive
    'Salinity_Score': 5.0,           # Penalty for salinity mismatch (if implemented)
}

# --- Disease Prediction Parameters (Unchanged) ---
DISEASE_PARAM_WEIGHTS = { 'Temperature': 2.0, 'Humidity': 2.5, 'Rainfall': 0.75, 'Soil_Moisture': 1.5, 'Soil_pH': 0.5, }
ENV_RANGES = { 'Temperature': 30, 'Humidity': 50, 'Rainfall': 50, 'Soil_Moisture': 60, 'Soil_pH': 3, }

# --- Crop Generalization Mapping (Unchanged) ---
GENERAL_TO_SPECIFIC_MAP = {
    "Millet": [ "Pearl Millet", "Finger Millet", "Foxtail Millet", "Kodo Millet", "Proso Millet", "Barnyard Millet", "Italian Millet", "Little Millet", "Browntop Millet", "Guinea Millet", "Japanese Millet", "Sawa Millet", "Cheena", "Kangni", "Kutki", "Samak Rice", "Ragi" ],
    "Bean": [ "Common Bean", "Kidney Bean", "Adzuki Bean", "Fava Bean", "Lima Bean", "Mung Bean", "Black Gram (Vigna mungo)", "Black Gram", "Rice Bean", "Winged Bean", "Yardlong Bean", "Hyacinth Bean (Lablab)", "Hyacinth Bean", "Velvet Bean (Mucuna)", "Velvet Bean", "Jack Bean", "Sword Bean", "Tepary Bean", "Marama Bean", "Yam Bean", "Field Bean", "Dolichos Bean", "Cluster Bean", "Guar", "Rajma", "Lobia", "Moong", "Urad" ],
    "Squash": [ "Pumpkin", "Zucchini", "Winter Squash", "Summer Squash", "Acorn Squash", "Butternut Squash", "Spaghetti Squash", "Delicata Squash", "Kabocha Squash", "Hubbard Squash", "Turban Squash", "Pattypan Squash", "Crookneck Squash", "Buttercup Squash", "Ash Gourd", "Bottle Gourd", "Ridge Gourd", "Snake Gourd", "Luffa Gourd", "Bitter Gourd", "Pointed Gourd", "Spiny Gourd", "Wax Gourd", "Opo Squash", "Gourd", "Tinda", "Parwal", "Karela", "Calabaza" ],
    "Pepper": [ "Bell Pepper", "Chili Pepper", "Sweet Pepper", "Hot Pepper", "Chilli" ],
    "Citrus": [ "Orange", "Lime", "Lemon", "Grapefruit", "Pomelo", "Mandarin" ],
    "Palm": [ "Oil Palm", "Coconut Palm", "Date Palm", "Areca Nut", "Areca Palm", "Arecanut" ],
    "Maize": [ "Corn", "Maize (Corn - use this one)", "Sweetcorn" ]
}
SPECIFIC_TO_GENERAL_MAP = {}
for general, specifics in GENERAL_TO_SPECIFIC_MAP.items():
    for specific in specifics: SPECIFIC_TO_GENERAL_MAP[specific] = general
print(f"DEBUG: Created SPECIFIC_TO_GENERAL_MAP with {len(SPECIFIC_TO_GENERAL_MAP)} entries.")

# --- Global Variables ---
scaler = None; label_encoder = None; crop_model = None
mongo_client = None
crop_details_collection = None; crop_images_collection = None
disease_collection = None
crop_image_map = None # Atlas
df_crop_suitability = None # Local CSV (NEW STRUCTURE)
df_disease = None # Atlas (standardized)
disease_crop_map = {} # From Atlas data
model_load_lock = threading.Lock()
model_downloaded_this_session = False

# --- Helper Function for Parsing Ranges (Used for Disease Info) ---
def parse_range(range_str):
    if pd.isna(range_str) or not isinstance(range_str, str): return None, None
    range_str = range_str.strip(); match = re.match(r'^([\d.]+)\s*[-–—]\s*([\d.]+)$', range_str)
    if match:
        try: return float(match.group(1)), float(match.group(2))
        except ValueError: return None, None
    try: return float(range_str), float(range_str)
    except ValueError: return None, None

# --- Startup Function (Loading NEW CSV) ---
@app.on_event("startup")
async def startup_event():
    global scaler, label_encoder, mongo_client, crop_details_collection, crop_images_collection
    global disease_collection, crop_image_map
    global df_crop_suitability, df_disease, disease_crop_map # Removed param_ranges, SUITABILITY_PARAM_COLS

    print("--- Starting FastAPI Application ---")
    print("Loading ML Scaler and Encoder artifacts...")
    try:
        if not os.path.exists(SCALER_LOAD_NAME): raise FileNotFoundError(f"Scaler file not found at {SCALER_LOAD_NAME}")
        if not os.path.exists(ENCODER_LOAD_NAME): raise FileNotFoundError(f"Encoder file not found at {ENCODER_LOAD_NAME}")
        scaler = joblib.load(SCALER_LOAD_NAME)
        label_encoder = joblib.load(ENCODER_LOAD_NAME)
        print(f"OK: Loaded scaler and label encoder. Num classes: {len(label_encoder.classes_)}")
    except Exception as e: print(f"ERROR loading ML scaler/encoder: {e}"); scaler = label_encoder = None

    # --- Load NEW Crop Suitability Conditions from LOCAL CSV ---
    print(f"Loading NEW crop suitability conditions from LOCAL CSV: {CROP_CONDITIONS_FILE}...")
    try:
        if not os.path.exists(CROP_CONDITIONS_FILE):
            raise FileNotFoundError(f"NEW Crop suitability CSV file not found at {CROP_CONDITIONS_FILE}")

        # Load the entire CSV, handle dtypes later if needed
        df_crop_suitability = pd.read_csv(CROP_CONDITIONS_FILE, index_col='Crop')

        # Basic check for expected columns (add more checks as needed)
        required_cols = ['Min_Optimal_Temp', 'Max_Optimal_Temp', 'Min_Optimal_Rainfall', 'Max_Optimal_Rainfall', 'Min_Optimal_pH', 'Max_Optimal_pH']
        missing_cols = [col for col in required_cols if col not in df_crop_suitability.columns]
        if missing_cols:
            print(f"Warning: NEW CSV is missing expected columns: {missing_cols}")
            # Decide how to handle: raise error, or proceed with limitations

        # Convert numeric columns, coercing errors to NaN
        # List all expected numeric columns from the new structure
        numeric_cols_new = [
            'Min_Optimal_Temp', 'Max_Optimal_Temp', 'Min_Absolute_Temp', 'Max_Absolute_Temp',
            'Min_Optimal_Rainfall', 'Max_Optimal_Rainfall', 'Min_Survival_Rainfall',
            'Min_Optimal_pH', 'Max_Optimal_pH', 'Min_Tolerated_pH', 'Max_Tolerated_pH',
            'Min_Optimal_Humidity', 'Max_Optimal_Humidity',
            'Nitrogen_Requirement', 'Phosphorus_Requirement', 'Potassium_Requirement',
            'Min_Elevation', 'Max_Elevation',
            'Min_Optimal_Moisture', 'Max_Optimal_Moisture',
            'Max_Tolerable_Wind'
            # Add others if numeric (like CEC if you keep it numeric)
        ]
        for col in numeric_cols_new:
            if col in df_crop_suitability.columns:
                df_crop_suitability[col] = pd.to_numeric(df_crop_suitability[col], errors='coerce')

        print(f"OK: Loaded NEW suitability data from CSV. Shape: {df_crop_suitability.shape}.")
        print(f"DEBUG: Columns loaded: {df_crop_suitability.columns.tolist()}")

    except FileNotFoundError as e:
        print(f"ERROR: {e}. NEW Crop suitability data unavailable.")
        df_crop_suitability = None
    except Exception as e:
        print(f"ERROR loading NEW suitability data from CSV: {e}")
        df_crop_suitability = None

    # --- Connect to MongoDB Atlas (Unchanged) ---
    print(f"Connecting to MongoDB Atlas: {ATLAS_CLUSTER_URL}...")
    try:
        mongo_client = MongoClient(ATLAS_CONNECTION_STRING_BASE, serverSelectionTimeoutMS=7000)
        mongo_client.admin.command('ping') # Verify connection
        print("OK: Connected to MongoDB Atlas")
        db_crop_info = mongo_client[DB_NAME_CROP_INFO]; db_crop_img = mongo_client[DB_NAME_CROP_IMG]; db_disease_info = mongo_client[DB_NAME_DISEASE_INFO]
        crop_details_collection = db_crop_info[CROP_DETAILS_COLLECTION]; crop_images_collection = db_crop_img[CROP_IMAGES_COLLECTION]; disease_collection = db_disease_info[DISEASE_COLLECTION]
        print("OK: Got handles for required MongoDB collections.")
    except Exception as e:
        print(f"FATAL ERROR connecting to MongoDB Atlas: {e}")
        mongo_client = None; crop_details_collection = None; crop_images_collection = None; disease_collection = None

    # --- Load Crop Images & Disease Data from MongoDB (Unchanged) ---
    if mongo_client:
        print(f"Loading crop images from DB '{DB_NAME_CROP_IMG}', Collection '{CROP_IMAGES_COLLECTION}'...")
        crop_image_map = {}
        try:
            cursor = crop_images_collection.find({}, {"crop_name": 1, "image_url": 1, "_id": 0})
            for doc in cursor:
                if doc.get("crop_name") and doc.get("image_url"): crop_image_map[doc["crop_name"]] = doc["image_url"]
            print(f"OK: Loaded {len(crop_image_map)} crop image URLs from MongoDB.")
        except Exception as e: print(f"ERROR loading crop images from MongoDB: {e}"); crop_image_map = {}

        print(f"Loading ALL disease data from DB '{DB_NAME_DISEASE_INFO}', Collection '{DISEASE_COLLECTION}'...")
        try:
            disease_docs = list(disease_collection.find({}, {"_id": 0}))
            if not disease_docs: raise ValueError("No documents found")
            df_disease = pd.DataFrame.from_records(disease_docs)
            # ... (rest of disease data processing unchanged) ...
            for col in ['name', 'display_name', 'crop_hosts', 'symptoms', 'treatment', 'soil_pH_Preference']:
                if col not in df_disease.columns: df_disease[col] = None
            df_disease['name'].fillna('unknown_disease', inplace=True); df_disease['display_name'].fillna(df_disease['name'], inplace=True)
            df_disease['crop_hosts'].fillna('', inplace=True); df_disease['symptoms'].fillna('Not Available', inplace=True)
            df_disease['treatment'].fillna('Not Available', inplace=True)
            disease_numeric_cols = [ 'min_Optimal_Temp', 'max_Optimal_Temp', 'min_Optimal_Humidity', 'max_Optimal_Humidity', 'min_Rainfall_mm', 'max_Rainfall_mm', 'min_Soil_Moisture', 'max_Soil_Moisture' ]
            for col in disease_numeric_cols:
                if col in df_disease.columns: df_disease[col] = pd.to_numeric(df_disease[col], errors='coerce')
                else: df_disease[col] = np.nan
            print("Processing disease hosts from Atlas data using generalization map...")
            disease_crop_map = {} ; unique_disease_hosts_found = set()
            for index, row in df_disease.iterrows():
                if pd.notna(row['crop_hosts']) and isinstance(row['crop_hosts'], str):
                    hosts = [h.strip() for h in row['crop_hosts'].split(';') if h.strip()]
                    unique_disease_hosts_found.update(hosts)
                    for specific_host in hosts:
                        if specific_host not in disease_crop_map: disease_crop_map[specific_host] = []
                        if index not in disease_crop_map[specific_host]: disease_crop_map[specific_host].append(index)
                        general_category = SPECIFIC_TO_GENERAL_MAP.get(specific_host)
                        if general_category and general_category != specific_host:
                            if general_category not in disease_crop_map: disease_crop_map[general_category] = []
                            if index not in disease_crop_map[general_category]: disease_crop_map[general_category].append(index)
            print(f"OK: Disease data loaded from Atlas. Shape: {df_disease.shape}. Crop map size: {len(disease_crop_map)}")
        except Exception as e: print(f"ERROR loading disease data from MongoDB: {e}"); df_disease = None; disease_crop_map = {}
    else:
         crop_image_map = {}; df_disease = None; disease_crop_map = {}

    print("--- Application Startup Complete ---")

# --- Shutdown Event (Unchanged) ---
@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client: mongo_client.close(); print("MongoDB connection closed.")

# --- get_model (Using gdown - Unchanged) ---
def get_model():
    global crop_model, model_downloaded_this_session
    if crop_model is not None: return crop_model
    with model_load_lock:
        if crop_model is None:
            if not os.path.exists(MODEL_LOAD_NAME):
                print(f"Model file not found locally at {MODEL_LOAD_NAME}.")
                if not MODEL_DOWNLOAD_URL_OR_ID: raise RuntimeError("Model file missing and download URL/ID not configured.")
                if model_downloaded_this_session: raise RuntimeError("Model download previously failed, cannot proceed.")
                print(f"Attempting to download model using gdown from source: '{MODEL_DOWNLOAD_URL_OR_ID[:50]}...'")
                download_start_time = time.time()
                try:
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    print(f"Downloading model to: {MODEL_LOAD_NAME}")
                    output_path = gdown.download(url=MODEL_DOWNLOAD_URL_OR_ID, output=MODEL_LOAD_NAME, quiet=False, fuzzy=True)
                    if output_path is None or not os.path.exists(MODEL_LOAD_NAME): raise RuntimeError(f"gdown download failed.")
                    download_end_time = time.time(); print(f"\nOK: Model downloaded successfully using gdown in {download_end_time - download_start_time:.2f}s.")
                    model_downloaded_this_session = True
                except Exception as e:
                    print(f"\nERROR: Failed to download model using gdown: {e}")
                    if os.path.exists(MODEL_LOAD_NAME):
                        try: os.remove(MODEL_LOAD_NAME); print("Cleaned up potentially incomplete download file.")
                        except OSError as rm_err: print(f"Warning: Could not remove incomplete file {MODEL_LOAD_NAME}: {rm_err}")
                    model_downloaded_this_session = True # Mark attempt made
                    raise RuntimeError(f"Failed to download ML model using gdown. Error: {e}")
            if os.path.exists(MODEL_LOAD_NAME):
                print(f"Loading Crop Recommendation model from local file: {MODEL_LOAD_NAME}")
                load_start_time = time.time()
                try:
                    loaded_model = joblib.load(MODEL_LOAD_NAME)
                    load_end_time = time.time(); print(f"Model loaded successfully in {load_end_time - load_start_time:.2f}s.")
                    crop_model = loaded_model
                except Exception as e: crop_model = None; raise RuntimeError(f"Failed to load ML model from file. Error: {e}")
            else: crop_model = None; raise RuntimeError("Model file could not be loaded or downloaded.")
    return crop_model

# === NEW SUITABILITY CALCULATION FUNCTION (V2) ===
def calculate_suitability_v2(loc_conditions: Dict, crop_row_data: pd.Series, weights: Dict) -> float:
    """
    Calculates dissimilarity score using ranges from the new CSV structure.
    Lower score means more suitable. Incorporates hard cutoffs and penalties.
    NEEDS CAREFUL TUNING of weights and internal penalty logic.
    """
    total_dissimilarity = 0.0
    params_evaluated_count = 0
    CUTOFF_PENALTY = 1e9 # Very large penalty for being outside absolute limits

    # --- 1. Hard Cutoffs ---
    # Absolute Temperature
    loc_min_temp = loc_conditions.get('min_temperature')
    loc_avg_temp = loc_conditions.get('avg_temperature') # Use Avg as a proxy for max experienced day temp
    crop_min_abs_temp = crop_row_data.get('Min_Absolute_Temp')
    crop_max_abs_temp = crop_row_data.get('Max_Absolute_Temp')

    if pd.notna(crop_min_abs_temp) and pd.notna(loc_min_temp) and loc_min_temp < crop_min_abs_temp:
        # print(f"DEBUG Cutoff ({crop_row_data.name}): Loc Min Temp {loc_min_temp} < Crop Abs Min {crop_min_abs_temp}")
        total_dissimilarity += CUTOFF_PENALTY * weights.get('Absolute_Temp_Cutoff', 1.0)
    if pd.notna(crop_max_abs_temp) and pd.notna(loc_avg_temp) and loc_avg_temp > crop_max_abs_temp:
        # print(f"DEBUG Cutoff ({crop_row_data.name}): Loc Avg Temp {loc_avg_temp} > Crop Abs Max {crop_max_abs_temp}")
        total_dissimilarity += CUTOFF_PENALTY * weights.get('Absolute_Temp_Cutoff', 1.0)

    # Frost Sensitivity
    crop_frost_sensitive = crop_row_data.get('Frost_Sensitive') # Check NaN or specific value like 'Yes'/'No'
    if pd.notna(crop_frost_sensitive) and isinstance(crop_frost_sensitive, str) and crop_frost_sensitive.strip().lower() == 'yes':
        if pd.notna(loc_min_temp) and loc_min_temp <= 2.0: # Threshold near freezing (adjust if needed)
             # print(f"DEBUG Cutoff ({crop_row_data.name}): Frost sensitive and Loc Min Temp {loc_min_temp} <= 2.0")
             total_dissimilarity += CUTOFF_PENALTY * weights.get('Frost_Score', 1.0) # Apply high penalty

    # Minimum Survival Rainfall
    loc_total_rainfall = loc_conditions.get('total_rainfall')
    crop_min_survival_rain = crop_row_data.get('Min_Survival_Rainfall')
    if pd.notna(crop_min_survival_rain) and pd.notna(loc_total_rainfall) and loc_total_rainfall < crop_min_survival_rain:
        # print(f"DEBUG Cutoff ({crop_row_data.name}): Loc Rainfall {loc_total_rainfall} < Crop Min Survival {crop_min_survival_rain}")
        # Use a specific weight for this cutoff
        total_dissimilarity += CUTOFF_PENALTY * weights.get('Absolute_Rainfall_Cutoff', 1.0) # Apply high penalty

    # Tolerated pH Range
    loc_ph = loc_conditions.get('soil_ph')
    crop_min_tol_ph = crop_row_data.get('Min_Tolerated_pH')
    crop_max_tol_ph = crop_row_data.get('Max_Tolerated_pH')
    if pd.notna(loc_ph):
        if pd.notna(crop_min_tol_ph) and loc_ph < crop_min_tol_ph:
            # print(f"DEBUG Cutoff ({crop_row_data.name}): Loc pH {loc_ph} < Crop Min Tol pH {crop_min_tol_ph}")
            total_dissimilarity += CUTOFF_PENALTY * weights.get('Absolute_pH_Cutoff', 1.0)
        if pd.notna(crop_max_tol_ph) and loc_ph > crop_max_tol_ph:
            # print(f"DEBUG Cutoff ({crop_row_data.name}): Loc pH {loc_ph} > Crop Max Tol pH {crop_max_tol_ph}")
            total_dissimilarity += CUTOFF_PENALTY * weights.get('Absolute_pH_Cutoff', 1.0)

    # If any cutoff was hit, return immediately - prevents calculating scores for unsuitable crops
    if total_dissimilarity >= CUTOFF_PENALTY:
        return total_dissimilarity # Return the large penalty score

    # --- 2. Scoring Deviations from Optimal Ranges ---
    component_scores = {} # To store individual parameter scores for weighting

    # Temperature (Optimal Range) - Use loc_avg_temp
    min_opt_temp = crop_row_data.get('Min_Optimal_Temp')
    max_opt_temp = crop_row_data.get('Max_Optimal_Temp')
    if pd.notna(loc_avg_temp) and pd.notna(min_opt_temp) and pd.notna(max_opt_temp):
        score = 0.0
        optimal_range_width = max(max_opt_temp - min_opt_temp, 1e-6) # Min width 1e-6
        if loc_avg_temp < min_opt_temp:
            score = ((min_opt_temp - loc_avg_temp) / optimal_range_width) ** 2
        elif loc_avg_temp > max_opt_temp:
            score = ((loc_avg_temp - max_opt_temp) / optimal_range_width) ** 2
        component_scores['Temperature_Score'] = score
        params_evaluated_count += 1

    # Rainfall (Optimal Range) - Use loc_total_rainfall
    min_opt_rain = crop_row_data.get('Min_Optimal_Rainfall')
    max_opt_rain = crop_row_data.get('Max_Optimal_Rainfall')
    if pd.notna(loc_total_rainfall) and pd.notna(min_opt_rain) and pd.notna(max_opt_rain):
        score = 0.0
        optimal_range_width = max(max_opt_rain - min_opt_rain, 1.0) # Min width 1mm
        if loc_total_rainfall < min_opt_rain:
             # Penalize more heavily for being below minimum rainfall
             penalty_factor = 2.0 # TUNABLE: Increase penalty for drought
             score = (penalty_factor * (min_opt_rain - loc_total_rainfall) / optimal_range_width) ** 2
        elif loc_total_rainfall > max_opt_rain:
             # Penalize less heavily for too much rain (unless drainage is poor - handle below)
             penalty_factor = 0.8 # TUNABLE
             score = (penalty_factor * (loc_total_rainfall - max_opt_rain) / optimal_range_width) ** 2
        component_scores['Rainfall_Score'] = score
        params_evaluated_count += 1

    # pH (Optimal Range) - Use loc_ph
    min_opt_ph = crop_row_data.get('Min_Optimal_pH')
    max_opt_ph = crop_row_data.get('Max_Optimal_pH')
    if pd.notna(loc_ph) and pd.notna(min_opt_ph) and pd.notna(max_opt_ph):
        score = 0.0
        optimal_range_width = max(max_opt_ph - min_opt_ph, 0.1) # Min width 0.1 pH unit
        if loc_ph < min_opt_ph:
            score = ((min_opt_ph - loc_ph) / optimal_range_width) ** 2
        elif loc_ph > max_opt_ph:
            score = ((loc_ph - max_opt_ph) / optimal_range_width) ** 2
        component_scores['pH_Score'] = score
        params_evaluated_count += 1

    # Humidity (Optimal Range) - Use loc_avg_humidity
    loc_avg_humidity = loc_conditions.get('avg_humidity')
    min_opt_hum = crop_row_data.get('Min_Optimal_Humidity')
    max_opt_hum = crop_row_data.get('Max_Optimal_Humidity')
    if pd.notna(loc_avg_humidity) and pd.notna(min_opt_hum) and pd.notna(max_opt_hum):
        score = 0.0
        optimal_range_width = max(max_opt_hum - min_opt_hum, 1.0) # Min width 1%
        if loc_avg_humidity < min_opt_hum:
            score = ((min_opt_hum - loc_avg_humidity) / optimal_range_width) ** 2
        elif loc_avg_humidity > max_opt_hum:
            score = ((loc_avg_humidity - max_opt_hum) / optimal_range_width) ** 2
        component_scores['Humidity_Score'] = score
        params_evaluated_count += 1

    # --- 3. Scoring Target Values (N, P, K) ---
    # Using squared percentage deviation from target
    for param_name, key_suffix in [('Nitrogen', 'nitrogen'), ('Phosphorus', 'phosphorus'), ('Potassium', 'potassium')]:
        loc_val = loc_conditions.get(f'soil_{key_suffix}')
        target_val = crop_row_data.get(f'{param_name}_Requirement')
        if pd.notna(loc_val) and pd.notna(target_val) and target_val > 0:
             deviation_percent_sq = ((loc_val - target_val) / target_val) ** 2
             score = deviation_percent_sq # Score directly reflects squared % deviation
             component_scores[f'{param_name}_Score'] = score
             params_evaluated_count += 1

    # --- 4. Scoring Categorical Data (Drainage, Texture) ---
    # Placeholder: Requires loc_drainage, loc_texture in loc_conditions
    # This data is not currently in UserInput model

    crop_drainage_req = crop_row_data.get('Drainage_Requirement')
    loc_drainage = loc_conditions.get('drainage') # Example: Get from user input if available
    if pd.notna(crop_drainage_req) and crop_drainage_req != "Unknown":
        # If location drainage is unknown, add a medium penalty
        if loc_drainage is None or loc_drainage == "Unknown":
             component_scores['Drainage_Score'] = 0.5 # Penalty for uncertainty
             params_evaluated_count += 1
        else:
            # Simple matching logic (TUNABLE PENALTIES)
            drainage_map = {"Excellent": 3, "Good": 2, "Moderate": 1, "Poor": 0}
            loc_d = drainage_map.get(loc_drainage, -1)
            crop_d = drainage_map.get(crop_drainage_req, -1)
            score = 0.0
            if loc_d != -1 and crop_d != -1:
                if loc_d < crop_d: # Location drainage is worse than required (Higher penalty)
                    score = (crop_d - loc_d)**2 * 1.0 # Penalty multiplier = 1.0
                elif loc_d > crop_d: # Location drainage better than required (Lower penalty)
                    score = (loc_d - crop_d)**2 * 0.2 # Penalty multiplier = 0.2
            component_scores['Drainage_Score'] = score
            params_evaluated_count += 1


    crop_texture_pref = crop_row_data.get('Soil_Texture_Preference')
    loc_texture = loc_conditions.get('soil_texture') # Example: Get from user input if available
    if pd.notna(crop_texture_pref) and crop_texture_pref not in ["Unknown", "Any"]:
         if loc_texture is None or loc_texture == "Unknown":
             component_scores['Texture_Score'] = 0.5 # Penalty for uncertainty
             params_evaluated_count += 1
         else:
             # Simple mismatch penalty (Can be made more nuanced)
             if crop_texture_pref.lower() not in loc_texture.lower() and loc_texture.lower() not in crop_texture_pref.lower():
                 component_scores['Texture_Score'] = 1.0 # Penalty score = 1.0
             else:
                 component_scores['Texture_Score'] = 0.0
             params_evaluated_count += 1


    # --- 5. Elevation Score (outside optimal range) ---
    loc_elevation = loc_conditions.get('elevation')
    min_elev = crop_row_data.get('Min_Elevation')
    max_elev = crop_row_data.get('Max_Elevation')
    if pd.notna(loc_elevation) and (pd.notna(min_elev) or pd.notna(max_elev)):
        score = 0.0
        # Define a reference elevation range (e.g., 1000m) for normalization, TUNE THIS!
        elevation_normalization_range = 1000.0
        if pd.notna(min_elev) and loc_elevation < min_elev:
            # Ensure normalization range is positive
            norm_range = max(elevation_normalization_range, 1.0)
            score = ((min_elev - loc_elevation) / norm_range) ** 2
        elif pd.notna(max_elev) and loc_elevation > max_elev:
            norm_range = max(elevation_normalization_range, 1.0)
            score = ((loc_elevation - max_elev) / norm_range) ** 2
        component_scores['Elevation_Score'] = score
        params_evaluated_count += 1


    # --- 6. Combine Scores with Weights ---
    final_score = 0.0
    # Add base dissimilarity from optimal range deviations
    for score_name, score_value in component_scores.items():
        weight = weights.get(score_name, 1.0) # Get weight for this component
        final_score += score_value * weight

    # Add the initial hard cutoff penalties (which were 0 if no cutoff was hit)
    final_score += total_dissimilarity

    # Return large score if nothing could be evaluated
    return final_score if params_evaluated_count > 0 else 1e12


# --- fetch_crop_info (Unchanged) ---
def fetch_crop_info(crop_names: List[str]) -> Dict[str, Dict[str, Any]]:
    if crop_details_collection is None or not crop_names: return {}
    found_crops_info = {}
    try:
        print(f"Querying MongoDB '{CROP_DETAILS_COLLECTION}' for info on: {crop_names}")
        cursor = crop_details_collection.find( {"crop_name": {"$in": crop_names}}, {"_id": 0} )
        count = 0; [found_crops_info.update({doc["crop_name"]: doc}) or (count := count + 1) for doc in cursor if "crop_name" in doc]
        print(f"Found info for {count} out of {len(crop_names)} requested crops in MongoDB.") # Corrected log message
    except Exception as e: print(f"Error during MongoDB fetch from '{CROP_DETAILS_COLLECTION}': {e}")
    return found_crops_info

# --- calculate_disease_risk_score (Unchanged) ---
def calculate_disease_risk_score(user_conditions: Dict[str, Any], disease_data: pd.Series) -> float:
    total_risk_score = 0.0; params_calculated = 0
    def calculate_param_score(user_val, min_opt, max_opt, env_range):
        if user_val is None or pd.isna(min_opt) or pd.isna(max_opt): return None
        score = 0.0; env_range = max(env_range, 1e-6)
        if user_val < min_opt: score = (min_opt - user_val) / env_range
        elif user_val > max_opt: score = (user_val - max_opt) / env_range
        return max(0, score)
    temp_score=calculate_param_score(user_conditions.get('avg_temperature'),disease_data.get('min_Optimal_Temp'),disease_data.get('max_Optimal_Temp'),ENV_RANGES['Temperature'])
    if temp_score is not None:total_risk_score+=temp_score*DISEASE_PARAM_WEIGHTS['Temperature'];params_calculated+=1
    hum_score=calculate_param_score(user_conditions.get('avg_humidity'),disease_data.get('min_Optimal_Humidity'),disease_data.get('max_Optimal_Humidity'),ENV_RANGES['Humidity'])
    if hum_score is not None:total_risk_score+=hum_score*DISEASE_PARAM_WEIGHTS['Humidity'];params_calculated+=1
    moist_score=calculate_param_score(user_conditions.get('soil_moisture'),disease_data.get('min_Soil_Moisture'),disease_data.get('max_Soil_Moisture'),ENV_RANGES['Soil_Moisture'])
    if moist_score is not None:total_risk_score+=moist_score*DISEASE_PARAM_WEIGHTS['Soil_Moisture'];params_calculated+=1
    user_rainfall=user_conditions.get('total_rainfall');min_opt_rain=disease_data.get('min_Rainfall_mm');max_opt_rain=disease_data.get('max_Rainfall_mm')
    if user_rainfall is not None and(pd.notna(min_opt_rain)or pd.notna(max_opt_rain)):
        rain_score=0.0 # Start with 0 penalty
        if pd.notna(min_opt_rain)and user_rainfall<min_opt_rain: rain_score=1.0 # High penalty if below min
        elif pd.notna(max_opt_rain)and user_rainfall>max_opt_rain: rain_score=1.0 # High penalty if above max
        total_risk_score+=rain_score*DISEASE_PARAM_WEIGHTS['Rainfall'];params_calculated+=1
    user_ph=user_conditions.get('soil_ph');min_opt_ph,max_opt_ph=parse_range(disease_data.get('soil_pH_Preference')) # Use correct field name
    if user_ph is not None and min_opt_ph is not None and max_opt_ph is not None:
        ph_score=calculate_param_score(user_ph,min_opt_ph,max_opt_ph,ENV_RANGES['Soil_pH'])
        if ph_score is not None:total_risk_score+=ph_score*DISEASE_PARAM_WEIGHTS['Soil_pH'];params_calculated+=1
    if params_calculated==0:return 999.0
    return total_risk_score/params_calculated


# --- get_disease_predictions (Unchanged logic, uses Atlas data) ---
def get_disease_predictions(crop_name: str, user_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
    if df_disease is None or df_disease.empty or not disease_crop_map: return []
    specific_indices = set(disease_crop_map.get(crop_name, []))
    general_category = SPECIFIC_TO_GENERAL_MAP.get(crop_name)
    general_indices = set(disease_crop_map.get(general_category, [])) if general_category else set()
    all_relevant_indices = list(specific_indices.union(general_indices))
    if not all_relevant_indices and general_category and general_category in disease_crop_map:
         all_relevant_indices = list(disease_crop_map.get(general_category, []))
    if not all_relevant_indices: return []
    # print(f"DEBUG: Evaluating {len(all_relevant_indices)} potential disease indices for {crop_name}.")
    potential_diseases = []
    for index in all_relevant_indices:
        if index >= len(df_disease): continue
        disease_data = df_disease.iloc[index]; disease_display_name = disease_data.get('display_name')
        if pd.isna(disease_display_name) or disease_display_name == 'unknown_disease': continue
        try:
            risk_score = calculate_disease_risk_score(user_conditions, disease_data)
            potential_diseases.append({
                "name": disease_display_name,
                "symptoms": disease_data.get('symptoms') if pd.notna(disease_data.get('symptoms')) and disease_data.get('symptoms') != 'Not Available' else None,
                "treatment": disease_data.get('treatment') if pd.notna(disease_data.get('treatment')) and disease_data.get('treatment') != 'Not Available' else None,
                "_score": risk_score })
        except Exception as e: print(f"Warning: Error scoring disease index {index} ('{disease_display_name}'): {e}")
    if not potential_diseases: return []
    unique_best_diseases = {}
    for disease in potential_diseases:
        name = disease['name']; score = disease['_score']
        if name not in unique_best_diseases or score < unique_best_diseases[name]['_score']: unique_best_diseases[name] = disease
    sorted_unique_diseases = sorted(unique_best_diseases.values(), key=lambda x: x['_score'])
    top_3_disease_details = [{k: v for k, v in d.items() if k != '_score'} for d in sorted_unique_diseases[:3]]
    # print(f"Top predicted diseases for {crop_name}: {[d['name'] for d in top_3_disease_details]}")
    return top_3_disease_details


# --- get_recommendations_with_info (Using NEW SUITABILITY) ---
def get_recommendations_with_info(user_input_dict: Dict, top_n_model: int, scaler, encoder,
                                  df_crop_conditions: pd.DataFrame, # Expecting the NEW dataframe
                                  feature_cols_model: List[str],
                                  param_weights: Dict, # Parameter weights V2
                                  exclude_crops: List[str] = []) -> List[Dict[str, Any]]:
    """
    Generates recommendations using the ML model and re-ranks based on
    the NEW calculate_suitability_v2 function and the new CSV data structure.
    """
    model = get_model() # This will load or download the model
    if model is None: raise RuntimeError("ML Model not available")
    if scaler is None or encoder is None: raise RuntimeError("Scaler or Encoder not loaded.")
    if df_crop_conditions is None or df_crop_conditions.empty: raise RuntimeError("Crop suitability data (NEW CSV) not loaded or empty.")
    if crop_image_map is None: print("Warning: Crop image map not loaded.")
    if df_disease is None or not disease_crop_map: print("Warning: Disease data not loaded.")

    # 1. Get Top N Model Predictions (Unchanged)
    try:
        feature_values = [user_input_dict.get(col) for col in feature_cols_model]
        input_df = pd.DataFrame([feature_values], columns=feature_cols_model); input_df.fillna(0, inplace=True)
        input_scaled = scaler.transform(input_df); probabilities = model.predict_proba(input_scaled)[0]
        top_n_indices = np.argsort(probabilities)[::-1][:top_n_model]; top_n_probabilities = probabilities[top_n_indices]
        top_n_crop_names = encoder.inverse_transform(top_n_indices)
        # print(f"DEBUG: Exact crop names from encoder: {top_n_crop_names[:5]}...")
    except Exception as e: print(f"Error during ML prediction steps: {e}"); raise RuntimeError(f"ML prediction failed: {e}")
    print(f"\nTop {top_n_model} crops from ML (before re-rank): {list(zip(top_n_crop_names, top_n_probabilities))[:10]}") # Log top 10

    # 2. Re-rank using NEW Crop Suitability Score (calculate_suitability_v2)
    ranked_candidates = []
    # Prepare location conditions dict once (using keys from UserInput)
    loc_conditions_mapped = user_input_dict.copy()

    for i in range(len(top_n_crop_names)):
        crop_name = top_n_crop_names[i]
        ml_prob = top_n_probabilities[i]

        if crop_name in df_crop_conditions.index:
            ideal_conditions_row = df_crop_conditions.loc[crop_name] # Pandas Series
            try:
                suitability_score = calculate_suitability_v2(
                    loc_conditions=loc_conditions_mapped,
                    crop_row_data=ideal_conditions_row,
                    weights=param_weights # Pass the V2 weights
                )
                # Add slight penalty based on ML rank if scores are very close (optional tie-breaker)
                # suitability_score += (1 - ml_prob) * 0.01 # Example: tiny penalty
                ranked_candidates.append({'crop': crop_name, 'ml_prob': ml_prob, 'suitability_score': suitability_score})

            except Exception as e_calc:
                print(f"Warn: Suitability V2 calculation error for {crop_name}: {e_calc}. Assigning high score.")
                import traceback
                traceback.print_exc() # Print stack trace for debug
                ranked_candidates.append({'crop': crop_name, 'ml_prob': ml_prob, 'suitability_score': 1e12}) # Use very high score
        else:
            print(f"Warn: Crop '{crop_name}' predicted by model not found in NEW suitability data index.")
            ranked_candidates.append({'crop': crop_name, 'ml_prob': ml_prob, 'suitability_score': 1e12 + (1-ml_prob)})

    # --- SORTING: ONLY by suitability score (lower is better) ---
    print("Sorting candidates primarily by environmental suitability score (V2)...")
    ranked_candidates.sort(key=lambda x: x.get('suitability_score', 1e12))
    # --- End Sorting ---

    # 3. Filtering Historical Crops (Unchanged)
    if exclude_crops:
        exclude_crops_set = set(exclude_crops)
        original_count = len(ranked_candidates)
        filtered_candidates = [c for c in ranked_candidates if c['crop'] not in exclude_crops_set]
        print(f"\nExcluding historical crops: {exclude_crops_set}. Candidates reduced from {original_count} to {len(filtered_candidates)}.")
    else:
        filtered_candidates = ranked_candidates

    print("\nTop candidates AFTER V2 suitability ranking & filtering:")
    for item in filtered_candidates[:10]: # Print top 10
        print(f"  - {item['crop']}: Suitability={item.get('suitability_score', 'N/A'):.4f}, (ML Prob={item['ml_prob']:.4f})")

    # 4. Get Top 3 Crop NAMES (Unchanged)
    top_3_names = [item['crop'] for item in filtered_candidates[:3]]
    if len(top_3_names) < 3: print(f"Warning: Found only {len(top_3_names)} suitable crops.")
    print(f"\nFinal Top {len(top_3_names)} Recommended Crop Names: {top_3_names}")

    # 5. Fetch Crop Details & Image (Atlas), Predict Diseases (Atlas) - Unchanged Logic
    detailed_recommendations = []
    if top_3_names:
        fetched_info_map = fetch_crop_info(top_3_names)
        user_conditions_for_disease = {k: user_input_dict.get(k) for k in [
            'avg_temperature', 'avg_humidity', 'soil_moisture', 'total_rainfall', 'soil_ph'
            ] if k in user_input_dict}

        for name in top_3_names:
            crop_detail = fetched_info_map.get(name, {"crop_name": name, "crop_info": "Detailed info unavailable."})
            crop_detail["image_url"] = crop_image_map.get(name) if crop_image_map else None
            predicted_disease_details = get_disease_predictions(name, user_conditions_for_disease)
            crop_detail["diseases"] = predicted_disease_details
            crop_detail["predicted_diseases"] = [d['name'] for d in predicted_disease_details]
            detailed_recommendations.append(crop_detail)
    return detailed_recommendations


# === FastAPI Endpoint (Using NEW SUITABILITY) ===
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_crops_endpoint(user_input: UserInput):
    start_time = time.time(); print("\n--- Received /recommend request ---")

    # Verify prerequisites (NOW CHECKS NEW df_crop_suitability)
    missing_components = []
    if scaler is None: missing_components.append("Scaler")
    if label_encoder is None: missing_components.append("Label Encoder")
    if df_crop_suitability is None: missing_components.append("Crop Suitability Data (NEW CSV)") # Check new DF
    if crop_image_map is None: missing_components.append("Crop Image Map (DB)")
    if df_disease is None: missing_components.append("Disease Data (DB)")
    if disease_crop_map is None: missing_components.append("Disease Crop Map (DB)")
    if crop_details_collection is None: missing_components.append("MongoDB Crop Details Collection Handle")
    if disease_collection is None: missing_components.append("MongoDB Disease Collection Handle")

    # Critical check
    critical_missing = [comp for comp in missing_components if "Scaler" in comp or "Encoder" in comp or "Suitability" in comp or "Handle" in comp]
    if critical_missing: print(f"ERROR: Critical data components missing: {critical_missing}"); raise HTTPException(status_code=503, detail="Server configuration error: Core data components not ready.")
    elif missing_components: print(f"Warning: Non-critical data components missing: {missing_components}. Proceeding with limitations.")

    try:
        user_conditions_dict = user_input.dict(); print(f"User Input received: {user_conditions_dict}")

        # Call the main recommendation logic function (uses calculate_suitability_v2 internally)
        recommendations_with_details = get_recommendations_with_info(
            user_input_dict=user_conditions_dict,
            top_n_model=TOP_N_FROM_MODEL,
            scaler=scaler,
            encoder=label_encoder,
            df_crop_conditions=df_crop_suitability, # Pass the NEW dataframe
            feature_cols_model=MODEL_FEATURE_COLS, # For ML model prediction
            param_weights=PARAMETER_WEIGHTS_V2,    # Pass the NEW weights dict
            exclude_crops=user_input.historical_crops
        )

        end_time = time.time(); processing_time = end_time - start_time
        print(f"--- Recommendation processing completed in {processing_time:.3f} seconds ---")
        if not recommendations_with_details: print("No suitable recommendations found.")
        elif len(recommendations_with_details) < 3: print(f"Warning: Returning only {len(recommendations_with_details)} recommendations.")
        return RecommendationResponse( recommendations=recommendations_with_details, processing_time_seconds=round(processing_time, 3) )

    except HTTPException as http_exc: raise http_exc
    except RuntimeError as runtime_exc: print(f"Runtime Error during recommendation: {runtime_exc}"); raise HTTPException(status_code=500, detail=f"Processing error: {runtime_exc}")
    except Exception as e: print(f"Unexpected Error processing /recommend request: {e}"); import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Internal server error occurred.")


# --- Root Endpoint (Updated Status Message) ---
@app.get("/")
async def root():
    db_status = "Not Connected"
    if mongo_client:
        try: mongo_client.admin.command('ismaster'); db_status = "Connected"
        except (ConnectionFailure, OperationFailure): db_status = "Connection Lost/Error"
        except Exception: db_status = "Connection Check Error"
    model_status = "Loaded" if _is_model_loaded() else "Not Loaded Yet (or Load Failed)"
    scaler_status = "Loaded" if scaler is not None else "Not Loaded/Error"
    encoder_status = "Loaded" if label_encoder is not None else "Not Loaded/Error"
    image_map_status = f"Loaded ({len(crop_image_map)} images from DB)" if crop_image_map is not None else "Not Loaded/Error"
    # Check for the NEW suitability data
    suitability_data_status = f"Loaded ({df_crop_suitability.shape[0]} crops from NEW CSV)" if df_crop_suitability is not None else "Not Loaded/Error"
    disease_data_status = f"Loaded ({df_disease.shape[0]} diseases from DB, {len(disease_crop_map)} crops mapped)" if df_disease is not None else "Not Loaded/Error"
    model_download_source_status = "Set" if MODEL_DOWNLOAD_URL_OR_ID else "Not Set (Required for first run)"

    return {
        "message": "Crop Recommendation API V10 (Ranges Suitability) is running.", # Updated version message
        "status_check": {
             "database_connection": db_status,
             "ml_model_status": model_status,
             "ml_scaler_status": scaler_status,
             "ml_encoder_status": encoder_status,
             "model_download_source": model_download_source_status,
             "crop_suitability_data": suitability_data_status, # Reflects NEW CSV
             "crop_image_data": image_map_status,
             "disease_data": disease_data_status
        }
    }

# Helper to check model load status without triggering load/download (Unchanged)
def _is_model_loaded():
    with model_load_lock: return crop_model is not None

# --- uvicorn command for local testing ---
# Ensure you have 'gdown' and other dependencies in your requirements.txt!
# Make sure 'crop_ideal_conditions_new.csv' is in the 'data' folder.
# python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
