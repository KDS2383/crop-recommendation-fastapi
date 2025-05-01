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

# --- Local CSV File Path ---
CROP_CONDITIONS_FILE = os.path.join(DATA_DIR, 'crop_ideal_conditions_curated.csv')
# DISEASE_CONDITIONS_FILE = os.path.join(DATA_DIR, 'disease_conditions.csv') # No longer needed

# --- MongoDB Configuration ---
ATLAS_USERNAME = os.getenv("ATLAS_USERNAME", "precisionfarming")
ATLAS_PASSWORD = os.getenv("ATLAS_PASSWORD", "finalyear%40909")
ATLAS_CLUSTER_URL = os.getenv("ATLAS_CLUSTER_URL", "cluster0.ogil0jz.mongodb.net")
DB_NAME_CROP_INFO = os.getenv("MONGO_DB_CROP_INFO", "Crop_Information")
DB_NAME_CROP_IMG = os.getenv("MONGO_DB_CROP_IMG", "Crop_Img")
DB_NAME_DISEASE_INFO = os.getenv("MONGO_DB_DISEASE_INFO", "Disease_Info")
CROP_DETAILS_COLLECTION = os.getenv("MONGO_CROP_DETAILS_COLLECTION", "crop_details")
CROP_IMAGES_COLLECTION = os.getenv("MONGO_CROP_IMAGES_COLLECTION", "crop_images")
DISEASE_COLLECTION = os.getenv("MONGO_DISEASE_COLLECTION", "diseases_info_std_3") # Use the standardized collection

ATLAS_CONNECTION_STRING_BASE = f"mongodb+srv://{ATLAS_USERNAME}:{ATLAS_PASSWORD}@{ATLAS_CLUSTER_URL}/?retryWrites=true&w=majority"

# --- Pydantic Models ---
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

# --- Feature Columns ---
MODEL_FEATURE_COLS = [ 'elevation', 'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'soil_moisture', 'soil_cec', 'avg_temperature', 'min_temperature', 'avg_humidity', 'min_humidity', 'avg_wind_speed', 'total_rainfall' ]

# --- Recommendation Logic Parameters ---
TOP_N_FROM_MODEL = 20
PARAMETER_WEIGHTS = { 'Avg_Temperature': 2.0, 'Min_Temperature': 2.0, 'Total_Rainfall': 2.5, 'Soil_pH': 1.5, 'Soil_Moisture': 1.5, 'Soil_Nitrogen': 1.0, 'Soil_Phosphorus': 1.0, 'Soil_Potassium': 1.0, 'Avg_Humidity': 1.0, 'Min_Humidity': 1.0, 'Soil_CEC': 0.75, 'Avg_Wind_Speed': 0.5 }
SUITABILITY_PARAM_COLS = list(PARAMETER_WEIGHTS.keys())

# --- Disease Prediction Parameters ---
DISEASE_PARAM_WEIGHTS = { 'Temperature': 2.0, 'Humidity': 2.5, 'Rainfall': 0.75, 'Soil_Moisture': 1.5, 'Soil_pH': 0.5, }
ENV_RANGES = { 'Temperature': 30, 'Humidity': 50, 'Rainfall': 50, 'Soil_Moisture': 60, 'Soil_pH': 3, }

# --- Crop Generalization Mapping ---
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
# Collection Handles
crop_details_collection = None; crop_images_collection = None
disease_collection = None
# Data loaded
crop_image_map = None # Atlas
df_crop_suitability = None # Local CSV
param_ranges = None # From local CSV
df_disease = None # Atlas (standardized)
disease_crop_map = {} # From Atlas data
model_load_lock = threading.Lock()

# --- Helper Function for Parsing Ranges ---
def parse_range(range_str):
    if pd.isna(range_str) or not isinstance(range_str, str): return None, None
    range_str = range_str.strip(); match = re.match(r'^([\d.]+)\s*[-–—]\s*([\d.]+)$', range_str)
    if match:
        try: return float(match.group(1)), float(match.group(2))
        except ValueError: return None, None
    try: return float(range_str), float(range_str)
    except ValueError: return None, None

# --- Startup Function (Simplified - All Disease Info from Atlas) ---
@app.on_event("startup")
async def startup_event():
    global scaler, label_encoder, mongo_client, crop_details_collection, crop_images_collection
    global disease_collection, crop_image_map
    global df_crop_suitability, param_ranges, df_disease, disease_crop_map, SUITABILITY_PARAM_COLS

    print("--- Starting FastAPI Application ---")
    print("Loading ML artifacts...")
    try:
        scaler = joblib.load(SCALER_LOAD_NAME)
        label_encoder = joblib.load(ENCODER_LOAD_NAME)
        print(f"OK: Loaded scaler and label encoder. Num classes: {len(label_encoder.classes_)}")
    except Exception as e: print(f"ERROR loading ML models: {e}"); scaler = label_encoder = None

    # --- Load Crop Suitability Conditions from LOCAL CSV ---
    print(f"Loading crop suitability conditions from LOCAL CSV: {CROP_CONDITIONS_FILE}...")
    try:
        cols_to_load_suitability = ['Crop', 'Min_Elevation', 'Max_Elevation'] + list(PARAMETER_WEIGHTS.keys())
        dtype_map_suitability = {col: 'float32' for col in list(PARAMETER_WEIGHTS.keys()) + ['Min_Elevation', 'Max_Elevation']}
        df_crop_suitability = pd.read_csv( CROP_CONDITIONS_FILE, usecols=lambda c: c in cols_to_load_suitability, index_col='Crop', dtype=dtype_map_suitability )
        SUITABILITY_PARAM_COLS = [col for col in PARAMETER_WEIGHTS.keys() if col in df_crop_suitability.columns]
        numeric_cols_for_suitability = SUITABILITY_PARAM_COLS + ['Min_Elevation', 'Max_Elevation']
        print(f"DEBUG: Actual suitability columns found in CSV: {SUITABILITY_PARAM_COLS}")
        for col in df_crop_suitability.columns:
            if col in numeric_cols_for_suitability: df_crop_suitability[col] = pd.to_numeric(df_crop_suitability[col], errors='coerce')
        if SUITABILITY_PARAM_COLS:
            param_ranges = df_crop_suitability[SUITABILITY_PARAM_COLS].max() - df_crop_suitability[SUITABILITY_PARAM_COLS].min()
            param_ranges = param_ranges.replace(0, 1e-6); print(f"OK: Loaded suitability data from CSV. Shape: {df_crop_suitability.shape}. Calculated param ranges.")
        else: print("Warning: No valid suitability parameter columns found in CSV."); df_crop_suitability = None; param_ranges = pd.Series(dtype='float32')
    except FileNotFoundError: print(f"ERROR: Crop suitability CSV file not found at {CROP_CONDITIONS_FILE}."); df_crop_suitability = None; param_ranges = None
    except Exception as e: print(f"ERROR loading suitability data from CSV: {e}"); df_crop_suitability = None; param_ranges = None

    # --- Connect to MongoDB Atlas ---
    print(f"Connecting to MongoDB Atlas: {ATLAS_CLUSTER_URL}...")
    try:
        mongo_client = MongoClient(ATLAS_CONNECTION_STRING_BASE, serverSelectionTimeoutMS=7000)
        mongo_client.list_database_names()
        print("OK: Connected to MongoDB Atlas")
        # Get Specific Database handles
        db_crop_info = mongo_client[DB_NAME_CROP_INFO]
        db_crop_img = mongo_client[DB_NAME_CROP_IMG]
        db_disease_info = mongo_client[DB_NAME_DISEASE_INFO]
        # Get Collection handles needed
        crop_details_collection = db_crop_info[CROP_DETAILS_COLLECTION]
        crop_images_collection = db_crop_img[CROP_IMAGES_COLLECTION]
        disease_collection = db_disease_info[DISEASE_COLLECTION] # Use the standardized collection
        print("OK: Got handles for required MongoDB collections.")

        # --- Load Crop Images from MongoDB ---
        print(f"Loading crop images from DB '{DB_NAME_CROP_IMG}', Collection '{CROP_IMAGES_COLLECTION}'...")
        crop_image_map = {}
        try:
            cursor = crop_images_collection.find({}, {"crop_name": 1, "image_url": 1, "_id": 0})
            for doc in cursor:
                if doc.get("crop_name") and doc.get("image_url"): crop_image_map[doc["crop_name"]] = doc["image_url"]
            if not crop_image_map: print("Warning: Crop images collection is empty or data missing.")
            print(f"OK: Loaded {len(crop_image_map)} crop image URLs from MongoDB.")
        except Exception as e: print(f"ERROR loading crop images from MongoDB: {e}"); crop_image_map = {}

        # --- Load ALL Disease Data from MongoDB ---
        print(f"Loading ALL disease data from DB '{DB_NAME_DISEASE_INFO}', Collection '{DISEASE_COLLECTION}'...")
        try:
            disease_docs = list(disease_collection.find({}, {"_id": 0}))
            if not disease_docs: print(f"ERROR: No documents found in disease collection '{DISEASE_COLLECTION}'."); df_disease = None; disease_crop_map = {}
            else:
                df_disease = pd.DataFrame.from_records(disease_docs)
                # Ensure required columns exist and handle types/NAs
                for col in ['name', 'display_name', 'crop_hosts', 'symptoms', 'treatment', 'soil_pH_Preference']:
                    if col not in df_disease.columns: df_disease[col] = None
                df_disease['name'].fillna('unknown_disease', inplace=True); df_disease['display_name'].fillna(df_disease['name'], inplace=True)
                df_disease['crop_hosts'].fillna('', inplace=True); df_disease['symptoms'].fillna('Not Available', inplace=True)
                df_disease['treatment'].fillna('Not Available', inplace=True)
                # Ensure numeric types
                disease_numeric_cols = [ 'min_Optimal_Temp', 'max_Optimal_Temp', 'min_Optimal_Humidity', 'max_Optimal_Humidity', 'min_Rainfall_mm', 'max_Rainfall_mm', 'min_Soil_Moisture', 'max_Soil_Moisture' ]
                for col in disease_numeric_cols:
                    if col in df_disease.columns: df_disease[col] = pd.to_numeric(df_disease[col], errors='coerce')
                    else: print(f"Warning: Disease condition column '{col}' not found in MongoDB collection."); df_disease[col] = np.nan

                # --- Rebuild disease_crop_map using Atlas data ---
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
                            if general_category:
                                if general_category not in disease_crop_map: disease_crop_map[general_category] = []
                                if index not in disease_crop_map[general_category]: disease_crop_map[general_category].append(index)
                print(f"OK: Disease data loaded from Atlas. Shape: {df_disease.shape}. Crop map size: {len(disease_crop_map)}")
                print(f"DEBUG: Found {len(unique_disease_hosts_found)} unique hosts in disease collection.")

        except Exception as e: print(f"ERROR loading disease data from MongoDB: {e}"); df_disease = None; disease_crop_map = {}

    except Exception as e: # Catch DB connection or other errors
        print(f"FATAL ERROR during MongoDB connection or initial data loading: {e}")
        mongo_client = None; crop_details_collection = None; crop_images_collection = None; disease_collection = None
        crop_image_map = {}; df_disease = None; disease_crop_map = {}
        if 'df_crop_suitability' not in locals(): df_crop_suitability = None; param_ranges = None

    print("--- Application Startup Complete ---")

# --- Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client: mongo_client.close(); print("MongoDB connection closed.")

# --- get_model ---
def get_model():
    global crop_model
    if crop_model is None:
        with model_load_lock:
            if crop_model is None:
                print("Loading Crop Recommendation model ON DEMAND...")
                load_start_time = time.time()
                try:
                    if not os.path.exists(MODEL_LOAD_NAME): raise FileNotFoundError(f"Model file not found at {MODEL_LOAD_NAME}")
                    crop_model = joblib.load(MODEL_LOAD_NAME)
                    load_end_time = time.time(); print(f"Model loaded successfully in {load_end_time - load_start_time:.2f}s.")
                except Exception as e: print(f"ERROR: Failed to load model: {e}"); crop_model = None
    return crop_model

# === Helper Functions ===

# --- calculate_suitability ---
def calculate_suitability(loc_conditions, crop_ideal, param_cols, ranges, param_weights, elevation_penalty=100, crop_name_log="Unknown Crop", debug_print=False):
    total_dissimilarity = 0.0; loc_elevation = loc_conditions.get('elevation');
    if loc_elevation is not None and isinstance(loc_elevation, (int, float)):
        min_elev = crop_ideal.get('Min_Elevation'); max_elev = crop_ideal.get('Max_Elevation')
        if pd.notna(min_elev) and loc_elevation < min_elev: total_dissimilarity += float(elevation_penalty)
        if pd.notna(max_elev) and loc_elevation > max_elev: total_dissimilarity += float(elevation_penalty)
    calculated_count = 0
    for param in param_cols:
        loc_val = loc_conditions.get(param); ideal_val = crop_ideal.get(param)
        if loc_val is not None and ideal_val is not None and pd.notna(loc_val) and pd.notna(ideal_val) and isinstance(loc_val, (int, float)) and isinstance(ideal_val, (int, float)):
            param_range = ranges.get(param, 1e-6); difference = abs(loc_val - ideal_val); normalized_diff = difference / param_range
            weight = param_weights.get(param, 1.0); weighted_diff = normalized_diff * weight; total_dissimilarity += weighted_diff; calculated_count += 1
    return total_dissimilarity if calculated_count > 0 else 1e9

# --- fetch_crop_info ---
def fetch_crop_info(crop_names: List[str]) -> Dict[str, Dict[str, Any]]:
    if crop_details_collection is None or not crop_names: return {}
    found_crops_info = {}
    try:
        print(f"Querying MongoDB '{CROP_DETAILS_COLLECTION}' for info on: {crop_names}")
        cursor = crop_details_collection.find( {"crop_name": {"$in": crop_names}}, {"_id": 0} )
        count = 0; [found_crops_info.update({doc["crop_name"]: doc}) or (count := count + 1) for doc in cursor if "crop_name" in doc]
        print(f"Found info for {count} crops in MongoDB.")
    except Exception as e: print(f"Error during MongoDB fetch from '{CROP_DETAILS_COLLECTION}': {e}")
    return found_crops_info

# --- calculate_disease_risk_score ---
def calculate_disease_risk_score(user_conditions: Dict[str, Any], disease_data: pd.Series) -> float:
    total_risk_score = 0.0; params_calculated = 0
    def calculate_param_score(user_val, min_opt, max_opt, env_range):
        if user_val is None or pd.isna(min_opt) or pd.isna(max_opt): return None
        score = 0.0; env_range = max(env_range, 1e-6) # Avoid division by zero
        if user_val < min_opt: score = (min_opt - user_val) / env_range
        elif user_val > max_opt: score = (user_val - max_opt) / env_range
        return score
    # Use standardized field names expected after loading/renaming
    temp_score=calculate_param_score(user_conditions.get('avg_temperature'),disease_data.get('min_Optimal_Temp'),disease_data.get('max_Optimal_Temp'),ENV_RANGES['Temperature'])
    if temp_score is not None:total_risk_score+=temp_score*DISEASE_PARAM_WEIGHTS['Temperature'];params_calculated+=1
    hum_score=calculate_param_score(user_conditions.get('avg_humidity'),disease_data.get('min_Optimal_Humidity'),disease_data.get('max_Optimal_Humidity'),ENV_RANGES['Humidity'])
    if hum_score is not None:total_risk_score+=hum_score*DISEASE_PARAM_WEIGHTS['Humidity'];params_calculated+=1
    moist_score=calculate_param_score(user_conditions.get('soil_moisture'),disease_data.get('min_Soil_Moisture'),disease_data.get('max_Soil_Moisture'),ENV_RANGES['Soil_Moisture'])
    if moist_score is not None:total_risk_score+=moist_score*DISEASE_PARAM_WEIGHTS['Soil_Moisture'];params_calculated+=1
    user_rainfall=user_conditions.get('total_rainfall');min_opt_rain=disease_data.get('min_Rainfall_mm');max_opt_rain=disease_data.get('max_Rainfall_mm')
    if user_rainfall is not None and(pd.notna(min_opt_rain)or pd.notna(max_opt_rain)):
        rain_score=0.5
        if pd.notna(min_opt_rain)and user_rainfall<min_opt_rain:rain_score+=1.0
        elif pd.notna(min_opt_rain)and pd.notna(max_opt_rain)and min_opt_rain<=user_rainfall<=max_opt_rain:rain_score=0.1
        total_risk_score+=rain_score*DISEASE_PARAM_WEIGHTS['Rainfall'];params_calculated+=1
    user_ph=user_conditions.get('soil_ph');min_opt_ph,max_opt_ph=parse_range(disease_data.get('soil_pH_Preference')) # Use correct field name
    ph_score=calculate_param_score(user_ph,min_opt_ph,max_opt_ph,ENV_RANGES['Soil_pH'])
    if ph_score is not None:total_risk_score+=ph_score*DISEASE_PARAM_WEIGHTS['Soil_pH'];params_calculated+=1
    if params_calculated==0:return 999.0
    return total_risk_score/params_calculated

# --- get_disease_predictions (UPDATED - Uses Atlas df_disease, returns unique details) ---
def get_disease_predictions(crop_name: str, user_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ Predicts top 3 diseases based on Atlas data and returns their UNIQUE details. """
    if df_disease is None or not disease_crop_map: return [] # Check Atlas DF
    disease_indices = disease_crop_map.get(crop_name, [])
    if not disease_indices: return []

    potential_diseases = []
    processed_indices = set()
    for index in disease_indices:
        if index >= len(df_disease) or index in processed_indices: continue
        processed_indices.add(index)
        disease_data = df_disease.iloc[index]
        disease_display_name = disease_data.get('display_name', 'Unknown Disease')
        if pd.isna(disease_display_name) or disease_display_name == 'Unknown Disease': continue
        try:
            risk_score = calculate_disease_risk_score(user_conditions, disease_data)
            potential_diseases.append({
                "name": disease_display_name, # Use display name for grouping & final output
                "symptoms": disease_data.get('symptoms') if pd.notna(disease_data.get('symptoms')) and disease_data.get('symptoms') != 'Not Available' else None,
                "treatment": disease_data.get('treatment') if pd.notna(disease_data.get('treatment')) and disease_data.get('treatment') != 'Not Available' else None,
                "_score": risk_score })
        except Exception as e: print(f"Warning: Error scoring disease index {index} ({disease_display_name}): {e}")

    if not potential_diseases: return []

    unique_best_diseases = {}
    for disease in potential_diseases:
        name = disease['name']; score = disease['_score']
        if name not in unique_best_diseases or score < unique_best_diseases[name]['_score']:
            unique_best_diseases[name] = disease

    sorted_unique_diseases = sorted(unique_best_diseases.values(), key=lambda x: x['_score'])
    top_3_disease_details = [{k: v for k, v in d.items() if k != '_score'} for d in sorted_unique_diseases[:3]]
    print(f"Top predicted diseases for {crop_name}: {[d['name'] for d in top_3_disease_details]}")
    return top_3_disease_details


# --- get_recommendations_with_info (Simplified disease part) ---
def get_recommendations_with_info(user_input_dict, top_n_model, scaler, encoder, df_crop_global,
                                  feature_cols_model, param_cols_suitability,
                                  param_ranges_suitability, param_weights_suitability,
                                  exclude_crops: List[str] = []) -> List[Dict[str, Any]]:
    model = get_model()
    if model is None: raise RuntimeError("ML Model not available")
    if df_crop_global is None or param_ranges_suitability is None: raise RuntimeError("Crop suitability data not loaded.")

    # 1. Get Top N Model Predictions
    try:
        feature_values = [user_input_dict.get(col) for col in feature_cols_model]
        input_df = pd.DataFrame([feature_values], columns=feature_cols_model); input_df.fillna(0, inplace=True)
        input_scaled = scaler.transform(input_df); probabilities = model.predict_proba(input_scaled)[0]
        top_n_indices = np.argsort(probabilities)[::-1][:top_n_model]; top_n_probabilities = probabilities[top_n_indices]
        top_n_crop_names = encoder.inverse_transform(top_n_indices)
        print(f"DEBUG: Exact crop names from encoder: {top_n_crop_names[:5]}...")
    except Exception as e: print(f"Error during ML prediction steps: {e}"); raise RuntimeError(f"ML prediction failed: {e}")
    print(f"\nTop {top_n_model} crops from ML (before re-rank): {list(top_n_crop_names)}")

    # 2. Re-rank using Crop Suitability Score (Uses local data)
    ranked_candidates = []
    for i in range(len(top_n_crop_names)):
        crop_name = top_n_crop_names[i]
        if crop_name in df_crop_global.index:
            ideal_conditions_dict = df_crop_global.loc[crop_name].to_dict()
            try:
                loc_conditions = { "elevation": user_input_dict.get('elevation'), "Soil_pH": user_input_dict.get('soil_ph'), "Soil_Nitrogen": user_input_dict.get('soil_nitrogen'), "Soil_Phosphorus": user_input_dict.get('soil_phosphorus'), "Soil_Potassium": user_input_dict.get('soil_potassium'), "Soil_Moisture": user_input_dict.get('soil_moisture'), "Soil_CEC": user_input_dict.get('soil_cec'), "Avg_Temperature": user_input_dict.get('avg_temperature'), "Min_Temperature": user_input_dict.get('min_temperature'), "Avg_Humidity": user_input_dict.get('avg_humidity'), "Min_Humidity": user_input_dict.get('min_humidity'), "Avg_Wind_Speed": user_input_dict.get('avg_wind_speed'), "Total_Rainfall": user_input_dict.get('total_rainfall') }
                current_suit_cols = [col for col in param_cols_suitability if col in ideal_conditions_dict and pd.notna(ideal_conditions_dict[col])]
                suitability_score = calculate_suitability( loc_conditions, ideal_conditions_dict, current_suit_cols, param_ranges_suitability, param_weights_suitability, crop_name_log=crop_name ) if current_suit_cols else 1e9
                ranked_candidates.append({'crop': crop_name, 'ml_prob': top_n_probabilities[i], 'suitability_score': suitability_score})
            except Exception as e_calc: print(f"Warn: Suitability calc error for {crop_name}: {e_calc}. High score."); ranked_candidates.append({'crop': crop_name, 'ml_prob': top_n_probabilities[i], 'suitability_score': 1e9})
        else: print(f"Warn: Crop '{crop_name}' not in suitability index.")
    ranked_candidates.sort(key=lambda x: (x.get('suitability_score', 1e9), -x['ml_prob']))

    # 3. Filtering Historical Crops (Unchanged)
    if exclude_crops: exclude_crops_set = set(exclude_crops); filtered_candidates = [c for c in ranked_candidates if c['crop'] not in exclude_crops_set]; print(f"\nExcluding historical crops: {exclude_crops_set}")
    else: filtered_candidates = ranked_candidates
    print("\nTop candidates AFTER re-ranking & filtering:"); [print(f"  - {item['crop']}: Suitability={item.get('suitability_score', 'N/A'):.4f}, Prob={item['ml_prob']:.4f}") for item in filtered_candidates[:10]]

    # 4. Get Top 3 Crop NAMES (Unchanged)
    top_3_names = [item['crop'] for item in filtered_candidates[:3]]
    print(f"\nFinal Top 3 Recommended Crop Names: {top_3_names}")

    # 5. Fetch Crop Details & Image (Atlas), Predict Diseases with Details (Atlas)
    detailed_recommendations = []
    if top_3_names:
        fetched_info_map = fetch_crop_info(top_3_names) # Atlas crop details
        for name in top_3_names:
            crop_detail = fetched_info_map.get(name, {"crop_name": name, "crop_info": "Detailed info unavailable."})
            crop_detail["image_url"] = crop_image_map.get(name) if crop_image_map else None # Atlas images

            # --- Predict Diseases & Add Details (Simplified) ---
            # get_disease_predictions now uses Atlas data and returns details directly
            predicted_disease_details = get_disease_predictions(name, user_input_dict)
            crop_detail["diseases"] = predicted_disease_details # Add full details list
            crop_detail["predicted_diseases"] = [d['name'] for d in predicted_disease_details] # Keep simple list of names

            detailed_recommendations.append(crop_detail)
    return detailed_recommendations

# === FastAPI Endpoint ===
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_crops_endpoint(user_input: UserInput):
    start_time = time.time()
    # Verify prerequisites (local suitability CSV, Atlas data/handles, model)
    if not all([ scaler is not None, label_encoder is not None,
                 df_crop_suitability is not None, param_ranges is not None, # Local suitability
                 crop_image_map is not None, # Atlas images map
                 df_disease is not None, disease_crop_map is not None, # Atlas disease data
                 crop_details_collection is not None, disease_collection is not None # Atlas handles
                ]):
        print("ERROR: One or more required data components missing or failed to load during startup.")
        raise HTTPException(status_code=503, detail="Server configuration error: Core data not ready.")
    if get_model() is None:
         raise HTTPException(status_code=503, detail="Server configuration error: ML Model could not be loaded.")

    try:
        user_conditions_dict = user_input.dict()
        # Pass the correct global variables
        recommendations_with_details = get_recommendations_with_info(
            user_input_dict=user_conditions_dict, top_n_model=TOP_N_FROM_MODEL,
            scaler=scaler, encoder=label_encoder, df_crop_global=df_crop_suitability, # Use local DF
            feature_cols_model=MODEL_FEATURE_COLS, param_cols_suitability=SUITABILITY_PARAM_COLS, # Use cols from CSV
            param_ranges_suitability=param_ranges, param_weights_suitability=PARAMETER_WEIGHTS, # Use ranges from CSV
            exclude_crops=user_input.historical_crops )

        end_time = time.time(); processing_time = end_time - start_time
        if not recommendations_with_details: print("No suitable recommendations found or info missing in DB.")
        elif len(recommendations_with_details) < 3: print(f"Warning: Returning only {len(recommendations_with_details)} recommendations.")
        return RecommendationResponse( recommendations=recommendations_with_details, processing_time_seconds=round(processing_time, 3) )

    except HTTPException as http_exc: raise http_exc
    except RuntimeError as runtime_exc: print(f"Runtime Error during recommendation: {runtime_exc}"); raise HTTPException(status_code=500, detail=f"Processing error: {runtime_exc}")
    except Exception as e: print(f"Unexpected Error processing /recommend request: {e}"); import traceback; traceback.print_exc(); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- Root Endpoint ---
@app.get("/")
async def root():
    db_status = "Connected" if mongo_client is not None else "Not Connected"
    model_status = "Loaded" if _is_model_loaded() else "Not Loaded Yet"
    image_map_status = f"Loaded ({len(crop_image_map)} images from DB)" if crop_image_map is not None else "Not Loaded/Error"
    suitability_data_status = f"Loaded ({df_crop_suitability.shape[0]} crops from CSV)" if df_crop_suitability is not None else "Not Loaded/Error"
    # Updated status check for disease data loaded from Atlas
    disease_data_status = f"Loaded ({df_disease.shape[0]} diseases from DB, {len(disease_crop_map)} crops mapped)" if df_disease is not None else "Not Loaded/Error"

    return { "message": "Crop Recommendation API V7 (Atlas Diseases) is running.",
        "model_status": model_status, "database_status": db_status,
        "image_map_status": image_map_status,
        "suitability_data_status": suitability_data_status,
        "disease_data_status": disease_data_status # Updated key name
        }

# Helper to check model load status without triggering load
def _is_model_loaded():
    with model_load_lock: return crop_model is not None





# --- uvicorn command ---
# python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000