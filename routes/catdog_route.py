import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import File, UploadFile
from fastapi import APIRouter
from schemas.catdog_schema import CatDogResponse
from config.catdog_cfg import ModelConfig
from models.catdog_predictor import Predictor

router = APIRouter()
predictor = Predictor(
        model_name=ModelConfig.MODEL_NAME,
        model_weights=ModelConfig.MODEL_WEIGHT,
        device=ModelConfig.DEVICE
)

@router.post("/predict")
async def predict(file_upload:UploadFile = File(...)):
        response = await predictor.predict(file_upload.file)
        return CatDogResponse(**response)


