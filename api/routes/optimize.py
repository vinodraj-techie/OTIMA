from fastapi import APIRouter, UploadFile, File
import shutil
import os
from datetime import datetime
import zipfile

from api.services.pipeline_service import run_optima_pipeline

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/optimize")
async def optimize_dataset(file: UploadFile = File(...)):

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_folder = os.path.join(UPLOAD_DIR, f"run_{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    zip_path = os.path.join(run_folder, file.filename)

    # save uploaded zip
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # extract zip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(run_folder)

    # detect dataset folder
    dataset_path = run_folder
    extracted_items = os.listdir(run_folder)

    for item in extracted_items:
        item_path = os.path.join(run_folder, item)
        if os.path.isdir(item_path):
            dataset_path = item_path
            break

    # run pipeline
    results_path = run_optima_pipeline(dataset_path)

    return {
        "status": "success",
        "run_id": run_id,
        "dataset_path": dataset_path,
        "results_folder": results_path
    }