from fastapi import APIRouter
from fastapi.responses import FileResponse
import shutil
import os
from api.services.visualization_service import generate_visualizations

router = APIRouter()

@router.get("/visualize/{run_id}")
def visualize_results(run_id: str):
    results_path = f"results/run_{run_id}"
    charts_path = generate_visualizations(results_path)
    return {
        "status": "success",
        "charts_folder": charts_path
    }

@router.get("/download/{run_id}")
def download_results(run_id: str):
    results_path = f"results/run_{run_id}"
    zip_filename = f"results_run_{run_id}"
    zip_filepath = f"results/{zip_filename}.zip"
    
    # Create the zip archive from the results directory if it exists
    if os.path.exists(results_path):
        shutil.make_archive(f"results/{zip_filename}", 'zip', results_path)
        return FileResponse(path=zip_filepath, filename=f"{zip_filename}.zip", media_type='application/zip')
    
    return {"status": "error", "message": "Results not found"}