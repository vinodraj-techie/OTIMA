from fastapi import APIRouter
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