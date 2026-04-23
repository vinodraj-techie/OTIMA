from inference.run_pipeline import run_pipeline

def run_optima_pipeline(dataset_path):

    results_folder = run_pipeline(dataset_path)

    return results_folder