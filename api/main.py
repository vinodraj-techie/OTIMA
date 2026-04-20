from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from api.routes import optimize
from api.routes import visualization

app = FastAPI(
    title="OPTIMA Warehouse Optimization API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create results folder if not exists
os.makedirs("results", exist_ok=True)
app.mount("/results", StaticFiles(directory="results"), name="results")

app.include_router(optimize.router)
app.include_router(visualization.router)

@app.get("/")
def home():
    return {"message": "OPTIMA backend running"}