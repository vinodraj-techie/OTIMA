from fastapi import FastAPI
from api.routes import optimize
from api.routes import visualization

app = FastAPI(
    title="OPTIMA Warehouse Optimization API",
    version="1.0"
)

app.include_router(optimize.router)
app.include_router(visualization.router)

@app.get("/")
def home():
    return {"message": "OPTIMA backend running"}