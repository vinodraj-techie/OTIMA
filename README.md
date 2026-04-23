# OPTIMA Warehouse Optimization Platform

OPTIMA is an automated pipeline consisting of a **FastAPI Machine Learning Backend** and a beautiful **React Frontend Sandbox** where you can upload your warehouse datasets and observe intelligent AI-driven insights—like storage distributions, SLAP layout configurations, picking routes, and data-driven supply chain forecasts.

## How to Run the Project

This system requires two separate terminals: one to run the Python inference backend, and one to run the frontend interface.

### 1. Start the Backend API (FastAPI)
1. Open up your first terminal window.
2. Navigate to the root directory where this `README.md` is located (`d:\OTIMA`).
3. Make sure to activate your virtual environment if you possess one:
   ```powershell
   .\venv\Scripts\activate
   ```
4. Start the server using Uvicorn:
   ```powershell
   uvicorn api.main:app --reload
   ```
*The backend will now be actively listening on `http://127.0.0.1:8000`.*

### 2. Start the Frontend Application (Vite + React)
1. Open a **second**, new terminal window.
2. Navigate directly into the frontend directory:
   ```powershell
   cd frontend
   ```
3. Boot up the Vite development server:
   ```powershell
   npm run dev
   ```
*The frontend will boot up and provide a local link, typically `http://localhost:5173`. Hold CTRL/CMD and click the link to explore your warehouse sandbox!*

---

## Technical Flow Overview
1. You provide a `.zip` file into the frontend containing CSV datasets (like `warehouse_layout.csv` and `pick_list.csv`).
2. The zip is sent to the backend endpoint `POST /optimize`.
3. The inference pipelines extract it, run Feature Engineering (GMM, ABC), and push data through predictive modeling (DQN, SLAP, LSTM).
4. The outputs are generated into a new `results/run_{timestamp}` folder.
5. The frontend polls `GET /visualize/{run_id}` to receive the resulting PNG graphs to cleanly display within the dynamic dashboard!
