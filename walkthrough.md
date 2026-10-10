# React Frontend Walkthrough

I have successfully created and integrated a beautiful, modern React application for the OPTIMA API! 

## Backend Setup

1. **CORS Configured**: I updated `api/main.py` using `CORSMiddleware` so the new Vite frontend at port `5173` can cleanly talk to the FastAPI backend.
2. **Static Volumes Mounted**: Added `StaticFiles` in `main.py` to route `/results` calls directly to the generated CSV and PNG results folder, allowing the UI to actually render the visualizations!

## Frontend Setup

1. Built a Vite React foundation in the `frontend` folder (`d:\OTIMA\frontend`).
2. Installed `lucide-react` for beautiful iconography and streamlined UX.
3. Created an all-new design system within `index.css`:
   - Deep rich navy/slate backgrounds (`--bg-dark: #0f172a`).
   - Vibrant purple/blue gradients mapped to text using `-webkit-background-clip: text`.
   - Subtle glassmorphism across components (`rgba` + `backdrop-filter: blur(16px)`).
   - Fluid keyframe animations (`@keyframes float` and smooth interactive hover effects).

## Features Designed

### The Global Layout (`App.jsx`)
Features a sticky style glass-header with a bold gradient logo. Sets up dynamic state management that toggles the UI into two core phases: **Upload** vs **Dashboard**.

### The File Uploader (`FileUpload.jsx`)
A clean, interactive drop zone component.
- Implements `dragenter` and `drop` overrides so users can drag a ZIP directly onto the screen.
- POSTs directly to `http://localhost:8000/optimize` by appending to a `FormData` object.
- Features a loading `loader` spinner during extraction and processing.

### The Visualization Dashboard (`Dashboard.jsx`)
A comprehensive dashboard to review optimization results elegantly grid-mapped.
- Queries `http://localhost:8000/visualize/{runId}` upon mounting.
- Knows the pre-defined charts (ABC, GMM Clusters, Forecast, Route mapping, etc.) and seamlessly forms URLs (`http://localhost:8000/results/run_{runId}/charts/...")` to load them beautifully into image cards.
- Provides robust custom `onError` callbacks for images to render a smooth "Data not available" empty state if an execution didn't yield a specific graph (e.g. SLAP optimization was skipped).

## Verification Checks

- [x] Connected components logically without unmounting leaks.
- [x] Added generic error boundary displays to gracefully display Axios / Fetch networking errors.
- [x] Tested syntax with an `npm run build` which compiled flawlessly.

You can now start both your backends via `uvicorn api.main:app` and frontend via `npm run dev` in the `frontend` application to start playing with the application.



# Navigate into the frontend folder we just created
cd d:\OTIMA\frontend

# Start the Vite development server
npm run dev



# Navigate to the root directory
cd d:\OTIMA

# Activate your virtual environment (if you are using one)
.\venv\Scripts\activate

# Start the Python server using Uvicorn
uvicorn api.main:app --reload
