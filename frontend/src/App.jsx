import { useState } from 'react';
import FileUpload from './components/FileUpload';
import Dashboard from './components/Dashboard';

function App() {
  const [currentRunId, setCurrentRunId] = useState(null);

  return (
    <div className="app-container">
      <header>
        <div style={{ width: '40px', height: '40px', background: 'var(--primary)', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '1.2rem', color: 'white' }}>
          O
        </div>
        <h1>OPTIMA</h1>
        <div style={{ marginLeft: 'auto', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
          Warehouse Intelligence
        </div>
      </header>

      <main>
        {!currentRunId ? (
          <FileUpload onUploadSuccess={setCurrentRunId} />
        ) : (
          <Dashboard runId={currentRunId} onReset={() => setCurrentRunId(null)} />
        )}
      </main>
    </div>
  );
}

export default App;
