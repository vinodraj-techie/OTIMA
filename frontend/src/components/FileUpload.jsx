import { useState } from 'react';
import { UploadCloud, CheckCircle, Database } from 'lucide-react';

export default function FileUpload({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/optimize", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to optimize dataset");
      }

      const data = await response.json();
      onUploadSuccess(data.run_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card">
      <h2><Database style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} /> Upload Dataset</h2>
      <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', marginTop: '0.5rem' }}>
        Provide a ZIP file containing your warehouse data to run the OPTIMA pipeline.
      </p>

      <div 
        className={`upload-area ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-upload').click()}
      >
        <UploadCloud className="upload-icon" />
        <h3 style={{ marginBottom: '0.5rem' }}>Drag & Drop your ZIP file here</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>or click to browse from your computer</p>
        
        <input 
          id="file-upload" 
          type="file" 
          accept=".zip" 
          style={{ display: 'none' }}
          onChange={handleChange}
        />

        {file && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <CheckCircle color="var(--primary)" size={20} />
            <span style={{ fontWeight: 600 }}>{file.name}</span>
          </div>
        )}
      </div>

      <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        {error && <div style={{ color: '#ef4444' }}>{error}</div>}
        {!error && <div></div>}
        
        <button 
          className="btn" 
          onClick={handleSubmit} 
          disabled={!file || loading}
        >
          {loading ? (
            <>
              <div className="loader" style={{ width: '16px', height: '16px', border: '2px solid white', borderTopColor: 'transparent', borderRadius: '50%' }} /> Processing...
            </>
          ) : (
            'Start Optimization'
          )}
        </button>
      </div>
    </div>
  );
}
