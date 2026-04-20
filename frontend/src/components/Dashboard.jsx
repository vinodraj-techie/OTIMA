import { useState, useEffect } from 'react';
import { LayoutDashboard, ArrowLeft, Image as ImageIcon, DownloadCloud } from 'lucide-react';

export default function Dashboard({ runId, onReset }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // We know the potential chart names from the backend service
  const potentialCharts = [
    { id: 'abc_distribution.png', title: 'ABC Classification Distribution', desc: 'SKU categorization by value' },
    { id: 'gmm_clusters.png', title: 'GMM Cluster Distribution', desc: 'Product segmentation clusters' },
    { id: 'forecast_distribution.png', title: 'Demand Forecast Distribution', desc: 'Predicted future demand scope' },
    { id: 'slap_bin_usage.png', title: 'Top Used Warehouse Bins (SLAP)', desc: 'Optimized storage locations' },
    { id: 'dqn_route_distance.png', title: 'DQN Route Distance per Step', desc: 'Reinforcement learning picking route mapping' },
  ];

  useEffect(() => {
    // We notify the backend to generate the visualisations
    const fetchVisualizations = async () => {
      try {
        const res = await fetch(`http://localhost:8000/visualize/${runId}`);
        if (!res.ok) throw new Error("Failed to generate visualizations");
        // Once successful, we can just load the images directly using their known paths mapped via FastAPI's static mount
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchVisualizations();
  }, [runId]);

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <LayoutDashboard color="var(--primary)" /> Optimization Results
        </h2>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <a
            href={`http://localhost:8000/download/${runId}`}
            download
            className="btn"
            style={{ textDecoration: 'none' }}
          >
            <DownloadCloud size={16} /> Download Full Results
          </a>
          <button 
            className="btn" 
            onClick={onReset}
            style={{ background: '#334155', boxShadow: 'none' }}
          >
            <ArrowLeft size={16} /> New Run
          </button>
        </div>
      </div>

      <p style={{ color: 'var(--text-muted)' }}>
        Run ID: <span style={{ fontFamily: 'monospace', color: 'var(--primary)' }}>{runId}</span>
      </p>

      {loading && (
        <div style={{ padding: '4rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>
          <div className="loader" style={{ width: '40px', height: '40px', border: '3px solid var(--primary)', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 1rem' }} />
          <p>Generating intelligent visualizations...</p>
        </div>
      )}

      {error && (
        <div style={{ padding: '2rem', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '12px' }}>
          Error: {error}
        </div>
      )}

      {!loading && !error && (
        <div className="dashboard-grid">
          {potentialCharts.map((chart) => {
            const chartUrl = `http://localhost:8000/results/run_${runId}/charts/${chart.id}`;
            return (
              <div key={chart.id} className="chart-card">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.5rem' }}>
                  <ImageIcon size={18} color="var(--secondary)" />
                  <h3 style={{ fontSize: '1.1rem', margin: 0 }}>{chart.title}</h3>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1rem' }}>{chart.desc}</p>
                <div style={{ background: '#0f172a', borderRadius: '12px', padding: '0.5rem', border: '1px solid var(--border)' }}>
                  <img 
                    src={chartUrl} 
                    alt={chart.title} 
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.parentElement.innerHTML = `<div style="padding: 3rem 1rem; text-align: center; color: var(--text-muted)">Data not available for this metric</div>`;
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
