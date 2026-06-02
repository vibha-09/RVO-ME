import React, { useState } from 'react';
import { analyzeImage } from './services/api';
import ImageUploader from './components/ImageUploader';
import ResultsPanel from './components/ResultsPanel';
import MetricsDisplay from './components/MetricsDisplay';
import PhenotypingCard from './components/PhenotypingCard';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await analyzeImage(file);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen font-sans bg-slate-50">
      <header className="bg-medical-blue text-white py-6 shadow-md">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold tracking-tight">OCT Analysis System</h1>
          <p className="text-blue-100 mt-1 text-sm opacity-90">RVO-ME Detection & Treatment Prediction</p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {!results && !loading && (
          <div className="max-w-3xl mx-auto flex flex-col items-center justify-center mt-12 bg-white p-10 rounded-2xl shadow-sm border border-slate-100 transition-all hover:shadow-md">
             <h2 className="text-2xl font-semibold mb-6 text-slate-800">Welcome to OCT Analysis</h2>
             <ImageUploader onUpload={handleUpload} />
             {error && <div className="mt-6 text-red-500 bg-red-50 px-4 py-3 rounded-lg border border-red-100 w-full text-center">{error}</div>}
          </div>
        )}

        {loading && (
          <div className="max-w-3xl mx-auto flex flex-col items-center justify-center mt-20 py-20 bg-white rounded-2xl shadow-sm border border-slate-100">
             <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-medical-blue mb-6"></div>
             <p className="text-slate-600 font-medium animate-pulse">Running Analysis Pipeline...</p>
             <p className="text-slate-400 text-sm mt-2">Segmentation · Detection · Quantification · GMM Phenotyping</p>
          </div>
        )}

        {results && !loading && (
          <div className="animate-fade-in-up">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-slate-800">Analysis Results</h2>
                <button 
                  onClick={() => setResults(null)}
                  className="px-6 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium rounded-lg transition-colors border border-slate-200"
                >
                  Analyze New Image
                </button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 space-y-8">
                <ResultsPanel images={results} />
                <MetricsDisplay metrics={results.metrics} />
              </div>
              <div className="lg:col-span-1">
                <div className="sticky top-8">
                  <PhenotypingCard phenotyping={results.phenotyping} />
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
