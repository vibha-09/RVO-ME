import React from 'react';

const ResultsPanel = ({ images }) => {
  return (
    <div className="glass-panel p-6 rounded-2xl">
      <h3 className="text-xl font-semibold mb-4 text-slate-800">Visual Analysis</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div className="space-y-2">
            <p className="text-sm font-medium text-slate-500">Original Scan</p>
            <div className="border border-slate-200 rounded-lg overflow-hidden relative group">
                <img src={images.original_image} alt="Original OCT" className="w-full h-auto object-contain bg-black" />
            </div>
        </div>

        <div className="space-y-2">
            <p className="text-sm font-medium text-slate-500">Enhanced (CLAHE)</p>
            <div className="border border-slate-200 rounded-lg overflow-hidden">
                <img src={images.enhanced_image} alt="Enhanced OCT" className="w-full h-auto object-contain bg-black" />
            </div>
        </div>

        <div className="space-y-2">
            <div className="flex justify-between items-center">
                <p className="text-sm font-medium text-slate-500">Segmentation Mask</p>
                <div className="flex gap-2 text-[10px] font-bold">
                    <span className="text-blue-500">SRF</span>
                    <span className="text-red-500">IRF</span>
                    <span className="text-yellow-500">ELM</span>
                    <span className="text-green-500">EZ</span>
                </div>
            </div>
            <div className="border border-slate-200 rounded-lg overflow-hidden bg-black/5">
                <img src={images.segmentation_mask} alt="Segmentation Mask" className="w-full h-auto object-contain mix-blend-multiply" />
            </div>
        </div>

        <div className="space-y-2">
            <p className="text-sm font-medium text-slate-500">Final Overlay</p>
            <div className="border border-slate-200 rounded-lg overflow-hidden relative">
                <img src={images.overlay_image} alt="Overlay" className="w-full h-auto object-contain bg-black" />
                <div className="absolute top-2 right-2 bg-black/70 text-white text-[10px] px-2 py-1 rounded-full border border-white/20">
                    <span className="text-cyan-400 font-bold mr-1">■</span> HF Detections
                </div>
            </div>
        </div>

      </div>
    </div>
  );
};

export default ResultsPanel;
