import React, { useRef, useState } from 'react';

const ImageUploader = ({ onUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

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
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div 
      className={`w-full max-w-2xl border-2 border-dashed rounded-xl p-12 text-center transition-all ${dragActive ? 'border-medical-blue bg-blue-50/50 scale-[1.02]' : 'border-slate-300 hover:bg-slate-50'}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <div className="flex flex-col items-center justify-center space-y-4">
        <svg className="w-16 h-16 text-medical-blue/70" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
        </svg>
        <div className="space-y-1">
            <p className="text-xl font-medium text-slate-700">Drag and drop your OCT scan here</p>
            <p className="text-sm text-slate-500">Supports PNG, JPG, or JPEG up to 10MB</p>
        </div>
        <input 
          ref={inputRef}
          type="file" 
          accept="image/*" 
          onChange={handleChange} 
          className="hidden" 
        />
        <button 
            className="mt-6 px-8 py-3 bg-medical-blue text-white rounded-lg hover:bg-blue-700 hover:shadow-lg transition-all font-medium active:scale-95"
            onClick={() => inputRef.current.click()}
        >
          Select File
        </button>
      </div>
    </div>
  );
};

export default ImageUploader;
