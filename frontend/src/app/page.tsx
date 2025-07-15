'use client';

import { useState, useRef } from 'react';
import Image from 'next/image';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [generating, setGenerating] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
  };

  const handleGenerate = async () => {
    if (!file) return;
    setGenerating(true);
    await new Promise(r => setTimeout(r, 2000));   //  mock API call
    setGenerating(false);
    console.log('Generating 3D model from uploaded file');
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-4">
      {/* Smaller, centered card */}
      <div className="flex flex-col lg:flex-row gap-6 bg-[#0f0f11]/95 rounded-2xl shadow-2xl p-6 md:p-8 w-full max-w-4xl items-center">

        {/* ───────── LEFT COL ───────── */}
        <section className="w-full lg:w-1/2 space-y-6">
          <header className="space-y-3 text-center lg:text-left">
            <h1 className="text-3xl lg:text-4xl font-extrabold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              2D → 3D Converter
            </h1>
            <p className="text-gray-400 text-base">
              Transform your 2D images into stunning&nbsp;3D models with&nbsp;AI
            </p>
          </header>

          {/* Custom styled upload area */}
          <div 
            className="border-2 border-dashed border-blue-400 rounded-xl p-6 text-center cursor-pointer hover:bg-blue-400/5 transition-colors min-h-[160px] flex flex-col items-center justify-center"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files[0]) {
                handleFileSelect(e.dataTransfer.files[0]);
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.[0]) {
                  handleFileSelect(e.target.files[0]);
                }
              }}
            />
            <div className="text-lg font-bold text-white mb-2">Drag & Drop file here</div>
            <div className="text-gray-400 text-sm">or click to browse (PNG / JPG)</div>
            {file && (
              <div className="text-green-400 text-sm mt-3">
                Selected: {file.name}
              </div>
            )}
          </div>

          {/* generate */}
          <button
            onClick={handleGenerate}
            disabled={!file || generating}
            className="w-full h-12 text-base font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {generating ? 'Generating…' : 'Generate'}
          </button>
        </section>

        {/* ───────── RIGHT COL ───────── */}
        <section className="w-full lg:w-1/2 flex items-center justify-center p-4">
          <div className="w-40 h-40 lg:w-48 lg:h-48 relative">
            <Image
              src="/Robo..png"
              alt="Robot drawing"
              fill
              sizes="(min-width: 1024px) 192px, 160px"
              className="object-contain rounded-xl opacity-90 select-none pointer-events-none drop-shadow-lg"
              priority
              onError={(e) => {
                console.log('Image failed to load:', e);
              }}
            />
          </div>
        </section>
      </div>
    </main>
  );
}