"use client";

import { useState } from "react";
import { toast } from "react-hot-toast";

console.log("NEXT_PUBLIC_BACKEND_ENDPOINT:", process.env.NEXT_PUBLIC_BACKEND_ENDPOINT);

export default function Home() {
  // ───────────────────────────────── state ──────────────────────────────────
  const [image, setImage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);         // simple fake bar
  const [prompt, setPrompt] = useState("a clean 3-D asset");

  // ───────────────────────────── handle upload ──────────────────────────────
  const handleSketch = async (b64: string) => {
    setBusy(true);
    setImage(null);
    setProgress(0);

    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sketch: b64,
          prompt: prompt || "a clean 3-D asset"
        })
      });

      if (!r.ok) throw new Error(`HTTP ${r.status}`);

      const j = await r.json();
      if (j.error) throw new Error(j.error);

      // Handle the image response
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      setImage(url);
      setProgress(100);
      toast.success("3-D image generated!");
    } catch (err: any) {
      toast.error(err?.message ?? "Generation failed");
    } finally {
      setBusy(false);
    }
  };

  // ─────────────────────────── download helper ──────────────────────────────
  const downloadImage = () => {
    if (!image) return;
    const link = document.createElement("a");
    link.href = image;
    link.download = "3d_image.png";
    link.click();
  };

  // ─────────────────────────────────  UI  ───────────────────────────────────
  return (
    <main className="max-w-3xl mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-6">Sketch → 3-D Image Generator</h1>

      {/* prompt input */}
      <div className="mb-6">
        <label htmlFor="prompt" className="block text-sm font-medium mb-1">
          Generation prompt
        </label>
        <input
          id="prompt"
          type="text"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          disabled={busy}
          className="w-full rounded-md border px-3 py-2"
          placeholder="Describe the style, material…"
        />
      </div>

      {/* drag-and-drop */}
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition ${busy ? "opacity-50" : ""}`}
        onDragOver={e => { e.preventDefault(); }}
        onDrop={async e => {
          e.preventDefault();
          if (e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
              if (e.target?.result) {
                handleSketch(e.target.result as string);
              }
            };
            reader.readAsDataURL(file);
          }
        }}
        onClick={() => {
          if (!busy) {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/png';
            input.onchange = (e) => {
              const file = (e.target as HTMLInputElement).files?.[0];
              if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                  if (e.target?.result) {
                    handleSketch(e.target.result as string);
                  }
                };
                reader.readAsDataURL(file);
              }
            };
            input.click();
          }
        }}
        style={{ cursor: busy ? "not-allowed" : "pointer" }}
      >
        <div className="text-center">
          <div className="text-4xl mb-4">📷</div>
          <div className="text-lg font-medium mb-2">Drop a PNG sketch here</div>
          <div className="text-gray-500">(or click to select)</div>
          <div className="text-sm text-gray-400 mt-2">
            Upload a sketch to generate a 3D image
          </div>
        </div>
      </div>

      {/* prog bar */}
      {busy && (
        <div className="mt-4">
          <div className="w-full bg-gray-200 h-2 rounded-full overflow-hidden">
            <div
              className="bg-blue-600 h-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-2 text-blue-500 animate-pulse">
            Generating 3D image… {progress}%
          </p>
        </div>
      )}

      {/* results */}
      {image && (
        <div className="mt-6">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
            <h3 className="text-lg font-medium text-green-800 mb-2">✅ Generation Complete!</h3>
            <p className="text-green-700">
              Your 3D image has been generated successfully.
            </p>
          </div>
          
          {/* Display the generated image */}
          <div className="bg-white border border-gray-200 rounded-lg p-4 mb-4">
            <img 
              src={image} 
              alt="Generated 3D image" 
              className="w-full h-auto rounded-lg shadow-lg"
            />
          </div>
          
          <button
            onClick={downloadImage}
            className="rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600 transition-colors"
          >
            📥 Download 3D Image
          </button>
        </div>
      )}
    </main>
  );
}
