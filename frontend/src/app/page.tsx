"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Toaster, toast } from "react-hot-toast";
import JSZip from "jszip";

import DropZone from "@/components/DropZone";

const ModelPreview = dynamic(() => import("@/components/ModelPreview"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex justify-center items-center text-gray-500">
      Loading 3D Viewer...
    </div>
  ),
});

type GenerationState = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [generationState, setGenerationState] = useState<GenerationState>("idle");
  const [prompt, setPrompt] = useState("");
  const [sketch, setSketch] = useState<string | null>(null);
  const [mesh, setMesh] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isBusy = generationState === "loading";

  const handleSubmit = async () => {
    if (!sketch) {
      toast.error("Please upload a sketch first.");
      return;
    }

    setGenerationState("loading");
    setError(null);
    setMesh(null);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sketch, prompt: prompt || "a high-quality 3d asset" }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Generation failed with an unknown error.");
      }

      const zipBlob = await response.blob();
      const zip = await JSZip.loadAsync(zipBlob);
      const objFile = zip.file("model.obj");

      if (!objFile) {
        throw new Error("Could not find 'model.obj' in the returned zip file.");
      }

      const objData = await objFile.async("string");
      setMesh(objData);
      setGenerationState("success");
      toast.success("Your 3D model is ready!");

    } catch (err: any) {
      const errorMessage = err.message || "An unexpected error occurred.";
      setError(errorMessage);
      setGenerationState("error");
      toast.error(errorMessage);
    }
  };

  return (
    <>
      <Toaster position="top-center" />
      <div className="bg-gray-100 min-h-screen text-gray-900 font-sans">
        <header className="bg-white py-24 px-4 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">From 2D Vision to 3D Reality</h1>
          <p className="text-lg text-gray-700 max-w-2xl mx-auto">
            Upload a sketch or image and let Mono3D craft a ready-to-use 3D mesh (.obj).
          </p>
          {/* Placeholder for hero image if needed later */}
        </header>

        {/* 🚀 Tailwind verification banner */}
        <div className="bg-sky-500 text-white p-4 rounded-xl text-center mx-4 my-6">
          Tailwind v4 + Next 14 working!
        </div>

        <main className="flex flex-col items-center gap-12 py-12 px-4">
          <section className="bg-white p-8 rounded-xl w-full max-w-3xl shadow-lg">
            <h2 className="text-2xl font-semibold mb-5 text-center">1 · Upload</h2>
            <DropZone onFileChange={(_, dataUrl) => setSketch(dataUrl)} />
            <div className="mt-6 flex flex-col sm:flex-row gap-4">
              <input
                id="prompt"
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={isBusy}
                className="flex-1 px-4 py-3 border border-gray-300 rounded-xl w-full focus:ring-2 focus:ring-blue-500 transition"
                placeholder="Optional prompt (e.g. ‘cel-shade robot’)"
              />
              <button
                id="generate-btn"
                onClick={handleSubmit}
                disabled={isBusy || !sketch}
                className="bg-blue-600 text-white px-6 py-3 rounded-xl cursor-pointer hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
              >
                {isBusy ? "Generating..." : "Generate"}
              </button>
            </div>
          </section>

          {/* This section will be populated on generation */}
          {(generationState !== 'idle' || mesh) && (
             <section className="bg-white p-8 rounded-xl w-full max-w-3xl shadow-lg">
                <h2 className="text-2xl font-semibold mb-5 text-center">2 · Preview & Download</h2>
                <div id="viewer" className="w-full h-[400px] bg-black rounded-xl flex justify-center items-center">
                    {isBusy && <div className="text-white">Processing...</div>}
                    {generationState === "error" && <div className="text-red-500 p-4">{error}</div>}
                    {generationState === "success" && mesh && <ModelPreview objData={mesh} />}
                </div>
                {generationState === "success" && mesh && (
                    <div className="flex justify-center">
                        <a
                        href={`data:model/obj;charset=utf-8,${encodeURIComponent(mesh)}`}
                        download="mono3d_model.obj"
                        id="download-btn"
                        className="mt-5 inline-block bg-blue-600 text-white px-8 py-3 rounded-xl cursor-pointer hover:bg-blue-700 transition"
                        >
                        Download OBJ
                        </a>
                    </div>
                )}
             </section>
          )}

        </main>

        <footer className="text-sm py-8 text-gray-600 text-center">
          Mono3D · <a href="https://github.com/Anirudh3232/mono3d" target="_blank" className="underline hover:text-blue-600">GitHub</a>
        </footer>
      </div>
    </>
  );
}
