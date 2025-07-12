"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import DropZone from "@/components/DropZone";
import { toast } from "react-hot-toast";

const ModelPreview = dynamic(
  () => import("@/components/ModelPreview"),
  { ssr: false }
    // don't server-render Three.js
);

console.log("NEXT_PUBLIC_BACKEND_ENDPOINT:", process.env.NEXT_PUBLIC_BACKEND_ENDPOINT);

export default function Home() {
  // ───────────────────────────────── state ──────────────────────────────────
  const [mesh,     setMesh]     = useState<string | null>(null);
  const [busy,     setBusy]     = useState(false);
  const [progress, setProgress] = useState(0);         // simple fake bar
  const [prompt,   setPrompt]   = useState("a clean 3-D asset");

  // ───────────────────────────── handle upload ──────────────────────────────
  const handleSketch = async (b64: string) => {
    setBusy(true);
    setMesh(null);
    setProgress(0);

    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sketch:  b64,
          prompt:  prompt || "a clean 3-D asset",
          preview: true
        })
      });

      if (!r.ok) throw new Error(`HTTP ${r.status}`);

      const j = await r.json();
      if (j.error) throw new Error(j.error);

      setMesh(j.mesh);
      setProgress(100);
      toast.success("3-D model generated!");
    } catch (err: any) {
      toast.error(err?.message ?? "Generation failed");
    } finally {
      setBusy(false);
    }
  };

  // ─────────────────────────── download helper ──────────────────────────────
  const downloadOBJ = () => {
    if (!mesh) return;
    const link = document.createElement("a");
    link.href = `data:model/obj;base64,${mesh}`;
    link.download = "mono3d.obj";
    link.click();
  };

  // ─────────────────────────────────  UI  ───────────────────────────────────
  return (
    <main className="max-w-3xl mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-6">Sketch → 3-D Generator</h1>

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
      <DropZone onReady={handleSketch} />

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
            Generating… {progress}%
          </p>
        </div>
      )}

      {/* viewer + download */}
      {mesh && (
        <div className="mt-6">
          <ModelPreview objB64={mesh} />
          <button
            onClick={downloadOBJ}
            className="mt-4 rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600"
          >
            Download OBJ
          </button>
        </div>
      )}
    </main>
  );
}
