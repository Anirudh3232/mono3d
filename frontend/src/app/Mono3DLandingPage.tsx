"use client";
import React, { useState, useCallback } from "react";
import "./Mono3DLandingPage.css";

export default function Mono3DLandingPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [meshUrl, setMeshUrl] = useState("");
  const [imgUrl, setImgUrl] = useState("");

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  const browseFile = () => document.getElementById("m3d-file")?.click();
  const onBrowse = (e: React.ChangeEvent<HTMLInputElement>) => setFile(e.target.files?.[0] || null);

  const handleGenerate = async () => {
    if (!file) return alert("Please drop or select an image first.");
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);

      const res = await fetch("/generate", { method: "POST", body: fd });
      if (!res.ok) throw new Error("Generation failed");

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setImgUrl(url);

      // optionally request /mesh endpoint for OBJ
      const meshRes = await fetch("/mesh");
      if (meshRes.ok) {
        const meshBlob = await meshRes.blob();
        setMeshUrl(URL.createObjectURL(meshBlob));
      }
    } catch (err: any) {
      console.error(err);
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!meshUrl) return;
    const a = document.createElement("a");
    a.href = meshUrl;
    a.download = "mono3d_output.obj";
    a.click();
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--gray)" }}>
      <div style={{ display: "flex", background: "#fff", borderRadius: 16, boxShadow: "0 4px 18px rgba(0,0,0,.06)", maxWidth: 900, width: "100%", minHeight: 420 }}>
        {/* Left column */}
        <div style={{ flex: 1, padding: "2.5rem 2rem", display: "flex", flexDirection: "column", justifyContent: "center" }}>
          <h1 className="m3d-title">2D → 3D Converter</h1>
          <div
            className="m3d-upload-zone"
            onDragEnter={(e)=>e.preventDefault()}
            onDragOver ={(e)=>e.preventDefault()}
            onDragLeave={(e)=>e.preventDefault()}
            onDrop      ={onDrop}
            onClick={browseFile}
          >
            {file
              ? <strong>{file.name}</strong>
              : <>
                  <strong>Drag &amp; Drop file here</strong>
                  <p>or click to browse (PNG / JPG)</p>
                </>
            }
          </div>
          <input id="m3d-file" type="file" accept="image/*"
                 style={{display:"none"}} onChange={onBrowse} />
          <button
            className="m3d-generate-btn"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? "Generating…" : "Generate"}
          </button>
          {imgUrl && (
            <div style={{ marginTop: 16 }}>
              <div className="m3d-output-label">Output Preview</div>
              <div id="m3d-viewer">
                <img src={imgUrl} alt="Generated concept" className="m3d-output-img" />
              </div>
              <button
                id="m3d-download-btn"
                onClick={handleDownload}
                disabled={!meshUrl}
              >
                Download OBJ
              </button>
            </div>
          )}
        </div>
        {/* Right column */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "2.5rem 2rem" }}>
          <img src="/robo.png" alt="Robot sketch to 3D example" className="m3d-right-img" />
        </div>
      </div>
    </div>
  );
} 