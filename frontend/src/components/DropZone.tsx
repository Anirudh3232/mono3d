"use client";
import { useRef, useState } from "react";
import { fileToBase64 } from "@/lib/toBase64";

export default function DropZone({ onReady }:{ onReady:(b64:string)=>void }) {
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (file) {
      onReady(await fileToBase64(file));
    }
  };

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-8 text-center transition ${hover ? "bg-gray-100" : ""}`}
      onDragOver={e => { e.preventDefault(); setHover(true); }}
      onDragLeave={() => setHover(false)}
      onDrop={async e => {
        e.preventDefault(); setHover(false);
        if (e.dataTransfer.files[0]) {
          handleFile(e.dataTransfer.files[0]);
        }
      }}
      onClick={() => inputRef.current?.click()}
      style={{ cursor: "pointer" }}
    >
      <input
        type="file"
        accept="image/png"
        style={{ display: "none" }}
        ref={inputRef}
        onChange={e => {
          if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
          }
        }}
      />
      Drop a PNG sketch here<br />
      <span style={{ color: "#888", fontSize: "0.9em" }}>(or click to select)</span>
    </div>
  );
}
