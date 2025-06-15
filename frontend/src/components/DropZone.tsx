"use client";
import { useState } from "react";
import { fileToBase64 } from "@/lib/toBase64";

export default function DropZone({ onReady }:{ onReady:(b64:string)=>void }) {
  const [hover,setHover] = useState(false);

  return (
    <div

      className={`border-2 border-dashed rounded-xl p-8 text-center transition ${
        hover ? "bg-gray-100" : ""
      }`}
      onDragOver={e => { e.preventDefault(); setHover(true); }}
      onDragLeave={() => setHover(false)}
      onDrop={async e => {
        e.preventDefault(); setHover(false);
        if (e.dataTransfer.files[0]) {
          onReady(await fileToBase64(e.dataTransfer.files[0]));
        }
      }}>
      Drop a PNG sketch here
    </div>
  );
}
