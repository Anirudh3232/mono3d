"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface DropZoneProps {
  onFileChange: (file: File, dataUrl: string) => void;
}

const DropZone: React.FC<DropZoneProps> = ({ onFileChange }) => {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        const reader = new FileReader();
        reader.onload = () => {
          const dataUrl = reader.result as string;
          setPreview(dataUrl);
          onFileChange(file, dataUrl);
        };
        reader.readAsDataURL(file);
      }
    },
    [onFileChange]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"] },
    multiple: false,
  });

  const activeStyle = "bg-blue-50 border-blue-600";
  const baseStyle = "border-gray-400";

  return (
    <div
      {...getRootProps()}
      className={`upload-zone border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors duration-300 ease-in-out ${
        isDragActive ? activeStyle : baseStyle
      } hover:border-blue-600`}
    >
      <input {...getInputProps()} />
      {preview ? (
        <div className="relative w-full h-48">
          <img src={preview} alt="Sketch preview" className="w-full h-full object-contain rounded-md" />
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center text-white opacity-0 hover:opacity-100 transition-opacity">
            Click or drag to replace image
          </div>
        </div>
      ) : (
        <div>
          <strong className="text-gray-800">Drag & Drop file here</strong>
          <p className="text-gray-500 mt-2">or click to browse (PNG / JPG)</p>
        </div>
      )}
    </div>
  );
};

export default DropZone;
