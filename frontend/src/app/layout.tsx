import "./globals.css";
import { Toaster } from "react-hot-toast";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Mono3D",
  description: "Sketch → 3-D generator",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet" />
      </head>
      <body className="bg-[#0a0a0f] font-sans text-white min-h-screen">
        {children}
        <Toaster position="top-right" />
      </body>
    </html>
  );
}
