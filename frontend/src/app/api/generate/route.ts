import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const r = await fetch(
    process.env.NEXT_PUBLIC_BACKEND_ENDPOINT + "/generate",
    {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify(body)
    }
  );
  const text = await r.text();
  return new NextResponse(text, { status: r.status });
}
