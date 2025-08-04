import React from "react";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      <h1 className="text-4xl font-bold mb-6">Welcome to AI Task Manager</h1>
      <Link href="/demo">
        <button className="px-6 py-3 rounded-lg bg-white text-gray-900 font-semibold shadow hover:bg-gray-200 transition">View Liquid Glass Button Demo</button>
      </Link>
    </div>
  );
} 