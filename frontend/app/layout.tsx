import type { Metadata } from "next";
import { Lora, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Header from "@/components/ui/Header";

const lora = Lora({
  subsets: ["latin", "cyrillic"],
  variable: "--font-lora",
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin", "cyrillic"],
  variable: "--font-jetbrains-mono",
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Science Graph — Knowledge Explorer",
  description: "Interactive knowledge graph explorer for scientific papers, books and notes with AI-powered RAG chat.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" className={`${lora.variable} ${jetbrainsMono.variable}`}>
      <body style={{ height: "100%", overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <Header />
        {children}
      </body>
    </html>
  );
}
