import LibraryPage from "@/components/ui/LibraryPage";

async function getInitialLibraryData() {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const url = new URL(`${backendUrl}/api/documents`);
    url.searchParams.set("page", "1");
    url.searchParams.set("limit", "100");
    ["paper", "note", "book", "video", "webpage"].forEach((t) =>
      url.searchParams.append("source_type", t)
    );

    const res = await fetch(url.toString(), { cache: "no-store" });
    if (!res.ok) {
      console.error(`Failed to fetch initial library data: ${res.status}`);
      return null;
    }
    return await res.json();
  } catch (error) {
    console.error("Error fetching initial library data:", error);
    return null;
  }
}

export default async function LibraryRoutePage() {
  const initialData = await getInitialLibraryData();
  return <LibraryPage initialData={initialData} />;
}
