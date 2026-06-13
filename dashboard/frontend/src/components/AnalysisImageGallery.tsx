import { useEffect, useMemo, useState } from "react";
import { ExternalLink } from "lucide-react";

import type { VisualizationResponse } from "../types";
import { displayCategory } from "../utils/format";

export function AnalysisImageGallery({ response, loading }: { response: VisualizationResponse | null; loading: boolean }) {
  const [category, setCategory] = useState("all");
  const images = response?.images ?? [];
  const categories = useMemo(() => Array.from(new Set(images.map((image) => image.category))).filter(Boolean), [images]);
  const visibleImages = category === "all" ? images : images.filter((image) => image.category === category);

  useEffect(() => {
    setCategory("all");
  }, [response?.factor, response?.status]);

  if (loading) {
    return <div className="image-empty">Loading analysis images...</div>;
  }
  if (!response || response.status !== "ok") {
    return (
      <div className="image-empty">
        {response?.message || "No visualization artifacts for this run. Re-run analysis with visualization export enabled."}
      </div>
    );
  }
  if (images.length === 0) {
    return <div className="image-empty">No analysis images for this factor.</div>;
  }

  return (
    <section className="analysis-images">
      <div className="image-filter">
        <button type="button" className={category === "all" ? "active" : ""} onClick={() => setCategory("all")}>
          All
        </button>
        {categories.map((value) => (
          <button key={value} type="button" className={category === value ? "active" : ""} onClick={() => setCategory(value)}>
            {displayCategory(value)}
          </button>
        ))}
      </div>
      <div className="image-grid">
        {visibleImages.map((image) => (
          <article key={image.plot_id} className="analysis-image-card">
            <header>
              <div>
                <strong>{image.title}</strong>
                <small>
                  {displayCategory(image.category)}
                  {image.width && image.height ? ` - ${image.width} x ${image.height}` : ""}
                </small>
              </div>
              <a className="open-image-link" href={image.url} target="_blank" rel="noreferrer" title="Open PNG">
                <ExternalLink size={15} />
              </a>
            </header>
            <img src={image.url} alt={image.title} loading="lazy" />
          </article>
        ))}
      </div>
    </section>
  );
}
