from pipelines.counterfeit.color import (
    ColorProfileMeasurement,
    measure_color_profile,
)
from pipelines.counterfeit.embedding_anomaly import (
    EmbeddingAnomalyMeasurement,
    measure_embedding_anomaly,
)
from pipelines.counterfeit.holographic import (
    HolographicResult,
    analyze_holographic_parallax,
)
from pipelines.counterfeit.rosette import (
    DEFAULT_FREQ_BAND,
    DEFAULT_INNER_INSET_PX,
    RosetteMeasurement,
    is_likely_authentic,
    measure_rosette,
)
from pipelines.counterfeit.typography import (
    TypographyResult,
    analyze_typography,
)

__all__ = [
    # Rosette (FFT halftone) — ensemble detector #2 in the manifest order
    "DEFAULT_FREQ_BAND",
    "DEFAULT_INNER_INSET_PX",
    "RosetteMeasurement",
    "is_likely_authentic",
    "measure_rosette",
    # Color profile (CIELAB chroma + white-balance) — ensemble detector #4
    "ColorProfileMeasurement",
    "measure_color_profile",
    # Typography (OCR + Levenshtein vs identified name) — #5
    "TypographyResult",
    "analyze_typography",
    # Holographic parallax (front+tilt optical-flow ratio) — #3
    "HolographicResult",
    "analyze_holographic_parallax",
    # Embedding anomaly (cosine distance from authentic centroid) — #7
    "EmbeddingAnomalyMeasurement",
    "measure_embedding_anomaly",
]
