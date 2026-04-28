"""Stage 3 — card identification.

Match a dewarped canonical card image to a known card variant in the
catalog. Pipeline:

  1. Compute pHash + dHash (deterministic, fast).
  2. Pre-filter the catalog by Hamming distance on pHash. ~80% of
     identifications resolve here on a clean shot.
  3. For the remainder, compute the deep embedding (DINOv2-ViT-B by
     default; SimpleEmbedder fallback in dev/test).
  4. Find top-k nearest by cosine distance in pgvector HNSW.
  5. Combine pHash + embedding candidates into a ranked result with a
     calibrated confidence.

Returns IdentificationResult with up to top_k candidates and a chosen
best match (None if confidence is too low — the API surfaces that as
"unidentified, please confirm").
"""

from pipelines.identification.catalog import (
    CardCatalogEntry,
    CatalogHit,
    CatalogIndex,
    InMemoryCatalogIndex,
)
from pipelines.identification.embedding import (
    EMBEDDING_DIM,
    ImageEmbedder,
    SimpleEmbedder,
    get_embedder,
)
from pipelines.identification.hashing import (
    HASH_BITS,
    compute_dhash,
    compute_phash,
    hamming_distance,
    hash_to_hex,
    hex_to_hash,
)
from pipelines.identification.identify import (
    IdentificationCandidate,
    IdentificationResult,
    identify,
)

__all__ = [
    "CardCatalogEntry",
    "CatalogHit",
    "CatalogIndex",
    "EMBEDDING_DIM",
    "HASH_BITS",
    "IdentificationCandidate",
    "IdentificationResult",
    "ImageEmbedder",
    "InMemoryCatalogIndex",
    "SimpleEmbedder",
    "compute_dhash",
    "compute_phash",
    "get_embedder",
    "hamming_distance",
    "hash_to_hex",
    "hex_to_hash",
    "identify",
]
