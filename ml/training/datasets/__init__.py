"""Training-time dataset wrappers.

These read the JSONL records produced by `ml/data/ingestion/` and surface
them as torch.utils.data.Dataset instances. Datasets here MUST handle the
"too few records to train" case gracefully — early in the data flywheel
the corpus may be tiny, and we want training scripts to fail with a clear
message instead of a misleading NaN loss."""
