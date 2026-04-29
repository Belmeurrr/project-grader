"""Evaluation harnesses for the grading + counterfeit-detector pipelines.

Sub-packages here are read-only consumers of the production code in
ml/pipelines/. They produce reports (JSON, Markdown, console) but
have no persistence side effects. Each sub-package is invokable as a
module: `python -m evaluation.<name>`."""
