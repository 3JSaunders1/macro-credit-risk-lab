"""
services/pipeline_service.py
----------------------------
Single entry point for pipeline execution. Run metadata is saved only on request.
"""
from pipeline.run_pipeline import run_pipeline
from utils.run_metadata import save_run_metadata


def run_fresh_pipeline(save_metadata: bool = False, **kwargs):
    result = run_pipeline(**kwargs)
    if save_metadata:
        save_run_metadata(result)
    return result
