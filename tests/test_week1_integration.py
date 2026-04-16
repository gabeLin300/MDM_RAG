import shutil
from pathlib import Path

from pipeline.run_pipeline import run_week1_pipeline


def test_sample_profile_end_to_end():
    output_dir = Path("data/processed/test_sample_profile")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    result = run_week1_pipeline(
        profile="sample",
        input_path="data/raw/100_sample_advanced_rag.csv",
        output_dir=output_dir,
    )
    assert result["ingestion_report"]["rows_parsed"] > 0
    assert result["chunks_created"] > 0
    assert result["index_size"] > 0
    assert len(result["query_smoke"]["query_results"]) >= 1
    first = result["query_smoke"]["query_results"][0]
    assert "citations" in first
