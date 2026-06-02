# from __future__ import annotations

# from pathlib import Path

# from tax_talk.evals.dataset import load_golden_dataset


# def test_load_golden_dataset_reads_samples() -> None:
#     dataset_path = Path("data/eval/golden_qa_v0_30.jsonl")
#     samples = load_golden_dataset(dataset_path)

#     assert len(samples) == 30
#     assert samples[0].id
#     assert samples[0].question
#     assert samples[0].expected_answer
