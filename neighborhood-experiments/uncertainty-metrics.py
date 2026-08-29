#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: DirtyBoy
@Date: 2026/8/14 12:14
"""
import numpy as np
import json
import pyarrow.parquet as pq
from tqdm import tqdm

from lm_polygraph.estimators import (
    LexicalSimilarity,
    NumSemSets,
    EigValLaplacian,
    Eccentricity,
    DegMat,
    KernelLanguageEntropy,
    LUQ,
)

from lm_polygraph.stat_calculators.semantic_matrix import (
    SemanticMatrixCalculator,
)

from lm_polygraph.utils.deberta import Deberta


# ============================================================
# 1. JSONL utilities
# ============================================================

def write_to_jsonl(data, file_path):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            line = json.dumps(
                item,
                ensure_ascii=False,
            )
            f.write(line + "\n")


def read_from_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_id} "
                    f"in {file_path}: {error}"
                ) from error

    return data


# ============================================================
# 2. Load DeBERTa NLI model
# ============================================================

print("Loading DeBERTa NLI model...")

nli_model = Deberta(
    deberta_path="microsoft/deberta-large-mnli",
    batch_size=10,
    device="cuda:0",
)

semantic_calculator = SemanticMatrixCalculator(
    nli_model=nli_model
)

# ============================================================
# 3. Initialize uncertainty estimators
# ============================================================

# Lexical similarity
lexical_estimator = LexicalSimilarity(
    metric="rougeL"
)

# Number of semantic sets
num_sem_sets_estimator = NumSemSets()

# Graph Laplacian eigenvalue
eig_estimator = EigValLaplacian(
    similarity_score="NLI_score",
    affinity="entail",
)

# Graph eccentricity
ecc_estimator = Eccentricity(
    similarity_score="NLI_score",
    affinity="entail",
)

# Degree matrix
deg_estimator = DegMat(
    similarity_score="NLI_score",
    affinity="entail",
)

# Kernel Language Entropy
kle_estimator = KernelLanguageEntropy(
    t=0.3,
    normalize=True,
    scale=True,
)

# Long-text Uncertainty Quantification
luq_estimator = LUQ()


# ============================================================
# 4. Extract seven black-box uncertainty metrics
# ============================================================

def extract_uncertainty_from_answers(answers):
    """
    Extract seven black-box uncertainty metrics from multiple
    independently generated answers.

    Parameters
    ----------
    answers : list[str]
        Multiple answers generated for the same input.

    Returns
    -------
    dict
        Seven-dimensional uncertainty representation.
    """

    if not isinstance(answers, list):
        raise TypeError(
            "answers must be a list of strings."
        )

    if not all(isinstance(answer, str) for answer in answers):
        raise TypeError(
            "Every element in answers must be a string."
        )

    # Remove empty answers
    answers = [
        answer.strip()
        for answer in answers
        if answer.strip()
    ]

    if len(answers) < 2:
        raise ValueError(
            "At least two non-empty answers are required."
        )

    # LM-Polygraph expects sample_texts to have the shape:
    # List[List[str]]
    # The outer list represents different input samples.
    # The inner list contains multiple generations for one sample.
    stats = {
        "sample_texts": [answers]
    }

    # --------------------------------------------------------
    # 4.1 Lexical similarity
    # --------------------------------------------------------

    lexical_similarity = lexical_estimator(stats)[0]

    # --------------------------------------------------------
    # 4.2 Calculate NLI matrices once
    # --------------------------------------------------------

    semantic_stats = semantic_calculator(
        dependencies=stats,
        texts=[""],
        model=None,
    )

    stats.update(semantic_stats)

    # semantic_stats contains:
    #
    # semantic_matrix_entail
    # semantic_matrix_contra
    # semantic_matrix_classes
    # semantic_matrix_entail_logits
    # semantic_matrix_contra_logits

    # --------------------------------------------------------
    # 4.3 Semantic and graph-based uncertainty metrics
    # --------------------------------------------------------

    num_sem_sets = num_sem_sets_estimator(stats)[0]

    eigval_laplacian = eig_estimator(stats)[0]

    eccentricity = ecc_estimator(stats)[0]

    degmat = deg_estimator(stats)[0]

    # --------------------------------------------------------
    # 4.4 Additional black-box metrics
    # --------------------------------------------------------

    kernel_language_entropy = kle_estimator(stats)[0]

    luq = luq_estimator(stats)[0]

    # --------------------------------------------------------
    # 4.5 Construct the seven-dimensional representation
    # --------------------------------------------------------

    result = {
        "lexical_similarity": float(
            lexical_similarity
        ),
        "num_sem_sets": float(
            num_sem_sets
        ),
        "eigval_laplacian": float(
            eigval_laplacian
        ),
        "eccentricity": float(
            eccentricity
        ),
        "degmat": float(
            degmat
        ),
        "kernel_language_entropy": float(
            kernel_language_entropy
        ),
        "luq": float(
            luq
        ),
    }

    # Check for NaN and infinity
    invalid_metrics = {
        name: value
        for name, value in result.items()
        if not np.isfinite(value)
    }

    if invalid_metrics:
        raise ValueError(
            f"Non-finite uncertainty values detected: "
            f"{invalid_metrics}"
        )

    return result


# ============================================================
# 5. Main program
# ============================================================

if __name__ == "__main__":
    import argparse
    # WikiMIA snippet length
    parser = argparse.ArgumentParser(
        description="Extract black-box uncertainty features from WikiMIA."
    )

    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=256,
        help="WikiMIA snippet length, such as 32, 64, 128, or 256.",
    )

    args = parser.parse_args()

    length = args.length
    num_runs = 10


    dataset_path = (
        f"WikiMIA_24/"
        f"WikiMIA_length{length}.parquet"
    )


    # Output feature file
    output_path = (
        f"Answer/"
        f"wiki_{length}_features_7d.jsonl"
    )

    # --------------------------------------------------------
    # 5.1 Load WikiMIA
    # --------------------------------------------------------

    print(f"Loading dataset: {dataset_path}")

    records = (
        pq.read_table(dataset_path)
        .to_pandas()
        .to_dict("records")
    )[0:1384]

    num_samples = len(records)


    print(f"Number of samples: {num_samples}")
    print(f"Number of generations per sample: {num_runs}")

    # --------------------------------------------------------
    # 5.2 Load all generated answers
    # --------------------------------------------------------

    all_answers = []

    for run_id in range(num_runs):

        answer_path = (
            f"Answer/"
            f"wiki_{length}_{run_id}.jsonl"
        )

        print(f"Loading generated answers: {answer_path}")

        run_answers = read_from_jsonl(
            answer_path
        )

        if len(run_answers) != num_samples:
            raise ValueError(
                f"Answer length mismatch in {answer_path}: "
                f"{len(run_answers)} answers versus "
                f"{num_samples} dataset samples."
            )

        # Check whether every entry contains the "answer" field
        for sample_id, item in enumerate(run_answers):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Entry {sample_id} in {answer_path} "
                    f"is not a dictionary."
                )

            if "answer" not in item:
                raise KeyError(
                    f"Entry {sample_id} in {answer_path} "
                    f"does not contain the 'answer' field."
                )

            if not isinstance(item["answer"], str):
                raise TypeError(
                    f"The 'answer' field of entry {sample_id} "
                    f"in {answer_path} is not a string."
                )

        all_answers.append(run_answers)

    print("All generated-answer files loaded successfully.")

    # --------------------------------------------------------
    # 5.3 Extract uncertainty features
    # --------------------------------------------------------

    features = []

    progress_bar = tqdm(
        enumerate(records),
        total=num_samples,
        desc="Extracting 7D uncertainty features",
    )

    for sample_id, sample in progress_bar:

        # Collect the ten independently generated answers
        # corresponding to the current WikiMIA sample.
        sample_answers = [
            run_answers[sample_id]["answer"]
            for run_answers in all_answers
        ]

        try:
            uncertainty_metrics = (
                extract_uncertainty_from_answers(
                    sample_answers
                )
            )

        except Exception as error:
            raise RuntimeError(
                f"Failed to extract uncertainty features "
                f"for sample {sample_id}: {error}"
            ) from error

        features.append({
            "uc-metrics": uncertainty_metrics,
            "label": int(sample["label"]),
        })

    # --------------------------------------------------------
    # 5.4 Save the seven-dimensional features
    # --------------------------------------------------------

    write_to_jsonl(
        features,
        output_path,
    )

    print(f"Feature extraction completed.")
    print(f"Number of processed samples: {len(features)}")
    print(f"Features saved to: {output_path}")
