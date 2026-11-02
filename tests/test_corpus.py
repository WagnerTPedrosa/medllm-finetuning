from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from src.config import PretrainConfig, load_pretrain_config
from src.corpus import (
    is_boilerplate_heading,
    pack_blocks,
    remove_boilerplate_sections,
    remove_citations,
    split_by_stratum,
    tokenize_documents,
)

PRETRAIN_CONFIG_PATH = Path("configs/gpt2_pubmed_pretrain.yaml").resolve()


class FakeTokenizer:
    """One token per word, EOS = 0."""

    eos_token_id = 0

    def __call__(self, texts: list[str], add_special_tokens: bool) -> dict[str, list[list[int]]]:
        return {"input_ids": [[len(word) for word in text.split()] for text in texts]}


def test_load_pretrain_config_reads_corpus_section() -> None:
    config: PretrainConfig = load_pretrain_config(PRETRAIN_CONFIG_PATH)

    assert config.model.model_id == "openai-community/gpt2"
    assert config.corpus.validation_fraction == 0.01
    assert config.corpus.remove_boilerplate_sections is True
    assert config.mlflow.registered_model_name == "gpt2-pubmed-dapt"


@pytest.mark.parametrize(
    "heading",
    [
        "Author contributions",
        "5. Funding",
        "Funding/support",
        "Role of the funding source",
        "Supplemental Material",
        "Statement of Ethics",
        "Ethics and dissemination",
        "Transparency declaration",
        "Declaration of Competing Interest:",
        "Ethics approval and consent to participate",
        "Supplementary Information",
        "CRediT authorship contribution statement",
        "Data Availability Statement",
        "Abbreviations",
    ],
)
def test_is_boilerplate_heading_matches_administrative_sections(heading: str) -> None:
    assert is_boilerplate_heading(heading)


@pytest.mark.parametrize(
    "heading",
    [
        "Results",
        "Ethical issues in end-of-life care",
        "Funding gaps in rural hospitals",
        "Methods",
        "Implications of all the available evidence",
        "Who will take informed consent?",
    ],
)
def test_is_boilerplate_heading_keeps_scientific_sections(heading: str) -> None:
    assert not is_boilerplate_heading(heading)


def test_remove_boilerplate_sections_drops_section_until_next_heading() -> None:
    text = (
        "Title\n\nAbstract text.\n\n## Results\n\nWe found X.\n\n"
        "## Funding\n\nGrant 123.\n\nMore funding text.\n\n## Discussion\n\nX matters.\n\n"
        "## Author contributions\n\nAB wrote it."
    )

    cleaned = remove_boilerplate_sections(text)

    assert cleaned == (
        "Title\n\nAbstract text.\n\n## Results\n\nWe found X.\n\n## Discussion\n\nX matters."
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("as shown [1, 2].", "as shown."),
        ("risk factors [3–5] were", "risk factors were"),
        ("was reported (Smith et al., 2020).", "was reported."),
        ("in two cohorts (Smith and Jones, 2019; Lee, 2021).", "in two cohorts."),
    ],
)
def test_remove_citations_drops_citation_markers(raw: str, expected: str) -> None:
    assert remove_citations(raw) == expected


@pytest.mark.parametrize(
    "text",
    ["odds ratio [95% CI 1.2–3.4]", "intracellular [Ca2+] rose", "sample (n = 20) in 2020", "the (2020) guideline"],
)
def test_remove_citations_keeps_other_brackets(text: str) -> None:
    assert remove_citations(text) == text


def test_split_by_stratum_keeps_proportions_and_is_disjoint() -> None:
    dataset = Dataset.from_dict(
        {"pmcid": [f"PMC{i}" for i in range(200)], "stratum": ["a"] * 100 + ["b"] * 100}
    )

    splits: DatasetDict = split_by_stratum(dataset, validation_fraction=0.1, seed=1)

    assert len(splits["validation"]) == 20
    assert splits["validation"]["stratum"].count("a") == 10
    assert set(splits["train"]["pmcid"]).isdisjoint(splits["validation"]["pmcid"])
    assert split_by_stratum(dataset, 0.1, 1)["validation"]["pmcid"] == splits["validation"]["pmcid"]


def test_tokenize_documents_appends_eos_to_each_document() -> None:
    batch = {"text": ["aa b", "ccc"]}

    tokens = tokenize_documents(batch, FakeTokenizer())  # type: ignore[arg-type]

    assert tokens == {"input_ids": [[2, 1, 0], [3, 0]]}


def test_pack_blocks_concatenates_and_drops_the_remainder() -> None:
    batch = {"input_ids": [[1, 2, 3], [4, 5]]}

    assert pack_blocks(batch, block_size=2) == {"input_ids": [[1, 2], [3, 4]]}
