import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents():
    """Three fake documents for chunker tests."""
    return [
        Document(
            page_content="This is page one of the contract. " * 50,
            metadata={"source": "contract.pdf", "page": 0},
        ),
        Document(
            page_content="Payment terms are net 30 days. " * 50,
            metadata={"source": "contract.pdf", "page": 1},
        ),
        Document(
            page_content="The invoice total is $5,000. " * 50,
            metadata={"source": "invoice.pdf", "page": 0},
        ),
    ]


@pytest.fixture
def short_document():
    """A document shorter than the default chunk size."""
    return Document(
        page_content="Short text under chunk size.",
        metadata={"source": "memo.pdf", "page": 0},
    )
