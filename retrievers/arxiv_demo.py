
import arxiv
from langchain_core.documents import Document

client = arxiv.Client(
    page_size=3,
    delay_seconds=5,
    num_retries=3
)

search = arxiv.Search(
    query="deep learning",
    max_results=3,
    sort_by=arxiv.SortCriterion.Relevance
)

papers = client.results(search)

docs = []

for paper in papers:
    doc = Document(
        page_content=paper.summary,
        metadata={
            "title": paper.title,
            "authors": ", ".join(
                author.name for author in paper.authors
            ),
            "published": str(paper.published),
            "url": paper.entry_id,
            "pdf_url": paper.pdf_url
        }
    )

    docs.append(doc)

for doc in docs:
    print("Title:", doc.metadata["title"])
    print("Content:", doc.page_content[:300])
    print("Metadata:", doc.metadata)
    print("-" * 60)
