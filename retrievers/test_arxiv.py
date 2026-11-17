import arxiv
import logging

logging.basicConfig(level=logging.INFO)

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

for paper in client.results(search):
    print(paper.title)
    print(paper.entry_id)
    print("-" * 50)