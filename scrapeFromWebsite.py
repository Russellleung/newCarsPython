import praw
from dotenv import dotenv_values
import scrapelink
import lancedb
from docling.chunking import HybridChunker
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from typing import Optional, List
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter
from sitemap import get_sitemap_urls
from scrapelink import scrapeLinks

url = "https://www.carmagazine.co.uk/car-reviews/byd/sealion-7-suv/"

converter = DocumentConverter()
config = dotenv_values(".env")
model_name = "deepseek-ai/DeepSeek-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

chunks = []

hybridChunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=4096,
    merge_peers=True,
)

# result = converter.convert(url)
# chunk_iter = hybridChunker.chunk(dl_doc=result.document)
# chunks.extend(list(chunk_iter))


urls = scrapeLinks(url)
print(urls)

docs = []
for url in urls:
    try:
        result = converter.convert(url)
        if result.document:
            chunk_iter = hybridChunker.chunk(dl_doc=result.document)
            chunks.extend(list(chunk_iter))
    except:
        print("exception in url: " + url)

db = lancedb.connect("data/lancedb")

# Get the OpenAI embedding function
func = (
    get_registry()
    .get("sentence-transformers")
    .create(name="BAAI/bge-small-en-v1.5", device="cpu")
)


# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: Optional[str]
    page_numbers: Optional[List[int]]
    title: Optional[str]


# Define the main Schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


table = db.create_table(config["table"], schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------

# Create table with processed chunks
processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ]
            or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------

table.add(processed_chunks)

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table.to_pandas()
table.count_rows()
