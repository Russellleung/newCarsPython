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


# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter


converter = DocumentConverter()

# from docling import Document


config = dotenv_values(".env")

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=config["client_id"],
    client_secret=config["client_secret"],
    user_agent=config["user_agent"],
)

# Choose a subreddit (e.g., sports cars discussions)
subreddit = reddit.subreddit(
    config["subreddit"]
)  # or "exoticcars", "whatcarshouldIbuy"


model_name = "deepseek-ai/DeepSeek-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# chunker = WordChunker(
#     tokenizer="NousResearch/Nous-Capybara-3B-V1.9",  # Supports string identifiers
#     chunk_size=4096,  # Maximum tokens per chunk
#     chunk_overlap=0,  # Overlap between chunks
#     return_type="chunks",  # Return type of the chunker; "chunks" or "texts"
# )

hybridChunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=4096,
    merge_peers=True,
)

# Fetch top posts
# for post in subreddit.search("sports car", limit=1):
#     print(f"Title: {post.title}")

#     if post.is_self:  # This is a self post (text post)
#         print(f"Body text: {post.selftext}")
#     else:  # This is a link post
#         print(f"Link: {post.url}")  # Scrape the URL for link posts
#         scrapelink.scrapeLink(post.url)

#     print(f"Upvotes: {post.score}")
#     print(f"Comments: {post.num_comments}\n")

#     # Fetch comments
#     post.comments.replace_more(limit=0)  # To load all comments
#     for comment in post.comments[:5]:  # Get first 5 comments
#         print(f"- {comment.body}\n")

#     # Apply HybridChunker
#     chunk_iter = chunker.chunk(post)

#     # Print the chunks
#     for i, chunk in enumerate(chunk_iter):
#         print(f"Chunk {i+1}: {chunk}")

converter = DocumentConverter()
chunks = []
for post in subreddit.hot(limit=10):
    print(f"Title: {post.title}")
    print(f"permalink: {post.permalink}")

    post_text = f"Title: {post.title}\n\n"

    if post.is_self:  # Self post (text post)
        post_text += f"Body: {post.selftext}\n\n"
        print(f"Body: {post.selftext}")
    else:  # Link post
        # post_text += f"Link: {post.url}\n\n"
        # print(f"Link: {post.url}")
        # Add logic to scrape the link if needed
        try:
            result = converter.convert(post.url)
            chunk_iter = hybridChunker.chunk(dl_doc=result.document)
            chunks.extend(list(chunk_iter))
        except:
            print("pass")
        # print(type(chunks[0]))
        # print(chunks[0])
        # for i, chunk in enumerate(chunks):
        #     print(f"Chunk {i+1}: {chunk}")

    print(f"Upvotes: {post.score}")
    print(f"Comments: {post.num_comments}\n")
    post_text += f"Upvotes: {post.score}\nComments: {post.num_comments}\n\n"

    # Fetch comments
    post.comments.replace_more(limit=0)  # Load all comments
    comments_text = ""
    for i, comment in enumerate(post.comments[:5]):  # Get first 5 comments
        comments_text += f"Comment {i+1}: {comment.body}\n\n"
        print(f"- {comment.body}\n")

    post_text += comments_text  # Add comments to post text

    # Create a PDF document
    pdf = SimpleDocTemplate("output.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    style = styles["Normal"]  # Use default style
    paragraph = Paragraph(post_text, style)

    # Save the PDF
    pdf.build([paragraph])

    result = converter.convert("output.pdf")

    # Apply HybridChunker
    chunk_iter = hybridChunker.chunk(dl_doc=result.document)
    chunks.extend(list(chunk_iter))

    # Print the chunks
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunk}")


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
