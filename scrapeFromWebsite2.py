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
import scrapy
from scrapy.crawler import CrawlerProcess
import concurrent.futures



url = "https://www.carmagazine.co.uk/"

converter = DocumentConverter()
config = dotenv_values(".env")
model_name = "deepseek-ai/DeepSeek-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# chunks = []

hybridChunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=4096,
    merge_peers=True,
)

# result = converter.convert(url)
# chunk_iter = hybridChunker.chunk(dl_doc=result.document)
# chunks.extend(list(chunk_iter))




import scrapy



links=set()
class MySpider(scrapy.Spider):
    
    name = "my_spider"
    start_urls = [url]
    
    custom_settings = {
        "DEPTH_LIMIT": 4
    }

    def parse(self, response):
        
        # Create an empty list to store URLs
        urls = set()

        # Extract all links on the page and store them in the list
        for link in response.css("a::attr(href)").getall():
            # Append the absolute URL to the list
            urls.add(response.urljoin(link))

        # Optionally handle pagination by following the next page link
        next_page = response.css("a.next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)

        # Save the URLs list after the page is processed
        global links
        links|=urls

    # def save_urls(self, urls):
    #     # Here you can save the URLs to a file, database, or just print them
    #     # For example, save to a text file:
    #     with open("urls.txt", "a") as f:
    #         for url in urls:
    #             f.write(url + "\n")
        
    #     # Alternatively, just print the URLs
    #     print(urls)


process = CrawlerProcess()

process.crawl(MySpider)

# Start the crawling process
process.start()  # This will run the spider

print(links)
print("woooohoooo: " + str(len(links)))


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


# table = db.create_table(config["table"], schema=Chunks, mode="overwrite")
tableName=config["table"]
if tableName in db:
    print(f"Table {tableName} already exists.")
    table = db[tableName]  # Retrieve the existing table object
else:
    print(f"Table {tableName} does not exist. Creating it.")
    table = db.create_table(tableName, schema=Chunks, mode="overwrite")  # Initialize the table

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------

# Create table with processed chunks



def processLinkToChunks(url): 
    try:
        result = converter.convert(url)
        if result.document:
            chunk_iter = hybridChunker.chunk(dl_doc=result.document)
            chunks=list(chunk_iter)
            
            
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
        
        table.add(processed_chunks)
            
    except:
        print("exception in url: " + url)
            
   
    
def process_in_parallel(urls, num_threads=10):
    results = []
    
    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for parallel processing
        future_to_task = {executor.submit(processLinkToChunks, url): url for url in urls}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()  # Get the result from the future
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"Error processing {task}: {exc}")
    
    return results
       
results = process_in_parallel(links, 10)
print(results)

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table.to_pandas()
table.count_rows()
