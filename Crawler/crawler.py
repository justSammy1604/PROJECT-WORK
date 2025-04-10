import os
import sys
import psutil
import asyncio
import requests
from xml.etree import ElementTree
import cohere
from pathlib import Path

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

cohere_client = cohere.Client("wWwu0mw2feW52WWfvWDXVAQZV5LkuDzFB8cIsJYX")

def clean_text_with_ai(text):
    """Uses Cohere API to remove links while preserving all other content."""
    if not text or not isinstance(text, str):
        return text  # Skip non-string content

    response = cohere_client.chat(
        message=f"""
        You are an AI that cleans text by **removing all hyperlinks (URLs)** while keeping all other text unchanged.
        Ensure that the content remains fully intact except for any links.

        Here is the extracted website content:
        ```
        {text}
        ```
        
        **Return the text with only the links removed. Do not modify any other text.**
        """,
        model="command-a-03-2025",
        preamble="You are a text cleaner that removes links while preserving all other content.",
        temperature=0.1,  # Ensures consistency
    )

    return response.text  # Return cleaned text without links

async def crawl_parallel(urls: List[str], searchText, max_concurrent: int = 3):
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,   # corrected from 'verbos=False'
        extra_args=["--disable-gpu", 
                    "--disable-dev-shm-usage", 
                    "--no-sandbox",
        ]
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # We'll chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                # Unique session_id per concurrent sub-task
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Check memory usage prior to launching tasks
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory usage after tasks complete
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            # Evaluate results
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    success_count += 1
                    formatted_text = clean_text_with_ai(result.markdown_v2.raw_markdown)
                    # Save the crawled content to a file
                    file_name = f"crawled_data/{searchText.replace(' ', '_')}.txt"
                    with open(file_name, "a", encoding="utf-8") as file:
                        file.write(formatted_text)
                    print(f"Saved content to: {file_name}")
                else:
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")

def get_pydantic_ai_docs_urls():
    """
    Recursively fetch all URLs from a sitemap, including nested sitemaps.
    """
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Namespace for sitemap XML
        #namespace = {'ns': 'http://www.google.com/schemas/sitemap/0.84'}
        namespace = {'ns': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
        
        urls = []
        # Check if the sitemap contains other sitemaps
        for sitemap in root.findall('.//ns:sitemap/ns:loc', namespace):
            nested_sitemap_url = sitemap.text
            print(f"Found nested sitemap: {nested_sitemap_url}")
            urls.extend(get_all_sitemap_urls(nested_sitemap_url))  # Recursive call
        
        # Extract actual URLs if present
        for loc in root.findall('.//ns:url/ns:loc', namespace):
            urls.append(loc.text)
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {sitemap_url} - {e}")
        return []
    
    

def get_top_links(search):
    """Function to get the top links from Google search results using ScrapingDog API."""
    url = "https://api.scrapingdog.com/google"
    params = {
    #"api_key": os.getenv("SCRAP_KEY"),
    "api_key": "67f35d79575437d24b434a13",
    "query": search,
    "results": 50,
    "country": "us",
    "page": 0,
    "advance_search": "false"
}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
    
    links = []
    if "organic_results" in data:
        for result in data["organic_results"]:
            if "link" in result:
                links.append(result["link"])
                print(f"Link: {result['link']}")
    return links    

async def main():
    #urls = get_pydantic_ai_docs_urls()
    
    # urls = ["https://markets.ft.com/data",
    #         "https://edition.cnn.com/markets",
    #         "https://economictimes.indiatimes.com/stocks/marketstats/top-gainers",
    #         "https://economictimes.indiatimes.com/stocks/marketstats/top-losers",
    #         "https://economictimes.indiatimes.com/stocks/marketstats/most-active-value"
    #         ]
    # if urls:
    #     print(f"Found {len(urls)} URLs to crawl")
    #     await crawl_parallel(urls, max_concurrent=10)
    # else:
    #     print("No URLs found to crawl")    
    
    links=get_top_links("Top Gainers of the Market")
    await crawl_parallel(links, "Top Gainers of the Market")

if __name__ == "__main__":
    asyncio.run(main())