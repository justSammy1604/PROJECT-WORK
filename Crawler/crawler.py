import json
import asyncio
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

async def extract_website_content(url):
    # Create output directory if it doesn't exist
    output_dir = Path("crawled_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create or load existing data file
    output_file = output_dir / "crawled_content.json"
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []  # Initialize with empty list if file doesn't exist or is empty

    # schema code to extract the text from the website
    schema = {
    "name": "Complete Website Content",
    "baseSelector": "body",
    "fields": [
        {
            "name": "meta_description",
            "selector": "meta[name='description']",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "title",
            "selector": "h1, .article-heading, .page-title",
            "type": "text",
            "multiple": True
        },
        {
            "name": "all_text_content",
            "selector": "body",
            "type": "text",
            "excludeSelectors": ["script", "style", "noscript", "iframe"]
        },
        {
            "name": "main_content",
            "selector": "article, .article-body, .main-content, .content, main, #main, .post-content",
            "type": "text"
        },
        {
            "name": "paragraphs",
            "selector": "p, .paragraph, article p, main p",
            "type": "text",
            "multiple": True
        },
        {
            "name": "headings",
            "selector": "h1, h2, h3, h4, h5, h6",
            "type": "text",
            "multiple": True
        },
        {
            "name": "lists",
            "selector": "ul li, ol li, dl dt, dl dd",
            "type": "text",
            "multiple": True
        },
        {
            "name": "tables",
            "selector": "table",
            "type": "html",
            "multiple": True
        },
        {
            "name": "links",
            "selector": "a",
            "type": "text",
            "multiple": True
        },
        {
            "name": "images_alt_text",
            "selector": "img",
            "type": "attribute",
            "attribute": "alt",
            "multiple": True
        },
        {
            "name": "blockquotes",
            "selector": "blockquote, q, cite",
            "type": "text",
            "multiple": True
        },
        {
            "name": "sidebar_content",
            "selector": "aside, .sidebar, .widget",
            "type": "text",
            "multiple": True
        }
    ]
}

    # 2. Create the extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    # 3. Set up crawler config
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
    )

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            config=config
        )

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        # 5. Parse the extracted content
        data = json.loads(result.extracted_content)
        
        # Add metadata to the crawled content
        crawl_entry = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "content": data
        }
        
        # Append new data to existing entries
        existing_data.append(crawl_entry)
        
        # Save updated data to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f"Data successfully saved to {output_file}")

# Example usage with multiple URLs
urls = [
    "https://www.investopedia.com/terms/i/investment.asp",
    "https://en.wikipedia.org/wiki/Market_economy",
    "https://www.moneycontrol.com/stocksmarketsindia/",
    "https://coinmarketcap.com/",
    # Add more URLs as needed
]

async def main():
    for url in urls:
        await extract_website_content(url)

if __name__ == "__main__":
    asyncio.run(main())
