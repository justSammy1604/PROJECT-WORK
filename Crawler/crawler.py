import json
import asyncio
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import textwrap

#Code to format the data in a readable format
def format_content(content):
    """Format content by ensuring proper line breaks and indentation."""
    if isinstance(content, list):  # If content is a list, format each dictionary inside it
        return [format_content(item) for item in content]

    if isinstance(content, dict):  # If content is a dictionary, format key-value pairs
        formatted_content = {}
        for key, value in content.items():
            if isinstance(value, list):  # Ensure lists are formatted nicely
                formatted_content[key] = value
            elif isinstance(value, str):
                formatted_content[key] = textwrap.wrap(value, width=80)  # Wrap long text
            else:
                formatted_content[key] = value  # Keep other types unchanged
        return formatted_content
    
    return content  # Return as-is if not a dict or list


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
    "name": "Main Content",
    "baseSelector": "body",  # Start from body
    "fields": [
        {
            "name": "main_content",
            "selector": "article, .article-body, .main-content, .content, main, #main, .post-content, .entry-content",
            "type": "text",
            "excludeSelectors": [
                "script", "style", "noscript", "iframe",
                "header", "footer", "nav",
                ".advertisement", ".ads", ".social-share",
                ".related-articles", ".sidebar", 
                ".comments", "#comments",
                ".navigation", ".pagination",
                ".breadcrumb", ".breadcrumbs",
                ".checkbox","#checkbox"
            ]
            
        },
        {
            "name": "article_title",
            "selector": "article h1, .article-title, .post-title, .entry-title",
            "type": "text"
        },
        {
            "name": "article_text",
            "selector": "article p, .article-body p, .main-content p, .post-content p",
            "type": "text",
            "multiple": True,
            "excludeSelectors": [
                ".advertisement", ".ads",
                ".author-bio", ".copyright",
                ".social-share", ".tags"
            ]
        },
        {
            "name": "article_headings",
            "selector": "article h2, article h3, .main-content h2, .main-content h3",
            "type": "text",
            "multiple": True
        },
        {
            "name": "important_lists",
            "selector": "article ul, article ol, .main-content ul, .main-content ol",
            "type": "text",
            "multiple": True,
            "excludeSelectors": [
                ".social-links", ".menu", 
                ".navigation", ".share-buttons"
            ]
        }
    ]
}

    # 2. Create the extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    # 3. Set up crawler config
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
        excluded_tags=['form', 'header', 'footer', 'nav'],
        exclude_social_media_links=True,
        exclude_external_links=True,
        word_count_threshold=5,
        exclude_external_images=True
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
        formatted_data = format_content(data)
        
        # Add metadata to the crawled content
        crawl_entry = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "content": formatted_data
        }
        
        # Append new data to existing entries
        existing_data.append(crawl_entry)
        
        # Save updated data to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f"Data successfully saved to {output_file}")

#urls of website to be crawled
urls = [
    "https://economictimes.indiatimes.com/markets",
    "https://economictimes.indiatimes.com/markets/stocks",
    "https://economictimes.indiatimes.com/markets/candlestick-screener",
    "https://economictimes.indiatimes.com/markets/stocks/stock-watch/articlelist/81776766.cms",
    "https://economictimes.indiatimes.com/markets/ipo",
    "https://economictimes.indiatimes.com/stocks/marketstats/top-gainers",
    "https://economictimes.indiatimes.com/stocks/marketstats/top-losers",
    "https://economictimes.indiatimes.com/stocks/marketstats/most-active-value"
]

async def main():
    for url in urls:
        await extract_website_content(url)

if __name__ == "__main__":
    asyncio.run(main())
