import json
import asyncio
import cohere  # Cohere API instead of OpenAI
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import textwrap

# Cohere API Key (replace with your actual API key)
cohere_client = cohere.Client("wWwu0mw2feW52WWfvWDXVAQZV5LkuDzFB8cIsJYX")

# Function to send content to AI for cleaning
def clean_text_with_ai(text):
    """Uses Cohere API to clean unwanted text intelligently."""
    if not text or not isinstance(text, str):
        return text  # Skip non-string content
    
    # Using Cohere's chat endpoint
    # response = cohere_client.chat(
    #     message=f"""
    #     Clean this website content by removing navigation menus, advertisements, social media links, and non-informational text.
    #     Keep only the main article, financial data, and useful market insights.
    #     Here is the extracted content:
    #     {text}
    #     """,
    #     model="command",  # Using Cohere's command model
    #     preamble="You are a financial data cleaner.",
    #     temperature=0.2  # Lower temperature for consistency
    # )
    response = cohere_client.chat(
        message=f"""
        You are an AI that extracts **only the most important financial data** from a webpage.
        Remove **navigation menus, advertisements, social media links, copyright notices, author bios,website language options and other irrelevant text**.
        Keep **only the core financial insights** like:
        
        - **Stock Market Movements** (Top Gainers, Top Losers, Active Stocks)
        - **Company Financials** (Stock prices, market trends, earnings reports)
        - **Macroeconomic Data** (Interest rates, inflation, GDP growth)
        - **Cryptocurrency Data** (Bitcoin, Ethereum, etc.)
        - **IPO Announcements & Market Predictions**
        
        **Ensure the output is in a clean, structured JSON format.**
        
        Here is the extracted website content:
        ```
        {text}
        ```
        """,
        model="command",  # Using Cohere's Command Model
        preamble="You are a financial data cleaner that extracts only valuable insights.",
        temperature=0.1,  # Lower temperature for consistency
    )

    cleaned_text = response.text
    return cleaned_text

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
    """Extract and process website content."""
    output_dir = Path("crawled_data")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "crawled_content.json"
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Define the extraction schema
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

    # Set up extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)
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
        result = await crawler.arun(url=url, config=config)

        if not result.success:
            print(f"Crawl failed for {url}: {result.error_message}")
            return

        # Parse, clean, and format extracted content
        data = json.loads(result.extracted_content)
        cleaned_data = clean_text_with_ai(data)  # Use AI to clean content
        formatted_data = format_content(cleaned_data)

        # Add metadata
        crawl_entry = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "content": formatted_data
        }

        # Append new data to existing entries
        existing_data.append(crawl_entry)

        #Save updated data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Data successfully saved to {output_file}")

# Example usage with multiple URLs
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
    """Run the crawler for each URL in the list."""
    for url in urls:
        await extract_website_content(url)

if __name__ == "__main__":
    asyncio.run(main())
