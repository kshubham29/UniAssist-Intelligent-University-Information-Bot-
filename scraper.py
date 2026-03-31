import asyncio
import json
import os
import re
import hashlib
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

BASE_URLS = [
    "https://www.ncuindia.edu/",
    "https://www.ncuonline.edu.in/",
    "https://www.ncuindia.edu/contact-us/",
    "https://www.ncuindia.edu/programme/ph-d/",
    "https://www.ncuindia.edu/the-northcap-university/about-us/",
    "https://www.ncuindia.edu/international-relations/",
    "https://www.ncuindia.edu/fee-structure/",
    "https://www.ncuindia.edu/governance/",
    "https://www.ncuindia.edu/library/",
    "https://www.ncuindia.edu/societies-clubs/",
    "https://www.ncuindia.edu/the-northcap-incubation-innovation-centre/",
    "https://www.ncuindia.edu/careers/",
    "https://www.ncuindia.edu/our-placements-2-2/",
    "https://www.ncuindia.edu/educate-india-society/",
    "https://www.ncuindia.edu/scholarship/",
    "https://www.ncuindia.edu/school/school-of-engineering/",
    "https://www.ncuindia.edu/programme-type/ug-programmes/",
    "https://www.ncuindia.edu/prof-nupur-prakash/",
    "https://www.ncuindia.edu/announcements/"
]

MAX_PAGES = 1000
MAX_DEPTH = 2   # Reduced for cleaner crawl

allowed_domains = {urlparse(url).netloc for url in BASE_URLS}
visited = set()
queued = set(BASE_URLS)
queue = deque([(url, 0) for url in BASE_URLS])
data = []
seen_hashes = set()

os.makedirs("data", exist_ok=True)

def clean_url(url):
    defragged, _ = urldefrag(url)
    return defragged.strip()

def is_valid(url):
    parsed = urlparse(url)

    if parsed.netloc not in allowed_domains:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    # Block WordPress API and junk links
    blocked_keywords = [
        "wp-json",
        "feed",
        "format=xml",
        "?p=",
        "embed",
        "amp",
        ".xml"
    ]

    if any(keyword in url for keyword in blocked_keywords):
        return False

    blocked_extensions = [".pdf", ".jpg", ".png", ".zip", ".docx"]

    if any(url.endswith(ext) for ext in blocked_extensions):
        return False

    return True

def is_duplicate(text):
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if content_hash in seen_hashes:
        return True
    seen_hashes.add(content_hash)
    return False

def extract_content(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script","style","noscript","iframe","footer","nav","aside"]):
        tag.decompose()

    texts = []
    for tag in soup.find_all(["h1","h2","h3","h4","p","li"]):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) > 30:
            texts.append(text)

    return "\n".join(texts)

async def crawl():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        while queue and len(data) < MAX_PAGES:
            url, depth = queue.popleft()

            if url in visited:
                continue

            try:
                print(f"[Depth {depth}] Scraping: {url}")

                await page.goto(url, timeout=60000)

                # Faster than networkidle
                await page.wait_for_load_state("domcontentloaded")

                html = await page.content()
                visited.add(url)

                text = extract_content(html)

                if text and not is_duplicate(text):
                    data.append({"url": url, "content": text})

                if depth < MAX_DEPTH:
                    soup = BeautifulSoup(html, "html.parser")
                    for link in soup.find_all("a", href=True):
                        full_url = clean_url(urljoin(url, link["href"]))
                        if is_valid(full_url) and full_url not in visited and full_url not in queued:
                            queue.append((full_url, depth + 1))
                            queued.add(full_url)

            except Exception as e:
                print("Error:", e)

        await browser.close()

    with open("data/ncu_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\nScraping complete. Pages:", len(data))

if __name__ == "__main__":
    asyncio.run(crawl())