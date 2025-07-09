import google.generativeai as genai
from google.generativeai import GenerativeModel
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import mimetypes
import re
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv
import os

load_dotenv()
serp_api_key = os.environ.get("serp_api_key")
gemini_key = os.environ.get("gemini_key")

genai.configure(api_key= gemini_key)
model = GenerativeModel('gemini-2.0-flash-lite')
max_length = 16000
MAX_CHARS = 16000 
MIN_SCORE = 6 


def is_webpage(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        return 'text/html' in content_type
    except requests.exceptions.RequestException:
        return False


def is_pdf(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        return 'application/pdf' in content_type
    except requests.exceptions.RequestException:
        return False


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text(text, max_length=max_length):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


def scrape_with_selenium(url, driver):
    try:
        driver.get(url)
        time.sleep(3)  # You can replace with WebDriverWait for better control

        soup = BeautifulSoup(driver.page_source, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "No Title"

        # Scrape visible text from body
        body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
        body_text = clean_text(body_text)

        return title, body_text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "No Title", ""
    

def gemini_model_summary(title, content, topic):
    if not content.strip():
        return {"summary": "", "content_index": 0}

    prompt = f"""
You are an AI assistant analyzing an article to assess its relevance to a given topic and generate a clear, concise summary.

Your task:
1. Write a **concise summary** (2–4 sentences max) highlighting:
   - Key facts
   - Named entities
   - Relevant quotes or statistics
   - Information suitable for a social media carousel
2. Provide a **content_index** score between 0 and 10:
   - 0 = not relevant to the topic
   - 10 = highly relevant for carousel content
   - 1–9 = partially relevant

Only respond in valid JSON format, like below:

{{
  "summary": "Short, clear summary here.",
  "content_index": 7
}}

---
Topic: {topic}
Title: {title}

[CONTENT START]
{content}
[CONTENT END]
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Remove any code block formatting like ```json or ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Parse cleaned JSON
        parsed = json.loads(text)

        # Validate keys
        if not isinstance(parsed, dict) or 'summary' not in parsed or 'content_index' not in parsed:
            raise ValueError("Invalid response format")

        return parsed

    except Exception as e:
        return {"summary": f"Error generating summary: {e}", "content_index": 0}


def join_summaries(summaries, threshold=6):
    return " ".join([
        summary['summary'].replace("\n", " ").replace("\r", " ")
        for summary in summaries
        if summary.get('content_index', 0) > threshold
    ])


def extract_images_with_context(url, driver, valid_extensions=('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')):
    driver.get(url)
    time.sleep(3)  # Allow JS to load fully

    soup = BeautifulSoup(driver.page_source, "html.parser")
    images_data = []

    # Extract page title
    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    # 1. Try Open Graph image first (social preview image)
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        images_data.append({
            "title": page_title,
            "src": og_image["content"].strip(),
            "alt": "Open Graph Image",
            "context": "This is the Open Graph image used for previews",
            "width": None,
            "height": None,
            "position_index": -1,  # Prioritize OG image
            "is_probably_logo": False,
            "is_og_image": True
        })

    # 2. Extract regular <img> tags with additional metadata
    all_imgs = driver.find_elements("tag name", "img")  # For dimension info via Selenium
    bs_imgs = soup.find_all("img")

    for idx, img_el in enumerate(all_imgs):
        try:
            if idx >= len(bs_imgs):
                continue  # Avoid index mismatch

            img_tag = bs_imgs[idx]
            alt_text = img_tag.get("alt", "").strip()
            src = img_tag.get("src", "") or img_tag.get("data-src", "")
            src = src.strip()

            # Skip if src is empty or invalid
            if not src or not src.lower().endswith(valid_extensions):
                continue

            # Filter by common non-content patterns
            if re.search(r"(sprite|icon|logo|tracking|ads|analytics)", src.lower()):
                continue

            # Get dimensions
            width = img_el.size.get("width", 0)
            height = img_el.size.get("height", 0)
            if width < 50 or height < 50:
                continue  # Skip tiny/invisible images

            # Try to find context: figcaption > parent > siblings
            context = ""

            figure = img_tag.find_parent("figure")
            if figure:
                figcaption = figure.find("figcaption")
                if figcaption:
                    context = figcaption.get_text(strip=True)

            if not context:
                current = img_tag
                for _ in range(2):
                    parent = current.find_parent()
                    if parent and parent.get_text(strip=True):
                        context = parent.get_text(strip=True)
                        break
                    current = parent if parent else current

            if not context:
                prev = img_tag.find_previous_sibling(["p", "h2", "h3"])
                if prev:
                    context = prev.get_text(strip=True)

            images_data.append({
                "title": page_title,
                "src": src,
                "alt": alt_text,
                "context": context,
                "width": width,
                "height": height,
                "position_index": idx,
                "is_probably_logo": "logo" in src.lower() or "logo" in alt_text.lower(),
                "is_og_image": False
            })

        except Exception as e:
            print(f"Image {idx} skipped due to error: {e}")
            continue

    return images_data

def collect_valid_images_from_links(links, results_list, driver):

    all_valid_images = []

    for i, url in enumerate(links):
        title = results_list[i].get('title', f"Page {i+1}")
        print(f"🔗 Processing: {title} ({url})")
        try:
            images = extract_images_with_context(url, driver)
            all_valid_images.extend(images)
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")

    return all_valid_images

def convert_images_to_llm_strings(images_data):
    description_strings = []
    image_urls = []

    for idx, img in enumerate(images_data):
        description = (
            f"[{idx}] Title: {img.get('title', 'Unknown')}\n"
            f"Alt text: {img.get('alt', 'None')}\n"
            f"Context: {img.get('context', 'None')}\n"
            f"Dimensions: {img.get('width')}x{img.get('height')}\n"
            f"Likely Logo: {'Yes' if img.get('is_probably_logo') else 'No'}\n"
            f"Open Graph Image: {'Yes' if img.get('is_og_image') else 'No'}"
        )
        description_strings.append(description)
        image_urls.append(img.get('src', ''))

    return description_strings, image_urls

def chunk_descriptions(desc_strings, max_chars=MAX_CHARS):
    """Split image descriptions into chunks under max_chars."""
    chunks, current_chunk, current_len = [], [], 0

    for i, desc in enumerate(desc_strings):
        desc_len = len(desc)
        if current_len + desc_len > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk, current_len = [], 0
        current_chunk.append(desc)
        current_len += desc_len

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def build_prompt(chunk,topic):
    """Builds the LLM prompt for a chunk of image descriptions."""
    image_descriptions = "\n\n".join(chunk)

    return f"""
You are a helpful AI that evaluates a series of image descriptions to determine their usefulness for creating visual carousels related to topic: {topic}.

For each image, return:
- "index": the image number (as provided),
- "score": an integer 1–10 showing how relevant/valuable the image is,
- "reason": why this image may or may not be useful for content generation.

Only respond in JSON list format.

Images:
{image_descriptions}

Respond like:
[
  {{
    "index": 0,
    "score": 9,
    "reason": "Shows core idea with relevant caption"
  }},
  ...
]
"""

def evaluate_chunks_with_llm(chunks,topic):
    all_results = []

    for chunk in chunks:
        prompt = build_prompt(chunk, topic)
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Clean any markdown-style code block formatting
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            parsed = json.loads(text)

            for result in parsed:
                # Validate and append
                if isinstance(result, dict) and "index" in result:
                    all_results.append(result)

        except Exception as e:
            print(f"Error on chunk: {e}")
 
    return all_results


def filter_top_images(evaluated_results, img_links, threshold=MIN_SCORE):
    """Filter out top images based on score."""
    selected = []
    for item in evaluated_results:
        index = item.get("index")
        score = item.get("score", 0)
        reason = item.get("reason", "")

        if score >= threshold and index is not None and index < len(img_links):
            selected.append({
                "index": index,
                "score": score,
                "reason": reason,
            })

    return selected

def generate_carousel_prompt(topic, summary_text, top_images, json_template_str, Extra_Content_Description, Content_Specifications):
    import json
    # Ensure Content_Specifications is a list if None is passed
    if Content_Specifications is None:
        Content_Specifications = []

    # Step 1: Format top_images for display
    formatted_images = "\n".join([
        f"[{img['index']}] Score: {img['score']} – {img['reason']}"
        for img in top_images
    ])

    # Step 2: Serialize the JSON template
    json_template_str = json.dumps(json_template_str, indent=2)

    # Step 3: Optional content sections
    extra_content_section = f"\nAdditional creative guidance:\n{Extra_Content_Description.strip()}\n" if Extra_Content_Description.strip() else ""
    content_spec_section = "\nSpecific content constraints and requirements:\n" + "\n".join([f"- {spec}" for spec in Content_Specifications]) if Content_Specifications else ""
    optional_guidance = (extra_content_section + content_spec_section).strip()

    if optional_guidance:
        optional_guidance = f"\n---\n\n## Additional Notes\n{optional_guidance}"

    # Step 4: Construct the prompt
    prompt = f"""
You are an AI assistant helping to design visually engaging carousel slides for social media.

## Objective:
Fill in the following JSON template with content for a carousel about the topic: **"{topic}"**.

Use the summarized content and image options below to generate text and select suitable images.

---

### Rules for Filling the Template:

1. If `"content": "text"`:
   - Write **concise, relevant content** based on the summary and the element's `"description"`.
   - If a character or word limit is mentioned, **strictly follow it**.
   - The tone should match the title or description (e.g., persuasive, informative, punchy, etc.).

2. If `"content": "url"`:
   - Replace `"url"` with an image reference using this format:
     `"image:index"`
     Example: `"image:7"` = image at index 7.
   - **Only use the images from the provided list of `top_images`**.
   - Do **not invent or guess image content**.
   - Pick the best image based on its score and explanation (`reason`).

---

## Topic
{topic}

---

## Summarized Content
\"\"\"
{summary_text}
\"\"\"
{optional_guidance}

---

## Top Images to Choose From
Each entry includes an index, a relevance score (1–10), and a reason explaining its usefulness:

{formatted_images}

---

## JSON Template
Replace only the `"content"` values. Do not change any other keys or structure.

{json_template_str}

---

## Expected Output
Return ONLY the completed JSON object, with each `"content"` field correctly filled in.

- Use `"image:index"` for images (e.g., `"image:0"`).
- Follow all text formatting rules and word limits.
"""

    return prompt


def replace_image_indexes_with_urls(llm_output_json,img_links):
    """
    Replace 'image:X' placeholders in LLM output with actual image URLs from img_links list.
    """
    def replace_content(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict):
                    replace_content(value)
                elif isinstance(value, str) and value.startswith("image:"):
                    try:
                        index = int(value.split(":")[1])
                        if 0 <= index < len(img_links):
                            node[key] = img_links[index]
                    except (IndexError, ValueError):
                        continue  # Skip malformed entries
        elif isinstance(node, list):
            for item in node:
                replace_content(item)

    import copy
    updated = copy.deepcopy(llm_output_json)
    replace_content(updated)
    return updated


