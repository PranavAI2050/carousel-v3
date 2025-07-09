from flask import Flask, request, jsonify
import google.generativeai as genai
from google.generativeai import GenerativeModel
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import mimetypes
import PyPDF2
import re
import json
import time
import copy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ============ CONFIG ============
app = Flask(__name__)

SERP_API_KEY = "a14c34de897520de3a7cb1"
GEMINI_KEY = "AJk"
MAX_LENGTH = 16000
MIN_SCORE = 6

genai.configure(api_key=GEMINI_KEY)
model = GenerativeModel('gemini-2.0-flash-lite')

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=options)

# ============ UTILITIES ============
def is_webpage(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return 'text/html' in response.headers.get('Content-Type', '').lower()
    except requests.exceptions.RequestException:
        return False

def is_pdf(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return 'application/pdf' in response.headers.get('Content-Type', '').lower()
    except requests.exceptions.RequestException:
        return False

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def split_text(text, max_length=MAX_LENGTH):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def scrape_with_selenium(url):
    try:
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        title = soup.find("title").get_text(strip=True) if soup.find("title") else "No Title"
        body_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
        return title, clean_text(body_text)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "No Title", ""

def gemini_model_summary(title, content, topic):
    prompt = f"""
You are an AI assistant analyzing an article to assess its relevance to a given topic and generate a clear, concise summary.

Your task:
1. Write a **concise summary** (2–4 sentences max)
2. Provide a **content_index** score between 0 and 10

Only respond in valid JSON format:
{{"summary": "...", "content_index": 7}}

Topic: {topic}
Title: {title}
[CONTENT START]
{content}
[CONTENT END]
"""
    try:
        response = model.generate_content(prompt)
        text = re.sub(r"^```(?:json)?|```$", "", response.text.strip())
        return json.loads(text)
    except Exception as e:
        return {"summary": f"Error: {e}", "content_index": 0}

def join_summaries(summaries, threshold=6):
    return " ".join([
        s['summary'].replace("\n", " ") for s in summaries if s['content_index'] >= threshold
    ])

def extract_images_with_context(url):
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    images_data = []
    title = soup.find("title").get_text(strip=True) if soup.find("title") else ""

    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        images_data.append({
            "title": title, "src": og_image["content"].strip(), "alt": "OG", "context": "OG image",
            "width": None, "height": None, "index": -1, "is_logo": False, "is_og": True
        })

    all_imgs = driver.find_elements("tag name", "img")
    bs_imgs = soup.find_all("img")

    for idx, img_el in enumerate(all_imgs):
        if idx >= len(bs_imgs): continue
        img_tag = bs_imgs[idx]
        src = img_tag.get("src", "") or img_tag.get("data-src", "")
        if not src or not src.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')): continue

        width = img_el.size.get("width", 0)
        height = img_el.size.get("height", 0)
        if width < 50 or height < 50: continue

        context = ""
        fig = img_tag.find_parent("figure")
        if fig and fig.find("figcaption"):
            context = fig.find("figcaption").get_text(strip=True)
        if not context:
            prev = img_tag.find_previous_sibling(["p", "h2", "h3"])
            if prev: context = prev.get_text(strip=True)

        images_data.append({
            "title": title, "src": src.strip(), "alt": img_tag.get("alt", ""),
            "context": context, "width": width, "height": height,
            "index": idx, "is_logo": "logo" in src.lower(), "is_og": False
        })

    return images_data

def evaluate_images_with_llm(images_data, topic):
    descs = []
    for i, img in enumerate(images_data):
        descs.append(f"""[{i}] Title: {img.get('title')}
Alt: {img.get('alt')}
Context: {img.get('context')}
Size: {img.get('width')}x{img.get('height')}
Logo: {"Yes" if img.get("is_logo") else "No"}
OG: {"Yes" if img.get("is_og") else "No"}""")

    prompt = f"""
Evaluate image descriptions for the topic "{topic}".

Return JSON like:
[
  {{"index": 0, "score": 8, "reason": "Relevant image ..."}}
]

{'\n'.join(descs)}
"""
    try:
        response = model.generate_content(prompt)
        text = re.sub(r"^```(?:json)?|```$", "", response.text.strip())
        return json.loads(text)
    except:
        return []

def build_carousel_prompt(topic, summary, top_images, json_template_str, extra_content="", constraints=[]):
    img_section = "\n".join([f"[{img['index']}] Score: {img['score']} – {img['reason']}" for img in top_images])
    extras = f"\nAdditional Notes:\n{extra_content}" if extra_content else ""
    if constraints:
        extras += "\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
    return f"""
You are an AI helping design carousel content.

Topic: {topic}
Summary:
\"\"\"
{summary}
\"\"\"

Top Images:
{img_section}

JSON Template:
{json_template_str}

{extras}

Instructions:
- Replace all `"content"` fields in the template.
- Use `"image:index"` format for image placeholders.
"""

def replace_image_indexes(llm_json, img_links):
    def replace(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and v.startswith("image:"):
                    idx = int(v.split(":")[1])
                    if 0 <= idx < len(img_links):
                        node[k] = img_links[idx]
                elif isinstance(v, (dict, list)):
                    replace(v)
        elif isinstance(node, list):
            for item in node:
                replace(item)
    result = copy.deepcopy(llm_json)
    replace(result)
    return result

# ============ MAIN ROUTE ============
@app.route("/generate_carousel", methods=["POST"])
def generate_carousel():
    try:
        data = request.get_json()
        topic = data.get("topic", "")
        links_to_be_search = data.get("links_to_be_search", [])
        extra_content = data.get("Extra_Content_Description", "")
        constraints = data.get("Content_Specifications", [])
        json_template = data.get("json_template_str", {})

        # Search Google if no links provided
        results_list = []
        if not links_to_be_search:
            params = {'q': topic, 'api_key': SERP_API_KEY, 'engine': 'google', 'num': '5'}
            search = GoogleSearch(params)
            results = search.get_dict()
            for r in results.get("organic_results", []):
                link = r.get("link")
                if link:
                    results_list.append({'title': r.get("title", "No Title"), 'link': link})
        else:
            for l in links_to_be_search:
                results_list.append({'title': "Custom Link", 'link': l})

        # Scrape & summarize
        valid_links, results_chunks = [], []
        for r in results_list:
            url = r['link']
            if not url or is_pdf(url) or not is_webpage(url): continue
            title, content = scrape_with_selenium(url)
            if not content: continue
            valid_links.append(url)
            for chunk in split_text(content):
                results_chunks.append({'title': title, 'content': chunk})

        summaries = []
        for r in results_chunks:
            summary = gemini_model_summary(r['title'], r['content'], topic)
            summaries.append({
                'title': r['title'],
                'summary': summary.get('summary', ''),
                'content_index': summary.get('content_index', 0)
            })

        summary_text = join_summaries(summaries)

        # Image scraping and evaluation
        all_images, img_links = [], []
        for url in valid_links:
            imgs = extract_images_with_context(url)
            all_images.extend(imgs)
            img_links.extend([i['src'] for i in imgs])

        evaluated = evaluate_images_with_llm(all_images, topic)
        top_images = [e for e in evaluated if e['score'] >= MIN_SCORE]

        prompt = build_carousel_prompt(topic, summary_text, top_images, json.dumps(json_template, indent=2), extra_content, constraints)
        response = model.generate_content(prompt)
        output_raw = re.sub(r"^```(?:json)?|```$", "", response.text.strip())
        carousel_json = json.loads(output_raw)
        final_output = replace_image_indexes(carousel_json, img_links)

        return jsonify(final_output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ RUN APP ============
if __name__ == "__main__":
    app.run(debug=True)
