import os
import re
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import yaml
import feedparser
import requests
from dateutil import parser as dateparser
from bs4 import BeautifulSoup
from openai import OpenAI


DB_PATH = "state.sqlite"
FEEDS_YAML = "feeds.yaml"


@dataclass
class Item:
    source: str
    title: str
    url: str
    published: datetime
    summary: str
    kind: str  # "paper" | "blog" | "news" | "unknown"
    tags: List[str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    url = re.sub(r"(\?|&)(utm_[^=&]+)=[^&]+", r"\1", url)
    url = url.replace("?&", "?").rstrip("?&")
    return url


def item_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS seen (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def db_has_seen(url: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM seen WHERE id = ?", (item_id(url),))
    row = cur.fetchone()
    conn.close()
    return row is not None


def db_mark_seen(url: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO seen (id, url, first_seen_utc) VALUES (?, ?, ?)",
        (item_id(url), url, utc_now().isoformat()),
    )
    conn.commit()
    conn.close()


def load_config() -> Dict:
    with open(FEEDS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict):
    with open(FEEDS_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(" ", strip=True)


def parse_datetime(entry) -> datetime:
    for key in ["published", "updated", "created"]:
        val = getattr(entry, key, None)
        if val:
            try:
                dt = dateparser.parse(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return utc_now()


def guess_kind(url: str, tags: List[str]) -> str:
    u = (url or "").lower()
    if "arxiv.org" in u:
        return "paper"
    if "news" in tags:
        return "news"
    if any(t in tags for t in ["blog", "econ", "policy", "tech"]):
        return "blog"
    return "unknown"


def fetch_rss_items(feed_name: str, feed_url: str, tags: List[str]) -> List[Item]:
    fp = feedparser.parse(feed_url)
    items: List[Item] = []
    for e in fp.entries:
        url = normalize_url(getattr(e, "link", "") or "")
        title = (getattr(e, "title", "") or "").strip()
        summary = clean_html(getattr(e, "summary", "") or getattr(e, "description", "") or "")
        published = parse_datetime(e)

        if not url or not title:
            continue

        items.append(
            Item(
                source=feed_name,
                title=title,
                url=url,
                published=published,
                summary=(summary or "")[:1600],
                kind=guess_kind(url, tags),
                tags=tags,
            )
        )
    return items


def arxiv_rss_url(category: str) -> str:
    return f"http://export.arxiv.org/rss/{category}"


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def keyword_score(text: str, keywords: List[str]) -> float:
    t = (text or "").lower()
    score = 0.0
    for kw in keywords:
        k = (kw or "").strip().lower()
        if k and k in t:
            score += 1.0
    return score


def select_top(items: List[Item], cfg: Dict) -> List[Item]:
    rank_cfg = cfg.get("ranking", {})
    max_items = int(rank_cfg.get("max_items_to_consider", 120))
    top_k = int(rank_cfg.get("top_k", 5))
    diversify = bool(rank_cfg.get("diversify_by_type", True))

    keywords = cfg.get("arxiv", {}).get("keywords", []) + [
        "ai",
        "artificial intelligence",
        "llm",
        "automation",
        "labor",
        "jobs",
        "employment",
        "productivity",
        "wages",
        "tasks",
        "occupation",
        "inequality",
        "market power",
        "open source",
        "compute",
    ]

    now = utc_now()
    scored: List[Tuple[float, Item]] = []
    items = sorted(items, key=lambda x: x.published, reverse=True)[:max_items]

    for it in items:
        text = f"{it.title}\n{it.summary}"
        ks = keyword_score(text, keywords)
        hours_old = max(1.0, (now - it.published).total_seconds() / 3600.0)
        recency = 1.0 / (hours_old ** 0.35)
        kind_prior = 0.3 if it.kind == "paper" else 0.0
        score = (ks * 1.15) + recency + kind_prior
        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not diversify:
        return [it for _, it in scored[:top_k]]

    chosen: List[Item] = []
    paper = next((it for _, it in scored if it.kind == "paper"), None)
    if paper:
        chosen.append(paper)

    for _, it in scored:
        if len(chosen) >= top_k:
            break
        if it in chosen:
            continue
        chosen.append(it)

    final: List[Item] = []
    seen_keys = set()
    for it in chosen:
        k = re.sub(r"\W+", "", it.title.lower())[:90]
        if k in seen_keys:
            continue
        seen_keys.add(k)
        final.append(it)

    return final[:top_k]


def slack_post(webhook_url: str, text: str):
    resp = requests.post(webhook_url, json={"text": text}, timeout=25)
    if resp.status_code >= 300:
        raise RuntimeError(f"Slack webhook failed: {resp.status_code} {resp.text[:250]}")


def build_llm_prompt(items: List[Item], interest_profile: str, style_pack: str) -> str:
    blocks = []
    for i, it in enumerate(items, 1):
        blocks.append(
            f"[{i}] {it.title}\n"
            f"URL: {it.url}\n"
            f"Source: {it.source}\n"
            f"Published (UTC): {it.published.isoformat()}\n"
            f"Snippet: {it.summary}\n"
        )

    return f"""
You are helping an economist-researcher track AI impacts on the economy, society, and jobs.

INTEREST PROFILE:
{interest_profile}

STYLE PACK:
{style_pack}

TASK:
1) Select the top 4–5 items (you may drop low-signal ones).
2) For each selected item:
   - 5 bullet summary (plain language)
   - 2 bullets: economic interpretation / mechanism
   - 1 bullet: what I'd cite or watch for (data, identification, assumptions, missing evidence)
3) Draft:
   - 2 X posts (<= 240 words each)
   - 2 LinkedIn posts (120–250 words each)
Each post should reference 1–2 selected items with links and connect to the interest profile themes.
Be grounded and slightly skeptical. No hype. Do not invent facts not supported by the snippets.

ITEMS:
{chr(10).join(blocks)}

OUTPUT FORMAT (exact headings):
TOP PICKS:
- ...

SUMMARIES:
1) ...
2) ...

DRAFT POSTS:
X POST 1:
...
X POST 2:
...
LINKEDIN POST 1:
...
LINKEDIN POST 2:
...
""".strip()


def llm_run(api_key: str, model: str, prompt: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise research assistant. Do not invent facts. Use only provided snippets and common knowledge.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def extract_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://[^\s<>()]+", text or "")
    return [normalize_url(u) for u in urls if u]


def email_intake_add_feeds(cfg: Dict):
    email_cfg = cfg.get("email_intake", {})
    if not email_cfg.get("enabled", False):
        return

    host = os.environ.get("IMAP_HOST", "imap.gmail.com")
    user = os.environ.get("IMAP_USER", "")
    pw = os.environ.get("IMAP_APP_PASSWORD", "")
    subject_prefix = email_cfg.get("subject_prefix", "ADD FEED")

    if not user or not pw:
        return

    import imaplib
    import email

    M = imaplib.IMAP4_SSL(host)
    M.login(user, pw)
    M.select("INBOX")

    typ, data = M.search(None, f'(UNSEEN SUBJECT "{subject_prefix}")')
    if typ != "OK":
        M.logout()
        return

    msg_ids = data[0].split()
    if not msg_ids:
        M.logout()
        return

    new_urls: List[str] = []
    for msg_id in msg_ids:
        typ, msg_data = M.fetch(msg_id, "(RFC822)")
        if typ != "OK":
            continue
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        body_text = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition") or "")
                if ctype == "text/plain" and "attachment" not in disp:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_text += payload.decode(errors="ignore")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body_text = payload.decode(errors="ignore")

        new_urls.extend(extract_urls(body_text))
        M.store(msg_id, "+FLAGS", "\\Seen")

    M.logout()

    if not new_urls:
        return

    existing = {f.get("url") for f in cfg.get("feeds", []) if f.get("url")}
    added = 0
    for u in new_urls:
        if u in existing:
            continue
        cfg["feeds"].append(
            {
                "name": "Added via email",
                "url": u,
                "type": "rss",
                "tags": ["added"],
            }
        )
        existing.add(u)
        added += 1

    if added:
        save_config(cfg)


def main():
    ensure_db()
    cfg = load_config()

    email_intake_add_feeds(cfg)

    all_items: List[Item] = []

    for f in cfg.get("feeds", []):
        name = f.get("name", "Unknown")
        url = f.get("url", "")
        tags = f.get("tags", []) or []
        if not url:
            continue
        try:
            all_items.extend(fetch_rss_items(name, url, tags))
        except Exception:
            pass

    for cat in cfg.get("arxiv", {}).get("categories", []):
        try:
            rss = arxiv_rss_url(cat)
            all_items.extend(fetch_rss_items(f"arXiv {cat}", rss, ["paper", "arxiv", cat]))
        except Exception:
            pass

    days_back = int(cfg.get("ranking", {}).get("days_back", 2))
    cutoff = utc_now() - timedelta(days=days_back)

    fresh: List[Item] = []
    for it in all_items:
        if it.published < cutoff:
            continue
        if db_has_seen(it.url):
            continue
        fresh.append(it)

    if not fresh:
        webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
        if webhook:
            slack_post(webhook, f"*AI Econ Digest* ({utc_now().date().isoformat()}): No new items found in last {days_back} days.")
        return

    top_items = select_top(fresh, cfg)

    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY secret/env var.")

    interest_profile = load_text("interest_profile.txt")
    style_pack = load_text("style_pack.txt")
    prompt = build_llm_prompt(top_items, interest_profile, style_pack)
    llm_text = llm_run(api_key, model, prompt)

    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook:
        raise RuntimeError("Missing SLACK_WEBHOOK_URL secret/env var.")

    links_line = ", ".join([f"<{it.url}|{it.title[:70]}>" for it in top_items])
    header = f"*AI Econ Digest* ({utc_now().date().isoformat()})\nTop picks: {links_line}\n"
    message = header + "\n" + llm_text

    if len(message) > 35000:
        message = message[:34000] + "\n\n(Truncated due to Slack message length limits.)"

    slack_post(webhook, message)

    for it in top_items:
        db_mark_seen(it.url)


if __name__ == "__main__":
    main()
