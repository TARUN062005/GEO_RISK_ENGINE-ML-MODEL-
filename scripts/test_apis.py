"""Quick API connectivity test — TEMPORARY"""
import asyncio
import httpx
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import NEWSAPI_KEY, GNEWS_KEY

async def test():
    print(f"NEWSAPI_KEY: {'YES' if NEWSAPI_KEY else 'NO'} (len={len(NEWSAPI_KEY)})")
    print(f"GNEWS_KEY:   {'YES' if GNEWS_KEY else 'NO'} (len={len(GNEWS_KEY)})")
    
    # Test NewsAPI
    if NEWSAPI_KEY:
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get("https://newsapi.org/v2/everything", params={
                    "q": "conflict war military",
                    "language": "en",
                    "pageSize": 3,
                    "apiKey": NEWSAPI_KEY,
                })
                print(f"\nNewsAPI status: {resp.status_code}")
                data = resp.json()
                articles = data.get("articles", [])
                print(f"NewsAPI articles: {len(articles)}")
                for a in articles[:3]:
                    print(f"  - {a['title'][:70]}")
                if resp.status_code != 200:
                    print(f"  Error: {data.get('message', '?')}")
            except Exception as e:
                print(f"NewsAPI error: {e}")

    # Test GNews
    if GNEWS_KEY:
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get("https://gnews.io/api/v4/search", params={
                    "q": "conflict war",
                    "lang": "en",
                    "max": 3,
                    "token": GNEWS_KEY,
                })
                print(f"\nGNews status: {resp.status_code}")
                data = resp.json()
                articles = data.get("articles", [])
                print(f"GNews articles: {len(articles)}")
                for a in articles[:3]:
                    print(f"  - {a['title'][:70]}")
                if resp.status_code != 200:
                    print(f"  Error: {data}")
            except Exception as e:
                print(f"GNews error: {e}")

asyncio.run(test())
