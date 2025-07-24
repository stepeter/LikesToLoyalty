import pytest
import pandas as pd
from unittest.mock import patch

from src.scrape import ScrapeBluesky  # adjust import path as needed

@pytest.fixture
def scraper():
    return ScrapeBluesky(
                         n_posts_requested = 5,
                         identifier = "mock_username",
                         app_password = "mock_app_password"
                        )

def test_strip_urls(scraper):
    input_text = "Check this out https://example.com and www.test.com"
    expected = "Check this out  and "
    assert scraper._strip_urls(input_text) == expected

def test_detect_language_english(scraper):
    assert scraper._detect_language("This is a test sentence.") == "en"

def test_detect_language_failure(scraper):
    assert scraper._detect_language("") == "na"

def test_filter_by_time(scraper):
    df = pd.DataFrame({
        "created_at": ["2025-01-05", "2024-12-31", "2025-06-15"]
    })
    df_filtered = scraper.filter_by_time(df, date_start="2025-01-01", date_end="2025-07-21")
    assert len(df_filtered) == 2  # should drop the 2024-12-31 entry

def test_filter_by_language(scraper):
    df = pd.DataFrame({
        "text": ["Hello world", "Bonjour le monde", "こんにちは"]
    })
    df_filtered = scraper.filter_by_language(df)
    assert all(df_filtered["language"] == "en")  # Only "Hello world" should remain

@patch("httpx.post")
def test_create_session(mock_post, scraper):
    mock_post.return_value.json.return_value = {"accessJwt": "mocked_token"}
    token = scraper.create_session()
    assert token == "mocked_token"

@patch("httpx.get")
def test_search_posts(mock_get, scraper):
    # Mock paginated responses with a cursor
    mock_get.return_value.json.side_effect = [
        {"posts": [{"record": {"text": "post 1"}}], "cursor": "abc"},
        {"posts": [{"record": {"text": "post 2"}}], "cursor": None}
    ]
    posts = scraper.search_posts("hydration", access_token="fake_token", n_posts_requested=2)
    assert len(posts) == 2

def test_parse_metadata(scraper):
    input_data = [{
        "author": {"handle": "user1", "displayName": "User One"},
        "record": {"createdAt": "2025-01-01", "text": "Sample text"},
        "uri": "at://test",
        "replyCount": 2,
        "quoteCount": 1,
        "repostCount": 0,
        "embed": {"$type": "image"}
    }]
    df = scraper.parse_metadata(input_data)
    assert df.loc[0, "author_handle"] == "user1"
    assert df.loc[0, "embed_type"] == "image"