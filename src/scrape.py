import json
import httpx
import pandas as pd
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from langdetect import detect
from pyparsing import (
    infixNotation,
    Keyword,
    Word,
    alphanums,
    ParserElement,
    opAssoc,
    dblQuotedString,
    removeQuotes,
)

ParserElement.enablePackrat()

class ScrapeBluesky:
    """
    Handles authentication, scraping, parsing, and filtering of Bluesky posts.

    Attributes:
        identifier (str): Bluesky handle used for API authentication.
        password (str): App password for accessing Bluesky API.
        n_posts_requested (int): Number of posts to retrieve.
    """

    def __init__(self, n_posts_requested: int = 1000, auth_fid: str = "auth.json",
                 identifier: Optional[str] = None, app_password: Optional[str] = None,
                 datarootpath: Optional[str] = "data") -> None:
        """
        Initialize ScrapeBluesky by loading credentials and setting request size.

        Args:
            n_posts_requested (int, optional): Number of posts to request. Defaults to 1000.
            auth_fid (str, optional): Location of json with password and identifier for using the Bluesky API.
        """
        if identifier and app_password:
            self.identifier: str = identifier
            self.password: str = app_password
        else:
            auth_info: Dict[str, str] = json.load(open(auth_fid))
            self.identifier: str = auth_info["identifier"]
            self.password: str = auth_info["app_password"]
        self.n_posts_requested: int = n_posts_requested
        self.datarootpath: str = datarootpath

    def scrape(
               self,
               query: str,
               savefilename: Optional[str] = None,
               stream: bool = False,
               date_start: str = "2025-01-01",
               date_end: str = "2025-07-21"
              ) -> Union[str, Tuple[pd.DataFrame, str]]:
        """
        Orchestrates full scraping pipeline: query, parse, filter, and save.

        Args:
            query (str): Keyword string for post search.
            savefilename (Optional[str]): Custom filename for output. Defaults to slugified query string.
            stream (str, optional): Determines whether to save the data (False) or output for streaming (True).
        """
        if savefilename is None:
            savefilename = "bsky_" + re.sub(r'[^a-zA-Z0-9]', '', query).lower()

        access_token: str = self.create_session()
        posts_data: List[Dict[str, Any]] = self.search_posts(query, access_token, self.n_posts_requested)
        df_posts: pd.DataFrame = self.parse_metadata(posts_data)
        df_posts = self.filter_by_time(df_posts, date_start, date_end)
        df_posts = self.filter_by_language(df_posts)
        df_posts["platform"] = "bluesky"
        
        # Only keep posts that match query
        parser = build_parser()
        parsed = parser.parseString(query)[0]
        df_posts["match"] = df_posts["text"].apply(parsed.eval)
        df_posts = df_posts[df_posts["match"]]
        df_posts.drop(columns=["match"], inplace=True)
        
        if stream:
            return df_posts, savefilename
        else:
            df_posts.to_csv(f"{self.datarootpath}/raw/{savefilename}.csv", index=False)
            print(f"Saved {len(df_posts)} posts!")
            return savefilename

    def create_session(self) -> str:
        """
        Authenticates with Bluesky and returns a valid access token.

        Returns:
            str: Bearer token for API authorization.
        """
        response = httpx.post(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            json={"identifier": self.identifier, "password": self.password}
        )
        session = response.json()
        return session["accessJwt"]

    def search_posts(self, query: str, access_token: str, n_posts_requested: int = 1000) -> List[Dict[str, Any]]:
        """
        Collects posts matching query from Bluesky with pagination.

        Args:
            query (str): Search keyword(s).
            access_token (str): Valid bearer token for API access.
            n_posts_requested (int, optional): Max post count to retrieve. Defaults to 1000.

        Returns:
            List[Dict[str, Any]]: List of raw post dictionaries.
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        cursor: Optional[str] = None
        posts_data: List[Dict[str, Any]] = []

        while True:
            params = {
                "q": query,
                "limit": min(100, n_posts_requested - len(posts_data))
            }
            if cursor:
                params["cursor"] = cursor

            response = httpx.get(
                "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
                headers=headers,
                params=params
            )
            data = response.json()
            posts = data.get("posts", [])
            posts_data.extend(posts)

            cursor = data.get("cursor")
            if not cursor or len(posts) == 0 or len(posts_data) >= n_posts_requested:
                break

        return posts_data

    def parse_metadata(self, posts_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extracts key metadata fields from Bluesky post JSON.

        Args:
            posts_data (List[Dict[str, Any]]): Raw post data.

        Returns:
            pd.DataFrame: Cleaned DataFrame with selected fields.
        """
        parsed_posts: List[Dict[str, Any]] = []
        for post in posts_data:
            record = {
                "author_handle": post.get("author", {}).get("handle"),
                "author_display_name": post.get("author", {}).get("displayName"),
                "created_at": post.get("record", {}).get("createdAt"),
                "text": post.get("record", {}).get("text"),
                "uri": post.get("uri"),
                "reply_count": post.get("replyCount"),
                "quote_count": post.get("quoteCount"),
                "repost_count": post.get("repostCount"),
                "embed_type": post.get("embed", {}).get("$type", None)
            }
            parsed_posts.append(record)

        return pd.DataFrame(parsed_posts)

    def filter_by_time(self, df: pd.DataFrame,
                       date_start: str = "2025-01-01",
                       date_end: str = "2025-07-21") -> pd.DataFrame:
        """
        Filters posts between January 1, 2025 and July 21, 2025.

        Args:
            df (pd.DataFrame): DataFrame with 'created_at' column.
            date_start (str, optional): Start date in YYYY-MM-DD format. Defaults to "2025-01-01".
            date_end (str, optional): End date in YYYY-MM-DD format. Defaults to "2025-07-21".

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", utc=True)
        df = df[
            (df["created_at"] >= pd.to_datetime(date_start, utc=True)) &
            (df["created_at"] <= pd.to_datetime(date_end, utc=True))
        ]
        return df

    def _strip_urls(self, text: str) -> str:
        """
        Removes URLs from a string using regex.

        Args:
            text (str): Original message text.

        Returns:
            str: Message text without URLs.
        """
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def _detect_language(self, text: str) -> str:
        """
        Detects language from text using langdetect.

        Args:
            text (str): Message text.

        Returns:
            str: Detected language code or 'na' if undetectable.
        """
        try:
            return detect(text)
        except BaseException:
            return "na"

    def filter_by_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out non-English posts based on detected language.

        Args:
            df (pd.DataFrame): DataFrame with 'text' column.

        Returns:
            pd.DataFrame: English-only DataFrame.
        """
        df["text"] = df["text"].apply(self._strip_urls)
        df["language"] = df["text"].apply(self._detect_language)
        return df[df["language"] == "en"]

    
# --- Code to check that query is in post ---
class Operand:
    def __init__(self, tokens):
        self.term = tokens[0]

    def eval(self, target: str) -> bool:
        return self.term.lower() in target.lower()


class Not:
    def __init__(self, tokens):
        self.arg = tokens[0][1]

    def eval(self, target: str) -> bool:
        return not self.arg.eval(target)


class And:
    def __init__(self, tokens):
        self.args = tokens[0][0::2]

    def eval(self, target: str) -> bool:
        return all(arg.eval(target) for arg in self.args)


class Or:
    def __init__(self, tokens):
        self.args = tokens[0][0::2]

    def eval(self, target: str) -> bool:
        return any(arg.eval(target) for arg in self.args)


def build_parser():
    # Match quoted phrases or bare words
    phrase = (
        dblQuotedString.setParseAction(removeQuotes)
        | Word(alphanums + "_-")
    )
    phrase.setParseAction(Operand)

    # Logical operators
    NOT = Keyword("NOT", caseless=True)
    AND = Keyword("AND", caseless=True)
    OR = Keyword("OR", caseless=True)

    expr = infixNotation(
        phrase,
        [
            (NOT, 1, opAssoc.RIGHT, Not),
            (AND, 2, opAssoc.LEFT, And),
            (OR, 2, opAssoc.LEFT, Or),
        ],
    )

    return expr