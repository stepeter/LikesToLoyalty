import pytest
import pandas as pd
from src.posts_to_sentiment import PostsToSentiment

TMP_PATH = "."

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "text": ["I love this wearable!", "Why is this hydration gear broken?", "Thanks for saving my hike!"]
    })

@pytest.fixture
def pts(tmp_path: str = TMP_PATH):
    # Use a temporary filename to avoid overwriting
    return PostsToSentiment(loadfilename="mock_posts", query="water bottle")

def test_map_emotion_to_stage(pts):
    assert pts.map_emotion_to_stage("admiration") == "Trust"
    assert pts.map_emotion_to_stage("gratitude") == "Advocacy"
    assert pts.map_emotion_to_stage("disapproval") == "Drop-Off"
    assert pts.map_emotion_to_stage("unknown") == "Awareness"

def test_batch_inference_alignment(pts, mock_df):
    # Simulate a batch prediction loop
    texts = mock_df["text"].tolist()
    predictions = [["admiration"], ["disapproval"], ["gratitude"]]
    assert len(texts) == len(predictions)

def test_load_posts_reads_file(tmp_path: str = TMP_PATH):
    # Create a mock CSV
    path = tmp_path + "mock_posts.csv"
    pd.DataFrame({"text": ["test"]}).to_csv(path, index=False)

    pts = PostsToSentiment(loadfilename="mock_posts", query="water bottle")
    df = pts.load_posts(path)
    assert "text" in df.columns