from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import torch
from typing import Optional, List, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

DEVICE = 0 if torch.cuda.is_available() else -1

class PostsToSentiment:
    """
    Loads scraped social media posts and assigns sentiment labels using a Hugging Face transformer model.
    Maps emotions to funnel stages for downstream analysis.
    """

    def __init__(self, 
                 loadfilename: str,
                 query: str,
                 model_name: str = "SamLowe/roberta-base-go_emotions",
                 batch_size: int = 100,
                 stream: bool = False,
                 suppress_neutral: bool = False,
                 datarootpath: str = "data"
                ) -> None:
        """
        Args:
            loadfilename (str): Name of the raw CSV file (without extension) containing social media posts.
            model_name (str, optional): Hugging Face model to use for sentiment classification. Defaults to GoEmotions BERT.
            batch_size (int, optional): Number of texts to process per inference batch. Defaults to 100.
            stream (bool, optional): Load/save to CSV if not streaming (False).
            suppress_neutral (bool, optional): Select highest non-neutral emotion from model result (for GoEmotions BERT).
        """
        self.model_name: str = model_name
        self.loadfilename: str = loadfilename
        self.batch_size: int = batch_size
        self.stream: bool = stream
        self.query: str = query
        self.suppress_neutral: bool =suppress_neutral
        self.datarootpath: str = datarootpath

    def load_posts(self, fid: Optional[str] = None) -> pd.DataFrame:
        """
        Loads scraped post data from CSV.
        
        Args:
            fid (str, optional): Location of data file with scraped posts.

        Returns:
            pd.DataFrame: Raw posts with text content.
        """
        if fid is None:
            fid = f"{self.datarootpath}/raw/{self.loadfilename}.csv"
            
        return pd.read_csv(fid)

    def create_prediction_pipe(self) -> pipeline:
        """
        Initializes the Hugging Face text classification pipeline with specified model.

        Returns:
            pipeline: Sentiment classifier pipeline object.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True
        )
        classifier = pipeline("text-classification",
                              model=model,
                              tokenizer=tokenizer,
                              top_k=2 if self.suppress_neutral else 1,
                              device=DEVICE)
        return classifier

    def map_emotion_to_stage(self, emotion_label: str) -> str:
        """
        Maps emotion label to corresponding funnel stage.

        Args:
            emotion_label (str): Predicted emotion from model.

        Returns:
            str: Funnel stage name.
            
        Emotion mapping and justification
        
        Emotion Label | Funnel Stage | Justification
        curiosity       Awareness      Spark of attention without commitment
        neutral         Awareness      Baseline mention, passive signal
        admiration      Trust          Positive signal of credibility
        optimism        Interest       Hopeful engagement, emotional investment
        excitement      Interest       High energy attention
        desire          Interest       Expressed want or preference
        anticipation    Interest       Planning or looking forward to action
        confusion       Drop-Off       Uncertainty that could hinder progression
        disapproval     Drop-Off       Explicit negative sentiment
        anger           Drop-Off       Strong disengagement or backlash
        gratitude       Advocacy       Endorsement behavior, thankfulness
        pride           Advocacy       Expressed ownership or promotion
        love            Advocacy       Emotional alignment with brand or idea
        """
        mapping: Dict[str, str] = {
            "curiosity": "Awareness",
            "neutral": "Awareness",
            "admiration": "Trust",
            "optimism": "Interest",
            "excitement": "Interest",
            "desire": "Interest",
            "anticipation": "Interest",
            "confusion": "Drop-Off",
            "disapproval": "Drop-Off",
            "anger": "Drop-Off",
            "gratitude": "Advocacy",
            "pride": "Advocacy",
            "love": "Advocacy"
        }
        return mapping.get(emotion_label, "Awareness")

    def predict_sentiment(
                          self,
                          savefilename: Optional[str] = None,
                          df: Optional[pd.DataFrame] = None,
                          max_text_len: int = 512
                         ) -> Union[str, Tuple[pd.DataFrame, str]]:
        """
        Runs sentiment classification on all texts, maps to funnel stages, and saves labeled output.

        Args:
            savefilename (Optional[str], optional): Output filename (without path). Defaults to auto-generated.

        Returns:
            str: Name of saved file (used downstream).
        """
        if savefilename is None:
            savefilename = "labeled_posts_" + self.loadfilename
        
        if self.stream:
            assert df is not None
        else:
            df = self.load_posts()
        df["text"] = f"Query: {self.query}. Post: " + df["text"]
        texts: List[str] = df["text"].tolist()
        texts = [t.strip()[:max_text_len] for t in texts if isinstance(t, str)]

        classifier = self.create_prediction_pipe()

#         predictions: List[List[Dict[str, Any]]] = []
#         for i in range(0, len(texts), self.batch_size):
#             batch = texts[i:i + self.batch_size]
#             batch_preds = classifier(batch)
#             predictions.extend(batch_preds)
        
        # Use batching and multithreading to speed up ML inference
        def _predict_batch(batch):
            return classifier(batch)
        chunks = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(_predict_batch, chunks))
        predictions = [item for sublist in results for item in sublist]

        if self.suppress_neutral:
            df["emotion"] = [pred[0]["label"] if \
                             (pred[0]["label"] != "neutral") \
                             else pred[1]["label"] for pred in predictions]
        else:
            df["emotion"] = [pred[0]["label"] for pred in predictions]
        df["funnel_stage"] = df["emotion"].apply(self.map_emotion_to_stage)
        
        if self.stream:
            return df, savefilename
        else:        
            df.to_csv(f"{self.datarootpath}/processed/{savefilename}.csv", index=False)

            print("Funnel stages assigned and saved.")
            return savefilename