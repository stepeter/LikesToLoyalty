# From Likes to Loyalty — A Sentiment-Driven Funnel Explorer

This interactive NLP-powered tool visualizes the emotional journey of users through a brand engagement funnel. Using social media messages collected from BlueSky, it classifies sentiment and maps messages to funnel stages, illustrating the progression from Awareness to Advocacy.

## Features
- Transformer-based sentiment tagging with BERT
- Funnel stage prediction and drop-off analytics
- Streamlit dashboard for message filtering and persona simulation

## Tech Stack
Python, HuggingFace Transformers, Streamlit, pytest, matplotlib

## Business Context
Inspired by analytics challenges for social media branding, this project demonstrates how NLP can help marketing teams optimize messaging tone, monitor campaign emotion curves, and identify high-conversion content strategies.

## Note
This code requires a *auth.json* file with your BlueSky credentials to scrape data:
    
    app_password: <your_bluesky_app_password>
    identifier: <your_bluesky_username>.bsky.social
