import os
import sys
import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import base64

# Append root path for loading local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scrape import ScrapeBluesky
from src.posts_to_sentiment import PostsToSentiment

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/labeled_posts_bsky_trekkingpoles.csv")
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
    df['week'] = df['created_at'].dt.to_period("W").astype(str)
    df['week'] = df['week'].str.split('/', expand=True).loc[:, 0]
    return df

emoji_map = {
    "Awareness": "👀 Awareness",
    "Interest": "💡 Interest",
    "Trust": "🤝 Trust",
    "Advocacy": "📣 Advocacy",
    "Drop-Off": "💔 Drop-Off"
}

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Sidebar Filters ---
if "user_query" not in st.session_state:
    st.session_state.user_query = "trekking poles"
    df = load_data()
if "df" not in st.session_state:
    st.session_state.df = df  # Default to full dataset

platforms = st.sidebar.multiselect("Platform",
                                   options=st.session_state.df["platform"].unique(),
                                   default=st.session_state.df["platform"].unique())
stages = st.sidebar.multiselect("Funnel Stage",
                                options=st.session_state.df["funnel_stage"].unique(),
                                default=st.session_state.df["funnel_stage"].unique())
emotions = st.sidebar.multiselect("Emotion Tags",
                                  options=st.session_state.df["emotion"].unique(),
                                  default=st.session_state.df["emotion"].unique())


if st.session_state.submitted:
    with st.spinner("Fetching posts and analyzing funnel stages..."):
        sb = ScrapeBluesky(st.session_state.n_posts_requested,
                           identifier = st.session_state.identifier,
                           app_password = st.session_state.app_password)
        df_tmp, savefid_scrape = sb.scrape(st.session_state.user_query, stream = True,
                                           date_start = st.session_state.start_date,
                                           date_end = st.session_state.end_date)

        ps = PostsToSentiment(savefid_scrape, query = st.session_state.user_query,
                              stream = True, suppress_neutral = True)
        df_tmp, savefid_predict = ps.predict_sentiment(df = df_tmp)
        df_tmp['week'] = df_tmp['created_at'].dt.to_period("W").astype(str)
    #     df_tmp['week'] = df_tmp['week'].str.split('/', expand=True).loc[:, 0]
        st.session_state.df = df_tmp  # Persist after button click 
        st.success("✅ Done! Funnel updated.")

# --- Filtered Data ---
st.session_state.filtered_df = st.session_state.df[
    st.session_state.df["platform"].isin(platforms) &
    st.session_state.df["funnel_stage"].isin(stages) &
    st.session_state.df["emotion"].isin(emotions)
]
        
# --- Config ---
st.set_page_config(page_title="Social Media Sentiment Explorer", layout="wide")

# --- Title ---
st.title("From Likes to Loyalty: Social Media Sentiment Explorer")
st.markdown("Explore how social media posts flow from awareness to advocacy using NLP and emotion classification.")

st.markdown("<p style='font-size:22px'>Current query: <b>" + st.session_state.user_query + "</b></p>", unsafe_allow_html=True)

# --- Stage Distribution ---
st.subheader("Funnel Stage Visualization Mode")
plot_mode = st.radio("Choose plot type:", ["Cumulative Trends", "Weekly Counts"], horizontal=True)

# Consistent colors across funnel values
funnel_colors = {
    "👀 Awareness": "#1f77b4",     # blue
    "💡 Interest": "#ff7f0e",      # orange
    "🤝 Trust": "#2ca02c",         # green
    "📣 Advocacy": "#9467bd",      # purple
    "💔 Drop-Off": "#d62728"       # red
}

if plot_mode == "Weekly Counts":
    stage_counts = st.session_state.filtered_df.groupby(["week", "funnel_stage"]).size().reset_index(name="count")
    stage_counts["funnel_stage"] = stage_counts["funnel_stage"].map(emoji_map)

    fig = px.bar(stage_counts,
                 x="week",
                 y="count",
                 color="funnel_stage",
                 color_discrete_map=funnel_colors,
                 category_orders={"funnel_stage": funnel_colors.keys()},
                 title="Weekly Funnel Stage Breakdown",
                 labels={"count": "Message Count"},
                 height=500)
    st.plotly_chart(fig, use_container_width=True)

elif plot_mode == "Cumulative Trends":
    stage_counts = st.session_state.filtered_df.groupby(["week", "funnel_stage"]).size().unstack(fill_value=0)
    cumulative_counts = stage_counts.cumsum()
    cumulative_long = cumulative_counts.reset_index().melt(id_vars="week",
                                                           var_name="funnel_stage",
                                                           value_name="cumulative_count")
    cumulative_long["funnel_stage"] = cumulative_long["funnel_stage"].map(emoji_map)

    fig = px.bar(cumulative_long,
                 x="week",
                 y="cumulative_count",
                 color="funnel_stage",
                 color_discrete_map=funnel_colors,
                 category_orders={"funnel_stage": funnel_colors.keys()},
                 title="Cumulative Funnel Stage Volume Over Time",
                 labels={"cumulative_count": "Cumulative Count"},
                 height=500)
    st.plotly_chart(fig, use_container_width=True)


# --- Funnel conversion ---
# Pre-processed funnel stage weekly counts
stage_counts = st.session_state.filtered_df.groupby(["week", "funnel_stage"]).size().unstack(fill_value=0)

# Ensure columns exist
for stage in ["Awareness", "Interest", "Trust", "Advocacy"]:
    if stage not in stage_counts.columns:
        stage_counts[stage] = 0

# Compute weekly conversion ratios
stage_counts["Interest_Rate"] = stage_counts["Interest"] / stage_counts["Awareness"]
stage_counts["Trust_Rate"] = stage_counts["Trust"] / stage_counts["Interest"]
stage_counts["Advocacy_Rate"] = stage_counts["Advocacy"] / stage_counts["Trust"]

# Replace NaN or infinite with 0
stage_counts.fillna(0, inplace=True)
stage_counts.replace([float("inf"), float("-inf")], 0, inplace=True)

# Melt for plotting
ratios_melted = stage_counts.reset_index()[["week", "Interest_Rate", "Trust_Rate", "Advocacy_Rate"]]
ratios_long = ratios_melted.melt(id_vars="week", var_name="conversion_step", value_name="rate")

# Apply emojis to melted conversion step column
ratios_long["conversion_step"] = ratios_long["conversion_step"].replace({
    "Interest_Rate": f"{emoji_map['Interest']} / {emoji_map['Awareness']}",
    "Trust_Rate": f"{emoji_map['Trust']} / {emoji_map['Interest']}",
    "Advocacy_Rate": f"{emoji_map['Advocacy']} / {emoji_map['Trust']}"
})

# Find annotation targets
peak_week = ratios_long.loc[ratios_long["rate"].idxmax()]
low_trust = ratios_long.query("conversion_step.str.contains('Trust')").sort_values("rate").iloc[0]

annotations = [
    dict(x=peak_week["week"], y=peak_week["rate"], text="🔥 Peak Conversion", showarrow=True, arrowhead=1),
#     dict(x=low_trust["week"], y=low_trust["rate"], text="⚠️ Drop in Trust", showarrow=True, arrowhead=1)
]

conversion_colors = {
    f"{emoji_map['Interest']} / {emoji_map['Awareness']}": "#ff7f0e",     # blue
    f"{emoji_map['Trust']} / {emoji_map['Interest']}": "#2ca02c",      # orange
    f"{emoji_map['Advocacy']} / {emoji_map['Trust']}": "#9467bd"         # green
}

st.subheader("Funnel Conversion")

fig = px.line(
    ratios_long,
    x="week",
    y="rate",
    color="conversion_step",
    color_discrete_map=conversion_colors,
    markers=True,
    labels={"rate": "Conversion Rate", "week": "Week"},
    title="🚦 Weekly Funnel Conversion Trends",
    height=500,
    category_orders={"conversion_step": [val for val in conversion_colors if val in ratios_long["conversion_step"]]}
)

fig.update_layout(
    yaxis=dict(tickformat=".0%"),
    annotations=annotations,
    legend_title_text="Funnel Transition",
)
st.plotly_chart(fig, use_container_width=True)

# --- Sample Messages ---
st.subheader("Sample Messages")
sample = st.session_state.filtered_df.sample(n=min(5, len(st.session_state.filtered_df)))
for _, row in sample.iterrows():
    st.markdown(f"**{row['platform']} · {emoji_map[row['funnel_stage']]} · {row['emotion']}**")
    st.write(row["text"].split('Post: ')[1] if 'Post: ' in row["text"] else row["text"])
    st.markdown("---")
    
# --- Query Interface Form ---

# with st.expander("🔎 Search Posts by Custom Query", expanded=True):
with st.form("query_form"):
    st.subheader("🔎 Analyze Custom Query")
    st.markdown("Use this form to scrape and analyze sentiment of a custom query.")

    st.session_state.user_query = st.text_input("Enter query keyword(s):", value=st.session_state.user_query)
    st.session_state.identifier = st.text_input("Bluesky username (optional):", value=None)
    st.session_state.app_password = st.text_input("Bluesky app password (optional):", value=None, type="password")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.text_input("Start Date (YYYY-MM-DD):", value="2025-01-01")
    with col2:
        st.session_state.end_date = st.text_input("End Date (YYYY-MM-DD):", value="2025-07-31")

    st.session_state.n_posts_requested = st.number_input("Number of posts to retrieve:", min_value=1, step=1, value=1000)
    
    st.session_state.submitted = st.form_submit_button("🚀 Analyze Query")
