import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

# Append root path for loading local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scrape import ScrapeBluesky
from src.posts_to_sentiment import PostsToSentiment

def set_dashboard_header() -> None:
    """
    Display the main dashboard header and current query info using Streamlit components.

    This function sets the app title and a subtitle describing the dashboard's purpose.
    It also renders the current user query from session state in a styled HTML box.

    Requirements:
        - st.session_state.user_query (str): Must exist before calling the function.

    Returns:
        None
    """
    st.title("From Likes to Loyalty: Social Media Sentiment Explorer")
    st.markdown("Explore how social media posts flow from awareness to advocacy using NLP and emotion classification.")

#     st.write("")
#     st.write("")

@st.cache_data
def load_data(fid: str = "data/processed/labeled_posts_bsky_trekkingpoles.csv") -> pd.DataFrame:
    """
    Load labeled social media sentiment data and compute 'week' periods for aggregation.

    Args:
        fid (str): Path to the CSV file containing post-level sentiment data.

    Returns:
        pd.DataFrame: DataFrame with parsed 'created_at' timestamps and a simplified 'week' column.

    Notes:
        - Assumes 'created_at' is present and parseable by pandas.
        - Strips end of ISO week ranges (e.g., '2024-06-17/2024-06-23' → '2024-06-17')
    """
    df = pd.read_csv(fid)
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
    df['week'] = df['created_at'].dt.to_period("W").astype(str)
    df['week'] = df['week'].str.split('/', expand=True).loc[:, 0]
    return df

def filter_data() -> None:
    """
    Filter the dashboard's DataFrame based on selected platform, funnel stage, and emotion.

    Updates:
        st.session_state.filtered_df: A filtered version of `st.session_state.df`.

    Requirements:
        - st.session_state.df (pd.DataFrame) must be initialized.
        - st.session_state.platforms, stages, and emotions must exist and be iterable.

    Returns:
        None
    """
    st.session_state.filtered_df = st.session_state.df[
        st.session_state.df["platform"].isin(st.session_state.platforms) &
        st.session_state.df["funnel_stage"].isin(st.session_state.stages) &
        st.session_state.df["emotion"].isin(st.session_state.emotions)
    ]

def set_sidebar_filters() -> None:
    """
    Initialize or update dashboard filter controls for platform, funnel stage, and emotion.

    - Loads default query and data if missing in session state.
    - Adds multiselect filter widgets to the sidebar.
    - Stores selected filter values in session state.

    Session Keys Used:
        - user_query (str): Query term for scraping.
        - df (pd.DataFrame): Loaded or scraped dataset.
        - platforms (List[str]): Selected platforms.
        - stages (List[str]): Selected funnel stages.
        - emotions (List[str]): Selected emotion tags.

    Returns:
        None
    """
    st.sidebar.title("Filter Controls")
    st.session_state.platforms = st.sidebar.multiselect("Platform",
                                       options=st.session_state.df["platform"].unique(),
                                       default=st.session_state.df["platform"].unique())
    st.session_state.stages = st.sidebar.multiselect("Funnel Stage",
                                    options=st.session_state.df["funnel_stage"].unique(),
                                    default=st.session_state.df["funnel_stage"].unique())
    st.session_state.emotions = st.sidebar.multiselect("Emotion Tags",
                                      options=st.session_state.df["emotion"].unique(),
                                      default=st.session_state.df["emotion"].unique())

def display_sample_messages(df: pd.DataFrame, emoji_map: dict[str, str]) -> None:
    """
    Display a random sample of social media messages with funnel/emotion tagging.

    Args:
        df (pd.DataFrame): Filtered messages with funnel stage and emotion columns.
        emoji_map (dict[str, str]): Mapping from funnel stage to corresponding emoji.

    Side Effects:
        - Updates st.session_state.new_sample with a new sample if refreshed.
        - Displays message metadata and text.

    Returns:
        None
    """
    st.subheader("Sample Messages")

    if st.button("🔁 Refresh Messages") or st.session_state.submitted:
        st.session_state.new_sample = df.sample(n=min(5, len(df)))

    if "new_sample" not in st.session_state:
        st.session_state.new_sample = df.sample(n=min(5, len(df)))

    sample = st.session_state.new_sample

    for _, row in sample.iterrows():
        st.markdown(f"**{row['platform']} · {emoji_map[row['funnel_stage']]} · {row['emotion']}**")
        st.write(row["text"].split('Post: ')[1] if 'Post: ' in row["text"] else row["text"])
        st.markdown("---")
        
def run_scraper_pipeline() -> None:
    """
    Execute the Bluesky scraping and sentiment prediction pipeline.

    Workflow:
        - Scrapes posts based on session query and date range.
        - Applies sentiment/emotion analysis via NLP model.
        - Augments with weekly aggregation label.

    Side Effects:
        - Saves processed posts to st.session_state.df.

    Returns:
        None
    """
    sb = ScrapeBluesky(
        n_posts_requested=st.session_state.n_posts_requested,
        identifier=st.session_state.identifier,
        app_password=st.session_state.app_password
    )

    df_tmp, savefid_scrape = sb.scrape(
        query=st.session_state.user_query,
        stream=True,
        date_start=st.session_state.start_date,
        date_end=st.session_state.end_date
    )

    ps = PostsToSentiment(
        savefid_scrape,
        query=st.session_state.user_query,
        stream=True,
        suppress_neutral=True
    )

    df_tmp, savefid_predict = ps.predict_sentiment(df=df_tmp)

    # Add weekly aggregation label
    df_tmp["week"] = df_tmp["created_at"].dt.to_period("W").astype(str)

    # Persist to session state
    st.session_state.df = df_tmp
    
def plot_funnel_weekly_counter(
    emoji_map: dict[str, str],
    funnel_colors: dict[str, str],
    plot_mode: str = "Weekly Counts"
) -> px.bar:
    """
    Visualize funnel progression as either weekly counts or cumulative volume.

    Args:
        emoji_map (dict[str, str]): Funnel stage to emoji label mapping.
        funnel_colors (dict[str, str]): Funnel stage to color mapping.
        plot_mode (str): Default plot type ("Weekly Counts" or "Cumulative Trends").

    Returns:
        None or plotly.graph_objects.Figure (if cumulative plot is shown)
    """
    st.subheader("Funnel Stage Visualization Mode")
    st.write("Note that funnel stages are based on manual mapping from predicted sentiment")
    plot_mode = st.radio("Choose plot type:", ["Cumulative Trends", "Weekly Counts"], horizontal=True)
    
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
    return fig
    
def compute_funnel_conversions(emoji_map: dict[str, str]) -> pd.DataFrame:
    """
    Compute weekly funnel conversion ratios (Interest, Trust, Advocacy).

    Args:
        emoji_map (dict[str, str]): Mapping from funnel stage to emoji symbol.

    Returns:
        pd.DataFrame: Melted long-format DataFrame with weekly conversion rates and labels.
    """
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
    return ratios_long

def plot_funnel_conversions(
    ratios_long: pd.DataFrame,
    conversion_colors: dict[str, str]
) -> px.line:
    """
    Plot conversion rates across funnel transitions over time.

    Args:
        ratios_long (pd.DataFrame): Melted conversion rate DataFrame.
        conversion_colors (dict[str, str]): Mapping from funnel step to color.

    Returns:
        plotly.graph_objects.Figure: Line chart visualization.
    """
    st.subheader("Aggregate Funnel Conversion")
    st.write("⚠️ Warning: Conversion rates are computed over the aggregate, not individual users and may therefore be inaccurate especially for low sample sizes")

    peak_week = ratios_long.loc[ratios_long["rate"].idxmax()]
    low_trust = ratios_long.query("conversion_step.str.contains('Trust')").sort_values("rate").iloc[0]

    annotations = [
        dict(x=peak_week["week"], y=peak_week["rate"], text="🔥 Peak Conversion", showarrow=True, arrowhead=1)
    ]
    
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
    return fig

def query_interface_form() -> None:
    """
    Render a form to customize scraping parameters and trigger NLP pipeline.

    Inputs:
        - Query string
        - Optional Bluesky credentials
        - Date range and post count

    Side Effects:
        - Updates session state with scraping parameters.
        - Sets `submitted` flag if user initiates the form.

    Returns:
        None
    """
    with st.form("query_form"):
        st.subheader("🔎 Analyze Custom Query")
        st.markdown("Use this form to scrape and analyze sentiment of a custom query.")

        user_query = st.text_input("Enter query keyword(s):", value=st.session_state.user_query)
        identifier = st.text_input("Bluesky username:", value=None)
        app_password = st.text_input("Bluesky app password:", value=None, type="password")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.text_input("Start Date (YYYY-MM-DD):", value="2025-01-01")
        with col2:
            end_date = st.text_input("End Date (YYYY-MM-DD):", value="2025-07-31")

        n_posts_requested = st.number_input("Number of posts to retrieve:",
                                            min_value=1, step=1, value=st.session_state.n_posts_requested)

        submitted = st.form_submit_button("🚀 Analyze Query")
        
    if submitted:
        st.session_state.submitted = submitted
        st.session_state.user_query = user_query
        st.session_state.identifier = identifier
        st.session_state.app_password = app_password
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.n_posts_requested = n_posts_requested
        
def set_initial_states() -> None:
    """
    Sets initial states for dashboard
    
    Returns:
        None
    
    """
    if "user_query" not in st.session_state:
        st.session_state.user_query = "trekking poles"
        df = load_data()
        if "df" not in st.session_state:
            st.session_state.df = df  # Default to full dataset
    if "identifier" not in st.session_state:
        st.session_state.identifier = ""
    if "app_password" not in st.session_state:
        st.session_state.app_password = ""
    if "start_date" not in st.session_state:
        st.session_state.start_date = "2025-01-01"
    if "end_date" not in st.session_state:
        st.session_state.end_date = "2025-07-31"
    if "n_posts_requested" not in st.session_state:
        st.session_state.n_posts_requested = 1000
        
def dashboard_overview() -> None:
    """
    Text describing the dashboard, its functionality, business motivation, and limitations.

    Returns:
        None
    """
    st.header("📝 Dashboard Purpose & Guide")
    st.markdown("""
    This dashboard helps you explore how social media posts evolve from awareness to brand advocacy using NLP emotion classification and funnel analysis.
    
    <p style='font-size:18px'>
        🚀 <a href='https://github.com/stepeter/LikesToLoyalty' target='_blank'>View GitHub Project</a>
    </p>

    ### 🔍 Functionality Highlights
    - **Query Sentiment Pipeline**: Scrapes posts from Bluesky, performs emotion tagging via NLP, and adds funnel stage metadata.
    - **Filter Controls**: Select platform, funnel stage, and emotion tags to zoom in on relevant data slices.
    - **Visual Funnels**: Toggle between *weekly breakdown* or *cumulative sentiment flows* across funnel stages.
    - **Conversion Metrics**: Track how efficiently posts move through stages (e.g. Awareness → Interest → Trust → Advocacy).
    - **Sample Explorer**: View real posts that correspond to each stage, with refreshable samples.
    - **Custom Queries**: Perform real-time analyses of custom queries on Bluesky posts.

    ### 💼 Business Motivation
    - **Brand Intelligence**: Understand not just what people say, but *how they feel* at each engagement stage.
    - **Campaign Performance**: Gauge emotional resonance and conversion barriers in social media messaging.
    - **Product Feedback Loop**: Spot trust gaps and emotional drop-offs that signal usability, messaging, or perception issues.

    ### ⚠️ Known Limitations
    - **Platform Scope**: Currently limited to Bluesky scraping.
    - **Emotion Detection**: May miss nuanced or sarcastic tones; NLP model often focuses on overall post sentiment instead of focusing on sentiment towards the query.
    - **Aggregate Conversion**: Conversion calculations are based on aggregate weekly results and may not have consistent users from one week to the next, leading to inaccurate results, particularly with small sample sizes.
    - **Time Granularity**: Aggregation is weekly; daily views or trend smoothing are not available.
    - **Query Stream Size**: Rate limits and app credential constraints may cap post volume for larger campaigns. Sentiment prediction is also slow because CPU is used.

    ---
    *Built for marketers, researchers, and developers who want to translate social signal noise into actionable insight.*
    """, unsafe_allow_html=True)
        