import streamlit as st

from app_utils import (
    display_sample_messages,
    run_scraper_pipeline,
    plot_funnel_weekly_counter,
    compute_funnel_conversions,
    plot_funnel_conversions,
    query_interface_form,
    set_dashboard_header,
    load_data,
    filter_data,
    set_sidebar_filters,
    dashboard_overview,
    set_initial_states
)

emoji_map = {
    "Awareness": "👀 Awareness",
    "Interest": "💡 Interest",
    "Trust": "🤝 Trust",
    "Advocacy": "📣 Advocacy",
    "Drop-Off": "💔 Drop-Off"
}

# Consistent colors across funnel values
funnel_colors = {
    "👀 Awareness": "#1f77b4",     # blue
    "💡 Interest": "#ff7f0e",      # orange
    "🤝 Trust": "#2ca02c",         # green
    "📣 Advocacy": "#9467bd",      # purple
    "💔 Drop-Off": "#d62728"       # red
}

conversion_colors = {
    f"{emoji_map['Interest']} / {emoji_map['Awareness']}": "#ff7f0e",     # blue
    f"{emoji_map['Trust']} / {emoji_map['Interest']}": "#2ca02c",      # orange
    f"{emoji_map['Advocacy']} / {emoji_map['Trust']}": "#9467bd"         # green
}

# --- Config ---
st.set_page_config(page_title="Social Media Sentiment Explorer", layout="wide")
set_initial_states()

set_dashboard_header()
markdown_placeholder = st.empty()
st.write("")
st.write("")

# --- Query Submission Form First ---
tab_labels = ["📝 Dashboard Overview", "📊 Funnel Analysis", "💬 Sample Messages", "🛠️ Create Custom Query"]
tabs = st.tabs(tab_labels)

with tabs[3]:  # "🛠️ Create Custom Query"
    query_interface_form()

# --- Run Scraper If Submitted ---
if "submitted" not in st.session_state:
    st.session_state.submitted = False

if st.session_state.submitted:
    with st.spinner("Fetching posts and analyzing sentiment..."):
        run_scraper_pipeline()
        st.success("✅ Done! Sentiment updated.")  

# --- Continue with Setup ---
load_data()
set_sidebar_filters()
filter_data()
ratios_long = compute_funnel_conversions(emoji_map)

with tabs[1]:  # Funnel Trends
    plot_funnel_weekly_counter(emoji_map, funnel_colors)
    plot_funnel_conversions(ratios_long, conversion_colors)

with tabs[2]:  # Sample Messages
    display_sample_messages(df=st.session_state.filtered_df, emoji_map=emoji_map, funnel_colors=funnel_colors)

with tabs[0]:  # Overview
    dashboard_overview()
    
if st.session_state.submitted:
    st.session_state.submitted = False  # Optional: Reset after run
    
markdown_placeholder.markdown(f"""
    <div style="
        background-color:#2C2F33;
        border:1px solid #ccc;
        padding:10px;
        border-radius:5px;
        font-size:20px;
    ">
    🔍 <b>Current query:</b> {st.session_state.user_query}
    </div>
""", unsafe_allow_html=True)
