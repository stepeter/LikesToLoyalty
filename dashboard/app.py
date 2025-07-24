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
    dashboard_overview
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

# --- Load Data ---
load_data()

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# --- Sidebar Filters ---
set_sidebar_filters()

if st.session_state.submitted:
    with st.spinner("Fetching posts and analyzing funnel stages..."):
        run_scraper_pipeline()
        st.success("✅ Done! Funnel updated.")

# --- Filtered Data ---
filter_data()

# --- Compute funnel conversions ---
ratios_long = compute_funnel_conversions(emoji_map)
        
# --- Config ---
st.set_page_config(page_title="Social Media Sentiment Explorer", layout="wide")

# --- Title ---
set_dashboard_header()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Funnel Trends",
                                  "💬 Sample Messages",
                                  "🛠️ Create Custom Query",
                                  "📝 Dashboard Overview"])

with tab1:
    # --- Funnel weekly counter ---
    plot_funnel_weekly_counter(emoji_map, funnel_colors)

    # --- Funnel conversion ---
    plot_funnel_conversions(ratios_long, conversion_colors)

with tab2:
    # --- Sample Messages ---
    display_sample_messages(
        df=st.session_state.filtered_df,
        emoji_map=emoji_map
    )

with tab3:
    # --- Query Interface Form ---
    query_interface_form()

with tab4:
    # --- Dashboard Oveview ---
    dashboard_overview()