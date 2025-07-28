import streamlit as st
import pandas as pd
import plotly.express as px
import re
from datetime import datetime
import base64

# Streamlit page config and CSS
st.set_page_config(page_title="Log Dashboard with AI-Bot Detection", page_icon="🤖", layout="wide")
st.markdown("""
<style>
body { background: #f8fafd;}
.metric-container { background-color: #fff4ef; border-radius: .4rem; margin: 0.2rem 0;padding: 1rem;}
.llm-highlight { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Crawler patterns (abbreviated for space, expand as needed)
CRAWLER_PATTERNS = {
    'OpenAI': [r'GPTBot', r'OAI-SearchBot', r'ChatGPT-User', r'OpenAI', r'GPT-4', r'GPT-3'],
    'Anthropic': [r'ClaudeBot', r'Claude-Web', r'AnthropicBot', r'Claude'],
    # ... (add more patterns for full detection)
    'Other Crawlers': [r'bot(?!AI)', r'spider', r'crawler', r'scraper', r'curl', r'wget', r'python-requests']
}
LLM_AI_CATEGORIES = ['OpenAI', 'Anthropic']  # expand as needed

# Fixed LOG_PATTERNS for all main formats
LOG_PATTERNS = {
    'Apache Combined': r'^(?P<ip>\S+) \S+ \S+ $(?P<timestamp>[^$]+)$ "(?P<request>[^"]*)" (?P<status>\d{3}) (?P<size>\d+|-) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"$',
    'Apache Common':   r'^(?P<ip>\S+) \S+ \S+ $(?P<timestamp>[^$]+)$ "(?P<request>[^"]*)" (?P<status>\d{3}) (?P<size>\d+|-)$',
    'Nginx':           r'^(?P<ip>\S+) - \S+ $(?P<timestamp>[^$]+)$ "(?P<request>[^"]*)" (?P<status>\d{3}) (?P<size>\d+|-) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"$',
    # IIS CSV-like format, basic support (many IIS logs start with a #Fields: header, so more handling may be needed)
    'IIS':             r'^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<s_ip>\S+) (?P<cs_method>\S+) (?P<cs_uri_stem>\S+) (?P<cs_uri_query>\S+) (?P<s_port>\d+) (?P<cs_username>\S+) (?P<c_ip>\S+) (?P<user_agent>.*?) (?P<referer>.*?) (?P<status>\d{3}) (?P<substatus>\d+) (?P<win32status>\d+) (?P<time_taken>\d+)'
}

def identify_crawler(user_agent):
    if pd.isna(user_agent) or user_agent == '-':
        return 'Unknown'
    user_agent = str(user_agent)
    for crawler_type in LLM_AI_CATEGORIES:
        for pattern in CRAWLER_PATTERNS.get(crawler_type, []):
            if re.search(pattern, user_agent, re.IGNORECASE):
                return crawler_type
    for crawler_type, patterns in CRAWLER_PATTERNS.items():
        if crawler_type not in LLM_AI_CATEGORIES:
            for pattern in patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    return crawler_type
    return 'Human/Other'

def categorize_crawler_type(crawler_type):
    if crawler_type in LLM_AI_CATEGORIES:
        return 'LLM/AI Bots'
    elif crawler_type in ['Other Crawlers']:
        return 'Other Bots'
    elif crawler_type == 'Human/Other':
        return 'Human Traffic'
    else:
        return 'Unknown'

def parse_log_line(line, log_format):
    pattern = LOG_PATTERNS.get(log_format)
    if not pattern:
        return None
    match = re.match(pattern, line.strip())
    if not match:
        return None
    d = match.groupdict()
    if "request" in d:
        method, url, protocol = "", "", ""
        if d['request']:
            parts = d['request'].split()
            if len(parts) == 3:
                method, url, protocol = parts
            elif len(parts) == 2:
                method, url = parts
        return {
            "ip": d.get("ip", "") or d.get("c_ip", ""),
            "timestamp": d.get("timestamp", "") or f"{d.get('date','')} {d.get('time','')}",
            "method": method or d.get("cs_method", ""),
            "url": url or d.get("cs_uri_stem", ""),
            "protocol": protocol,
            "status": d.get("status", ""),
            "size": d.get("size", ""),
            "referer": d.get("referer", ""),
            "user_agent": d.get("user_agent", "")
        }
    elif log_format == 'IIS':
        return {
            "ip": d.get("c_ip", ""),
            "timestamp": f"{d.get('date','')} {d.get('time','')}",
            "method": d.get("cs_method", ""),
            "url": d.get("cs_uri_stem", ""),
            "protocol": "",
            "status": d.get("status", ""),
            "size": "",
            "referer": d.get("referer", ""),
            "user_agent": d.get("user_agent", "")
        }
    return None

def parse_timestamp(timestamp_str, log_format):
    try:
        if log_format in ['Apache Combined', 'Apache Common', 'Nginx']:
            # e.g. 07/Jan/2025:00:24:17 +0000
            if " " in timestamp_str:
                dt_str = timestamp_str.split(" ")[0]
            else:
                dt_str = timestamp_str
            return datetime.strptime(dt_str, '%d/%b/%Y:%H:%M:%S')
        elif log_format == 'IIS':
            # e.g. 2025-01-07 00:24:17
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None

def debug_log_parsing(line, log_format):
    st.write(f"**Trying format:** {log_format}")
    st.write(f"**Sample line:** `{line[:200]}...`" if len(line) > 200 else f"**Sample line:** `{line}`")
    pattern = LOG_PATTERNS.get(log_format)
    if pattern:
        match = re.match(pattern, line.strip())
        if match:
            st.success(f"✅ **Pattern matched!** Groups found: {len(match.groups())}")
            return True
        else:
            st.error(f"❌ **Pattern didn't match**")
            return False
    return False

@st.cache_data
def process_log_file(file_content, log_format, _progress_bar=None):
    try:
        lines = file_content.decode('utf-8').splitlines()
    except UnicodeDecodeError:
        try:
            lines = file_content.decode('latin-1').splitlines()
        except Exception:
            return None, 0
    st.subheader("🔍 Debug Information")
    st.write(f"**Total lines in file:** {len(lines)}")
    st.write(f"**Selected format:** {log_format}")
    sample_lines = [line for line in lines[:10] if line.strip()][:3]
    if sample_lines:
        st.write("**First 3 sample lines:**")
        for i, line in enumerate(sample_lines, 1):
            with st.expander(f"Line {i} - Click to test parsing"):
                debug_log_parsing(line, log_format)
    total_lines = len(lines)
    parsed_data = []
    failed_lines = 0
    for i, line in enumerate(lines):
        if _progress_bar and i % 1000 == 0:
            _progress_bar.progress(i / total_lines)
        if line.strip():
            parsed = parse_log_line(line, log_format)
            if parsed:
                parsed_data.append(parsed)
            else:
                failed_lines += 1
                if failed_lines <= 3:
                    st.warning(f"Failed to parse line {i+1}: `{line[:100]}...`")
    if _progress_bar:
        _progress_bar.progress(1.0)
    st.write(f"**Parsing results:** {len(parsed_data)} successful, {failed_lines} failed")
    if not parsed_data:
        return None, failed_lines
    df = pd.DataFrame(parsed_data)
    df['datetime'] = df['timestamp'].apply(lambda x: parse_timestamp(x, log_format))
    df = df.dropna(subset=['datetime'])
    df['crawler_type'] = df['user_agent'].apply(identify_crawler)
    df['crawler_category'] = df['crawler_type'].apply(categorize_crawler_type)
    df['status'] = pd.to_numeric(df['status'], errors='coerce')
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['size'] = df['size'].fillna(0)
    return df, failed_lines

def create_llm_ai_highlight_chart(df):
    llm_df = df[df['crawler_category'] == 'LLM/AI Bots']
    if llm_df.empty:
        return None
    counts = llm_df['crawler_type'].value_counts()
    fig = px.bar(
        x=counts.values, y=counts.index, orientation='h',
        title='🤖 LLM/AI Bot Activity', labels={'x': 'Request Count', 'y': 'AI Bot Type'},
        color=counts.values, color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    return fig

def create_category_comparison_chart(df):
    cats = df['crawler_category'].value_counts()
    color_map = {
        'LLM/AI Bots': '#ff6b6b', 'Other Bots': '#ffeaa7',
        'Human Traffic': '#dda0dd', 'Unknown': '#cccccc'
    }
    colors = [color_map.get(cat, '#cccccc') for cat in cats.index]
    fig = px.pie(
        values=cats.values, names=cats.index, title='Traffic by Category',
        color_discrete_sequence=colors
    )
    fig.update_layout(height=400)
    return fig

def create_top_crawlers_chart(df):
    counts = df['crawler_type'].value_counts().head(10)
    colors = ['#ff6b6b' if c in LLM_AI_CATEGORIES else '#4ecdc4' for c in counts.index]
    fig = px.bar(
        x=counts.values, y=counts.index, orientation='h',
        title='Top 10 Crawlers', labels={'x': 'Requests', 'y': 'Crawler'},
        color=colors, color_discrete_map="identity"
    )
    fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    return fig

def create_status_codes_by_crawler_chart(df):
    top_crawlers = df['crawler_type'].value_counts().head(6).index
    filtered = df[df['crawler_type'].isin(top_crawlers)]
    crosstab = pd.crosstab(filtered['crawler_type'], filtered['status'])
    fig = px.bar(
        crosstab, title='Status by Crawler', barmode='stack',
        labels={'value': 'Requests', 'crawler_type': 'Crawler'}, color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400, xaxis_tickangle=-45, legend_title='Status Code')
    return fig

def create_top_urls_chart(df):
    counts = df['url'].value_counts().head(10)
    short_urls = [url[:40] + '...' if len(url) > 40 else url for url in counts.index]
    fig = px.bar(
        x=counts.values, y=short_urls, orientation='h',
        title='Top 10 URLs', labels={'x': 'Requests', 'y': 'URL'},
        color=counts.values, color_continuous_scale='plasma'
    )
    fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    return fig

def create_timeline_chart(df):
    df['date'] = df['datetime'].dt.date
    data = df.groupby(['date', 'crawler_category']).size().reset_index(name='count')
    color_map = {'LLM/AI Bots': '#ff6b6b', 'Other Bots': '#ffeaa7', 'Human Traffic': '#dda0dd'}
    fig = px.line(
        data, x='date', y='count', color='crawler_category',
        title='Timeline by Category', labels={'count': 'Requests', 'date': 'Date'},
        color_discrete_map=color_map
    )
    fig.update_layout(height=400)
    return fig

def create_status_pie_chart(df):
    counts = df['status'].value_counts()
    fig = px.pie(
        values=counts.values, names=counts.index, title='Status Code Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(height=400)
    return fig

def get_download_link(df, filename="filtered_logs.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def main():
    st.title("🤖 Log File Analysis Dashboard")
    st.markdown(
        "Analyze server logs for AI bots and other crawlers. "
        "Supports Apache, Nginx, IIS formats. See sidebar for file upload and log format selection."
    )
    with st.sidebar:
        st.header("File & Format")
        uploaded_file = st.file_uploader("Upload Log File", type=['txt', 'log'])
        log_format = st.selectbox("Log Format", list(LOG_PATTERNS.keys()), index=0)
        if uploaded_file:
            st.info(f"Loaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

    if uploaded_file:
        with st.spinner("Analyzing log file..."):
            progress_bar = st.progress(0)
            df, failed_lines = process_log_file(
                uploaded_file.getvalue(), log_format, _progress_bar=progress_bar
            )
            progress_bar.empty()
            if df is None or df.empty:
                st.error("No valid log entries found. Try a different format or check file contents.")
                return
            if failed_lines > 0:
                st.warning(f"{failed_lines} lines could not be parsed")
            st.success(f"Processed {len(df):,} log entries!")

            # Sidebar filters
            with st.sidebar:
                st.header("Filters")
                min_date, max_date = df['datetime'].min().date(), df['datetime'].max().date()
                date_range = st.date_input("Date Range", value=(min_date, max_date),
                                           min_value=min_date, max_value=max_date)
                status_codes = sorted(df['status'].dropna().unique())
                selected_statuses = st.multiselect(
                    "Status Codes", status_codes, default=status_codes)
                categories = sorted(df['crawler_category'].unique())
                selected_categories = st.multiselect(
                    "Categories", categories, default=categories)
                crawlers = sorted(df['crawler_type'].unique())
                selected_crawlers = st.multiselect(
                    "Crawlers", crawlers, default=crawlers)
            # Filtering
            fdf = df.copy()
            if len(date_range) == 2:
                start, end = date_range
                fdf = fdf[(fdf['datetime'].dt.date >= start) & (fdf['datetime'].dt.date <= end)]
            if selected_statuses:
                fdf = fdf[fdf['status'].isin(selected_statuses)]
            if selected_categories:
                fdf = fdf[fdf['crawler_category'].isin(selected_categories)]
            if selected_crawlers:
                fdf = fdf[fdf['crawler_type'].isin(selected_crawlers)]
            if fdf.empty:
                st.warning("No entries match the selected filters.")
                return
            # Metrics
            st.header("Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Requests", f"{len(fdf):,}")
            col2.metric("Unique IPs", f"{fdf['ip'].nunique():,}")
            count_llm = fdf[fdf['crawler_category'] == 'LLM/AI Bots'].shape[0]
            col3.metric("🤖 LLM/AI Bot Requests", f"{count_llm:,}",
                        f"{(count_llm/len(fdf))*100:.1f}%")
            bot_count = fdf[fdf['crawler_category'] != 'Human Traffic'].shape[0]
            col4.metric("Other Bot Requests", f"{bot_count:,}",
                        f"{(bot_count/len(fdf))*100:.1f}%")
            # Charts
            st.subheader("AI Bots & Category Overview")
            col1, col2 = st.columns(2)
            with col1:
                chart = create_llm_ai_highlight_chart(fdf)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("No LLM/AI bot activity in current filter selection.")
            with col2:
                st.plotly_chart(create_category_comparison_chart(fdf), use_container_width=True)
            st.subheader("Crawler & Traffic Details")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_top_crawlers_chart(fdf), use_container_width=True)
                st.plotly_chart(create_status_codes_by_crawler_chart(fdf), use_container_width=True)
            with col2:
                st.plotly_chart(create_timeline_chart(fdf), use_container_width=True)
                st.plotly_chart(create_status_pie_chart(fdf), use_container_width=True)
            st.subheader("Top URLs")
            st.plotly_chart(create_top_urls_chart(fdf), use_container_width=True)
            st.subheader("Data Export")
            st.markdown(get_download_link(fdf, "filtered_logs.csv"), unsafe_allow_html=True)
            llm_bots = fdf[fdf['crawler_category'] == 'LLM/AI Bots']
            if not llm_bots.empty:
                st.markdown(get_download_link(llm_bots, "llm_ai_bots.csv"), unsafe_allow_html=True)
            st.subheader("Raw Data")
            st.dataframe(fdf[['datetime', 'ip', 'method', 'url', 'status', 'crawler_type', 'crawler_category']],
                         use_container_width=True)

if __name__ == "__main__":
    main()
