import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import io
import base64
from collections import Counter
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Self Log - Advanced Log File Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding: 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stSelectbox label, .stMultiselect label {
        font-weight: 600;
        color: #1f77b4;
    }
    h1 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    h2, h3 {
        color: #2c3e50;
    }
    .llm-highlight {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Comprehensive crawler patterns with LLM/AI bots
CRAWLER_PATTERNS = {
    # LLM/AI Crawlers - Most Important Category
    'OpenAI': [
        r'GPTBot',
        r'OAI-SearchBot', 
        r'ChatGPT-User',
        r'OpenAI',
        r'GPT-4',
        r'GPT-3'
    ],
    'Anthropic': [
        r'ClaudeBot',
        r'Claude-Web',
        r'AnthropicBot',
        r'Anthropic',
        r'Claude'
    ],
    'Google AI': [
        r'Google-Extended',
        r'Bard',
        r'Gemini',
        r'GoogleAI',
        r'Google-AI',
        r'BardBot'
    ],
    'Meta AI': [
        r'Meta-ExternalAgent',
        r'Meta-ExternalFetcher',
        r'FacebookBot-AI',
        r'MetaAI',
        r'Meta-AI'
    ],
    'Perplexity': [
        r'PerplexityBot',
        r'Perplexity',
        r'PerplexityAI'
    ],
    'Microsoft': [
        r'BingAI',
        r'EdgeAI',
        r'CopilotBot',
        r'MicrosoftBot'
    ],
    'Character.AI': [
        r'CharacterBot',
        r'Character\.AI',
        r'CharacterAI'
    ],
    'Inflection': [
        r'PiBot',
        r'InflectionBot',
        r'Inflection'
    ],
    'Cohere': [
        r'CohereBot',
        r'Cohere'
    ],
    'AI21': [
        r'AI21Bot',
        r'AI21'
    ],
    'Hugging Face': [
        r'HuggingFaceBot',
        r'HuggingFace',
        r'HF-Bot'
    ],
    'Replicate': [
        r'ReplicateBot',
        r'Replicate'
    ],
    'Stability AI': [
        r'StabilityBot',
        r'StabilityAI',
        r'Stability-AI'
    ],
    'You.com': [
        r'YouBot',
        r'You\.com',
        r'YouAI'
    ],
    'Neeva': [
        r'NeevaBot',
        r'Neeva'
    ],
    'Kagi': [
        r'KagiBot',
        r'Kagi'
    ],
    'Brave': [
        r'BraveBot',
        r'Brave-AI',
        r'BraveAI'
    ],
    'DuckDuckGo AI': [
        r'DuckAssistBot',
        r'DuckDuckGo-AI',
        r'DuckAI'
    ],
    'Other AI Bots': [
        r'AI-Bot',
        r'AIBot',
        r'LLMBot',
        r'ChatBot',
        r'AssistantBot',
        r'LanguageModel',
        r'NeuralBot'
    ],
    
    # Traditional Search Engine Crawlers
    'Googlebot': [
        r'Googlebot(?!-AI)',  # Exclude AI variants
        r'GoogleBot(?!-AI)',
        r'Google-InspectionTool',
        r'GoogleOther',
        r'GoogleImageProxy'
    ],
    'Bingbot': [
        r'bingbot(?!AI)',  # Exclude AI variants
        r'BingBot(?!AI)',
        r'MSNBot',
        r'msnbot'
    ],
    'Yahoo': [
        r'Slurp',
        r'yahoo',
        r'YahooSeeker'
    ],
    'Baidu': [
        r'Baiduspider',
        r'baidu'
    ],
    'Yandex': [
        r'YandexBot',
        r'Yandex'
    ],
    'DuckDuckGo': [
        r'DuckDuckBot(?!-AI)',  # Exclude AI variants
        r'DuckDuckGo(?!-AI)'
    ],
    
    # Social Media Crawlers
    'Facebook': [
        r'facebookexternalhit(?!.*AI)',  # Exclude AI variants
        r'FacebookBot(?!-AI)'
    ],
    'Twitter': [
        r'Twitterbot',
        r'TwitterCardBot'
    ],
    'LinkedIn': [
        r'LinkedInBot',
        r'linkedin'
    ],
    'WhatsApp': [
        r'WhatsApp'
    ],
    'Telegram': [
        r'TelegramBot'
    ],
    'Apple': [
        r'Applebot'
    ],
    'Amazon': [
        r'AmazonBot'
    ],
    'Pinterest': [
        r'Pinterest'
    ],
    
    # SEO and Analytics Tools
    'Semrush': [
        r'SemrushBot',
        r'semrush'
    ],
    'Ahrefs': [
        r'AhrefsBot',
        r'ahrefs'
    ],
    'Moz': [
        r'MozBot',
        r'rogerbot'
    ],
    'Majestic': [
        r'MJ12bot',
        r'MajesticSEO'
    ],
    'Screaming Frog': [
        r'Screaming Frog',
        r'ScreamingFrog'
    ],
    
    # Other Crawlers
    'Other Crawlers': [
        r'bot(?!AI)',  # Generic bot but not AI
        r'Bot(?!AI)',
        r'spider',
        r'Spider',
        r'crawler',
        r'Crawler',
        r'scraper',
        r'Scraper',
        r'curl',
        r'wget',
        r'python-requests',
        r'axios',
        r'HTTPClient'
    ]
}

# LLM/AI Bot categories for special handling
LLM_AI_CATEGORIES = [
    'OpenAI', 'Anthropic', 'Google AI', 'Meta AI', 'Perplexity', 'Microsoft',
    'Character.AI', 'Inflection', 'Cohere', 'AI21', 'Hugging Face', 'Replicate',
    'Stability AI', 'You.com', 'Neeva', 'Kagi', 'Brave', 'DuckDuckGo AI', 'Other AI Bots'
]

# Log format regex patterns
LOG_PATTERNS = {
    'Apache Common': r'^(\S+) \S+ \S+ $([\w:/]+\s[+\-]\d{4})$ \"(\S+)\s?(\S+)?\s?(\S+)?\" (\d{3}|-) (\d+|-) \"([^\"]*)\" \"([^\"]*)\"?',
    'Apache Combined': r'^(\S+) \S+ \S+ $([\w:/]+\s[+\-]\d{4})$ \"(\S+)\s?(\S+)?\s?(\S+)?\" (\d{3}|-) (\d+|-) \"([^\"]*)\" \"([^\"]*)\"',
    'Nginx': r'^(\S+) - \S+ $([\w:/]+\s[+\-]\d{4})$ \"(\S+)\s?(\S+)?\s?(\S+)?\" (\d{3}|-) (\d+|-) \"([^\"]*)\" \"([^\"]*)\"',
    'IIS': r'^(\S+) \S+ \S+ (\S+ \S+) (\S+) (\d{3}) (\d+) (\d+) (\d+) (\d+) \"([^\"]*)\"'
}

@st.cache_data
def identify_crawler(user_agent):
    """Identify crawler type from user agent string with priority for LLM/AI bots"""
    if pd.isna(user_agent) or user_agent == '-':
        return 'Unknown'
    
    user_agent = str(user_agent)
    
    # Check LLM/AI bots first (higher priority)
    for crawler_type in LLM_AI_CATEGORIES:
        if crawler_type in CRAWLER_PATTERNS:
            patterns = CRAWLER_PATTERNS[crawler_type]
            for pattern in patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    return crawler_type
    
    # Then check other crawlers
    for crawler_type, patterns in CRAWLER_PATTERNS.items():
        if crawler_type not in LLM_AI_CATEGORIES:
            for pattern in patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    return crawler_type
    
    return 'Human/Other'

@st.cache_data
def categorize_crawler_type(crawler_type):
    """Categorize crawler into major groups"""
    if crawler_type in LLM_AI_CATEGORIES:
        return 'LLM/AI Bots'
    elif crawler_type in ['Googlebot', 'Bingbot', 'Yahoo', 'Baidu', 'Yandex', 'DuckDuckGo']:
        return 'Search Engines'
    elif crawler_type in ['Facebook', 'Twitter', 'LinkedIn', 'WhatsApp', 'Telegram', 'Apple', 'Amazon', 'Pinterest']:
        return 'Social Media'
    elif crawler_type in ['Semrush', 'Ahrefs', 'Moz', 'Majestic', 'Screaming Frog']:
        return 'SEO Tools'
    elif crawler_type in ['Other Crawlers']:
        return 'Other Bots'
    elif crawler_type == 'Human/Other':
        return 'Human Traffic'
    else:
        return 'Unknown'

@st.cache_data
def parse_log_line(line, log_format):
    """Parse a single log line based on the specified format"""
    pattern = LOG_PATTERNS.get(log_format)
    if not pattern:
        return None
    
    match = re.match(pattern, line.strip())
    if not match:
        return None
    
    groups = match.groups()
    
    if log_format in ['Apache Common', 'Apache Combined', 'Nginx']:
        return {
            'ip': groups[0],
            'timestamp': groups[1],
            'method': groups[2],
            'url': groups[3] if groups[3] else '',
            'protocol': groups[4] if groups[4] else '',
            'status': groups[5],
            'size': groups[6],
            'referer': groups[7] if len(groups) > 7 else '',
            'user_agent': groups[8] if len(groups) > 8 else ''
        }
    elif log_format == 'IIS':
        return {
            'ip': groups[0],
            'timestamp': groups[1],
            'method': groups[2],
            'url': '',
            'protocol': '',
            'status': groups[3],
            'size': groups[4],
            'referer': '',
            'user_agent': groups[8] if len(groups) > 8 else ''
        }
    
    return None

@st.cache_data
def parse_timestamp(timestamp_str, log_format):
    """Parse timestamp string to datetime object"""
    try:
        if log_format in ['Apache Common', 'Apache Combined', 'Nginx']:
            # Format: [10/Oct/2000:13:55:36 -0700]
            dt_str = timestamp_str.split()[0]
            return datetime.strptime(dt_str, '%d/%b/%Y:%H:%M:%S')
        elif log_format == 'IIS':
            # Format: 2000-10-10 13:55:36
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except:
        return None

@st.cache_data
def process_log_file(file_content, log_format, progress_bar=None):
    """Process uploaded log file and return DataFrame"""
    lines = file_content.decode('utf-8').splitlines()
    total_lines = len(lines)
    
    parsed_data = []
    failed_lines = 0
    
    for i, line in enumerate(lines):
        if progress_bar and i % 1000 == 0:
            progress_bar.progress(i / total_lines)
        
        if line.strip():
            parsed = parse_log_line(line, log_format)
            if parsed:
                parsed_data.append(parsed)
            else:
                failed_lines += 1
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    if not parsed_data:
        return None, failed_lines
    
    df = pd.DataFrame(parsed_data)
    
    # Process timestamps
    df['datetime'] = df['timestamp'].apply(lambda x: parse_timestamp(x, log_format))
    df = df.dropna(subset=['datetime'])
    
    # Identify crawlers
    df['crawler_type'] = df['user_agent'].apply(identify_crawler)
    df['crawler_category'] = df['crawler_type'].apply(categorize_crawler_type)
    
    # Clean and convert data types
    df['status'] = pd.to_numeric(df['status'], errors='coerce')
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['size'] = df['size'].fillna(0)
    
    return df, failed_lines

def create_llm_ai_highlight_chart(df):
    """Create special chart highlighting LLM/AI bot activity"""
    llm_df = df[df['crawler_category'] == 'LLM/AI Bots']
    
    if len(llm_df) == 0:
        return None
    
    llm_counts = llm_df['crawler_type'].value_counts()
    
    # Create a color palette that highlights LLM bots
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8', '#f7dc6f']
    
    fig = px.bar(
        x=llm_counts.values,
        y=llm_counts.index,
        orientation='h',
        title='🤖 LLM/AI Bot Activity Analysis',
        labels={'x': 'Request Count', 'y': 'AI Bot Type'},
        color=llm_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_category_comparison_chart(df):
    """Create chart comparing different crawler categories"""
    category_counts = df['crawler_category'].value_counts()
    
    # Custom colors for different categories
    color_map = {
        'LLM/AI Bots': '#ff6b6b',
        'Search Engines': '#4ecdc4', 
        'Social Media': '#45b7d1',
        'SEO Tools': '#96ceb4',
        'Other Bots': '#ffeaa7',
        'Human Traffic': '#dda0dd',
        'Unknown': '#cccccc'
    }
    
    colors = [color_map.get(cat, '#cccccc') for cat in category_counts.index]
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title='Traffic Distribution by Category',
        color_discrete_sequence=colors
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_top_crawlers_chart(df):
    """Create bar chart of top crawlers by request count"""
    crawler_counts = df['crawler_type'].value_counts().head(15)
    
    # Color LLM/AI bots differently
    colors = []
    for crawler in crawler_counts.index:
        if crawler in LLM_AI_CATEGORIES:
            colors.append('#ff6b6b')  # Red for LLM/AI
        else:
            colors.append('#4ecdc4')  # Teal for others
    
    fig = px.bar(
        x=crawler_counts.values,
        y=crawler_counts.index,
        orientation='h',
        title='Top 15 Crawlers by Request Count',
        labels={'x': 'Request Count', 'y': 'Crawler Type'},
        color=colors,
        color_discrete_map="identity"
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_status_codes_by_crawler_chart(df):
    """Create stacked bar chart of status codes by crawler type"""
    # Focus on top 10 crawlers for readability
    top_crawlers = df['crawler_type'].value_counts().head(10).index
    filtered_df = df[df['crawler_type'].isin(top_crawlers)]
    
    status_crawler = pd.crosstab(filtered_df['crawler_type'], filtered_df['status'])
    
    fig = px.bar(
        status_crawler,
        title='Status Code Distribution by Top Crawlers',
        labels={'value': 'Request Count', 'index': 'Crawler Type'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        legend_title='Status Code'
    )
    
    return fig

def create_top_urls_chart(df):
    """Create horizontal bar chart of most crawled URLs"""
    url_counts = df['url'].value_counts().head(20)
    
    # Truncate long URLs for display
    truncated_urls = [url[:50] + '...' if len(url) > 50 else url for url in url_counts.index]
    
    fig = px.bar(
        x=url_counts.values,
        y=truncated_urls,
        orientation='h',
        title='Top 20 Most Crawled URLs',
        labels={'x': 'Request Count', 'y': 'URL'},
        color=url_counts.values,
        color_continuous_scale='plasma'
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_timeline_chart(df):
    """Create timeline of crawler activity with LLM/AI emphasis"""
    df['date'] = df['datetime'].dt.date
    timeline_data = df.groupby(['date', 'crawler_category']).size().reset_index(name='count')
    
    # Custom colors for timeline
    color_map = {
        'LLM/AI Bots': '#ff6b6b',
        'Search Engines': '#4ecdc4', 
        'Social Media': '#45b7d1',
        'SEO Tools': '#96ceb4',
        'Other Bots': '#ffeaa7',
        'Human Traffic': '#dda0dd'
    }
    
    fig = px.line(
        timeline_data,
        x='date',
        y='count',
        color='crawler_category',
        title='Timeline of Crawler Activity by Category',
        labels={'count': 'Request Count', 'date': 'Date'},
        color_discrete_map=color_map
    )
    
    fig.update_layout(height=500)
    
    return fig

def create_status_pie_chart(df):
    """Create pie chart of status code distribution"""
    status_counts = df['status'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='Status Code Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(height=500)
    
    return fig

def get_download_link(df, filename="filtered_logs.csv"):
    """Generate download link for CSV export"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def main():
    st.title("🤖 Advanced Log File Analysis Dashboard")
    st.markdown("""
    Upload your log files to analyze crawler activity with **enhanced LLM/AI bot detection**. 
    Track OpenAI, Anthropic, Google AI, and other AI crawlers alongside traditional search engines.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose log file",
            type=['txt', 'log'],
            help="Upload .txt or .log files for analysis"
        )
        
        # Log format selection
        log_format = st.selectbox(
            "Select Log Format",
            options=list(LOG_PATTERNS.keys()),
            help="Choose the format that matches your log file"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📄 File size: {uploaded_file.size:,} bytes")
    
    if uploaded_file is not None:
        # Process the log file
        with st.spinner("Processing log file..."):
            progress_bar = st.progress(0)
            
            try:
                df, failed_lines = process_log_file(
                    uploaded_file.getvalue(),
                    log_format,
                    progress_bar
                )
                
                progress_bar.empty()
                
                if df is None:
                    st.error("❌ No valid log entries found. Please check your log format selection.")
                    return
                
                if failed_lines > 0:
                    st.warning(f"⚠️ {failed_lines} lines could not be parsed")
                
                st.success(f"✅ Successfully processed {len(df):,} log entries")
                
                # Filters in sidebar
                with st.sidebar:
                    st.header("🔧 Filters")
                    
                    # Date range filter
                    min_date = df['datetime'].min().date()
                    max_date = df['datetime'].max().date()
                    
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    # Status code filter
                    available_status_codes = sorted(df['status'].dropna().unique())
                    selected_status_codes = st.multiselect(
                        "Status Codes",
                        options=available_status_codes,
                        default=available_status_codes,
                        help="Select status codes to include"
                    )
                    
                    # Crawler category filter
                    available_categories = sorted(df['crawler_category'].unique())
                    selected_categories = st.multiselect(
                        "Crawler Categories",
                        options=available_categories,
                        default=available_categories,
                        help="Select crawler categories to include"
                    )
                    
                    # Crawler type filter
                    available_crawlers = sorted(df['crawler_type'].unique())
                    selected_crawlers = st.multiselect(
                        "Specific Crawlers",
                        options=available_crawlers,
                        default=available_crawlers,
                        help="Select specific crawler types to include"
                    )
                
                # Apply filters
                filtered_df = df.copy()
                
                # Date filter
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df['datetime'].dt.date >= start_date) &
                        (filtered_df['datetime'].dt.date <= end_date)
                    ]
                
                # Status code filter
                if selected_status_codes:
                    filtered_df = filtered_df[filtered_df['status'].isin(selected_status_codes)]
                
                # Category filter
                if selected_categories:
                    filtered_df = filtered_df[filtered_df['crawler_category'].isin(selected_categories)]
                
                # Crawler type filter
                if selected_crawlers:
                    filtered_df = filtered_df[filtered_df['crawler_type'].isin(selected_crawlers)]
                
                if len(filtered_df) == 0:
                    st.warning("⚠️ No data matches the selected filters.")
                    return
                
                # Enhanced summary statistics with LLM/AI focus
                st.header("📈 Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Requests",
                        f"{len(filtered_df):,}",
                        delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
                    )
                
                with col2:
                    unique_ips = filtered_df['ip'].nunique()
                    st.metric("Unique IPs", f"{unique_ips:,}")
                
                with col3:
                    llm_requests = len(filtered_df[filtered_df['crawler_category'] == 'LLM/AI Bots'])
                    llm_percentage = (llm_requests / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                    st.metric(
                        "🤖 LLM/AI Bot Requests", 
                        f"{llm_requests:,}", 
                        f"{llm_percentage:.1f}%"
                    )
                
                with col4:
                    crawlers_count = len(filtered_df[filtered_df['crawler_category'] != 'Human Traffic'])
                    crawler_percentage = (crawlers_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                    st.metric("All Bot Requests", f"{crawlers_count:,}", f"{crawler_percentage:.1f}%")
                
                # Additional statistics
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    success_rate = (len(filtered_df[filtered_df['status'] < 400]) / len(filtered_df)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col6:
                    error_rate = (len(filtered_df[filtered_df['status'] >= 400]) / len(filtered_df)) * 100
                    st.metric("Error Rate", f"{error_rate:.1f}%")
                
                with col7:
                    unique_urls = filtered_df['url'].nunique()
                    st.metric("Unique URLs", f"{unique_urls:,}")
                
                with col8:
                    llm_types = len(filtered_df[filtered_df['crawler_category'] == 'LLM/AI Bots']['crawler_type'].unique())
                    st.metric("🤖 AI Bot Types", f"{llm_types:,}")
                
                # Special LLM/AI Bot Analysis Section
                st.header("🤖 LLM/AI Bot Analysis")
                
                llm_chart = create_llm_ai_highlight_chart(filtered_df)
                if llm_chart:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(llm_chart, use_container_width=True)
                    with col2:
                        category_chart = create_category_comparison_chart(filtered_df)
                        st.plotly_chart(category_chart, use_container_width=True)
                    
                    # LLM/AI specific statistics
                    llm_df = filtered_df[filtered_df['crawler_category'] == 'LLM/AI Bots']
                    if len(llm_df) > 0:
                        st.subheader("🔍 LLM/AI Bot Insights")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            most_active_llm = llm_df['crawler_type'].value_counts().index[0]
                            st.metric("Most Active LLM Bot", most_active_llm)
                        
                
