import os
import re
import time
import tempfile
import subprocess
from typing import Optional, Tuple
import json
import numpy as np
import pandas as pd
import PyPDF2
import pyaudio
import soundfile as sf
import pygame
import torch
from langdetect import detect_langs
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
from kokoro import KPipeline
from ollama import Client
import streamlit as st
from streamlit_chat import message as st_chat_message
from PIL import Image
from googletrans import Translator
from deep_translator import GoogleTranslator
from duckduckgo_search import DDGs

st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="centered"
)
st.title("Chat with your own personal Chatbot!")

init_container = st.empty()

# Model settings
DEFAULT_MODEL = "gemma3:12b"
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"

TEXT_MAX_CHARS = 30000  # Maximum characters for text content
DATAFRAME_MAX_CHARS = 30000  # Maximum characters for dataframe string representation

# System prompts for different languages
ENGLISH_SYSTEM_PROMPT = """
Always respond in English regardless of input language.
Keep answers easy to read and understand with as few words as possible.
Pay attention to the conversation context and reference previous turns when relevant.
"""

CHINESE_SYSTEM_PROMPT = """
無論輸入語言如何，始終用繁體中文回答。
保持回答簡潔明了。
注意對話上下文，在相關時引用之前的對話內容。
"""

# Data analysis prompts
DATAFRAME_ANALYSIS_PROMPT_EN = """
You are a helpful assistant specialized in analyzing pandas DataFrames.
Analyze the pandas DataFrame below and answer the user's question.
Keep your answers concise and easy to read and understand, like explaining to a child.
Give simple examples if possible.

DataFrame:
```
{dataframe_string}
```

User Question: {user_prompt}

Answer:
"""

DATAFRAME_ANALYSIS_PROMPT_ZH = """
你是一個專門分析pandas DataFrame的助手。
請分析以下DataFrame並用繁體中文回答用戶的問題。
保持回答簡潔易讀易懂，就像向小孩解釋一樣。
如果可能的話，給出簡單的例子。

DataFrame:
```
{dataframe_string}
```

用戶問題: {user_prompt}

回答:
"""

TEXT_ANALYSIS_PROMPT_EN = """
You are a helpful assistant specialized in analyzing text content.
Analyze the text below and answer the user's question.
Keep your answers concise and easy to read and understand, like explaining to a child.
Give simple examples if possible.

Text Content:
```
{text_content_string}
```

User Question: {user_prompt}

Answer:
"""

TEXT_ANALYSIS_PROMPT_ZH = """
你是一個專門分析文本內容的助手。
請用繁體中文分析以下文本並回答用戶的問題。
保持回答簡潔易讀易懂，就像向小孩解釋一樣。
如果可能的話，給出簡單的例子。

文本內容:
```
{text_content_string}
```

用戶問題: {user_prompt}

請用繁體中文回答:"""

# Chain of thought prompts
COT_SYSTEM_PROMPT = """You are an expert AI assistant that explains your reasoning step by step. For each step:
1. Provide a clear title describing what you're examining
2. Give detailed analysis and reasoning
3. Consider multiple possibilities and approaches
4. Evaluate evidence and assumptions
5. Draw logical conclusions

USE AT LEAST 3 DIFFERENT APPROACHES:
- Direct analysis
- Alternative perspectives
- Edge cases and limitations
- Comparative analysis
- Validation and verification

CONSIDER LIMITATIONS:
- Assumptions and biases
- Data completeness
- Alternative interpretations
- Potential errors

Format your response in JSON with:
- 'title': Step title
- 'content': Detailed reasoning
- 'next_action': Either 'continue' or 'final_answer'
"""

COT_DATA_SYSTEM_PROMPT = """You are an expert AI assistant analyzing data. For each step:
1. Provide a clear title describing what you're examining
2. Give detailed analysis of the data
3. Consider multiple interpretations
4. Evaluate patterns and relationships
5. Draw data-driven conclusions

USE AT LEAST 3 DIFFERENT ANALYTICAL APPROACHES:
- Statistical analysis
- Pattern recognition
- Trend identification
- Outlier detection
- Data validation

CONSIDER LIMITATIONS:
- Data quality and completeness
- Statistical significance
- Potential biases
- Alternative interpretations

Format your response in JSON with:
- 'title': Step title
- 'content': Detailed analysis
- 'next_action': Either 'continue' or 'final_answer'
"""

# Audio configuration
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
CHUNK_DURATION = CHUNK_SIZE / SAMPLE_RATE

# Recording configuration
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 3  # seconds before stopping
MAX_RECORDING_DURATION = 30  # maximum recording duration in seconds

# Upload options
UPLOAD_OPTIONS = ["None", "Text File (TXT/PDF)", "Spreadsheet (CSV/Excel)", "Image (JPG/PNG)"]

# Chain of thought configuration
COT_CONFIG = {
    "temperature": 0.2,
    "max_steps": 5,
    "step_max_tokens": 300,
    "final_answer_max_tokens": 200,
    "max_retries": 3,
    "retry_delay": 0.5
}

# Default LLM configuration
DEFAULT_LLM = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
}

# Search configuration
SEARCH_CONFIG = {
    "max_results": 3,  # Number of search results to fetch per query
    "max_iterations": 3,  # Number of search iterations
    "min_confidence_threshold": 0.7,  # Threshold for when to dig deeper
    "max_query_length": 150,  # Maximum length of generated search queries
    "region": "wt-wt",  # Worldwide results
    "safesearch": "moderate"
}

# PDF analysis prompts
PDF_ANALYSIS_PROMPT_EN = """
You are a helpful assistant specialized in analyzing PDF documents.
Analyze the PDF content below and answer the user's question.
Keep your answers concise and easy to read and understand, like explaining to a child.
Give simple examples if possible.

PDF Content:
```
{pdf_content_string}
```

User Question: {user_prompt}

Answer:
"""

PDF_ANALYSIS_PROMPT_ZH = """
你是一個專門分析PDF文件的助手。
請分析以下PDF內容並用繁體中文回答用戶的問題。
保持回答簡潔易讀易懂，就像向小孩解釋一樣。
如果可能的話，給出簡單的例子。

PDF內容:
```
{pdf_content_string}
```

用戶問題: {user_prompt}

回答:
"""
# --- 9. Helper Functions ---
def generate_search_query(topic, previous_findings=None, knowledge_gaps=None):
    """Generate a focused search query based on topic and previous research."""
    try:
        # Create a more direct prompt that emphasizes the original question
        language = st.session_state.fixed_language
        
        if language == 'zh':
            prompt = f"""原始問題: {topic}
您的任務: 生成一個簡單直接的搜索查詢（5-10個詞）以查找有關此主題的信息。
- 只關注原始問題中的主要概念
- 不要包含像「這是」或「搜索查詢」等短語
- 不要使用引號或特殊字符
- 只寫實際的搜索詞，沒有其他內容
- 要具體並在適當時使用醫學/技術術語

好的查詢示例:
- DOAC VKA 心包填塞 比較
- 阿司匹林 抗炎 作用機制
- 氣候變化 珊瑚礁 影響

不好的查詢示例:
- 「這是關於DOAC的搜索查詢」
- 這裡有兩個可能的搜索查詢
- 我將搜索有關的信息

您的搜索查詢:"""
        else:
            prompt = f"""ORIGINAL QUESTION: {topic}
Your task: Generate a simple, direct search query (5-10 words) to find information about this topic.
- Focus ONLY on the main concepts from the original question
- Do NOT include phrases like "here is" or "search query"
- Do NOT use quotes or special characters
- Just write the actual search terms, nothing else
- Be specific and use medical/technical terms when appropriate

Example good queries:
- DOAC VKA cardiac tamponade comparison
- aspirin mechanism of action inflammation
- climate change impact coral reefs

Example bad queries:
- "Here is a search query about DOAC"
- Here are two possible search queries
- I will search for information about

YOUR SEARCH QUERY:"""
        
        response = st.session_state.ollama_client.chat(
            model=st.session_state.selected_model,
            messages=[
                {"role": "system", "content": "你是一個搜索查詢生成器。只生成搜索詞，沒有其他內容。" if language == 'zh' else "You are a search query generator. Generate only the search terms, nothing else."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Clean up the generated query
        query = response['message']['content'].strip()
        query = query.replace('"', '').replace("'", "").replace("(", "").replace(")", "")
        
        # Remove common prefixes that the model might still include
        prefixes_to_remove = [
            "Here is", "Here are", "Search query:", "Query:", "YOUR SEARCH QUERY:", 
            "A good search query would be", "I would search for", "Search for",
            "以下是", "查詢:", "搜索查詢:", "一個好的搜索查詢是", "我會搜索", "搜索"
        ]
        
        for prefix in prefixes_to_remove:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        
        # If query is too long, take just the first sentence or limit to 15 words
        if len(query.split()) > 15:
            if "." in query:
                query = query.split(".")[0].strip()
            else:
                query = " ".join(query.split()[:15])
        
        # Print the original question and the search query for clarity
        if language == 'zh':
            st.markdown(f"*原始問題:* {topic}")
            st.markdown(f"*生成的搜索詞:* {query}")
        else:
            st.markdown(f"*Original Question:* {topic}")
            st.markdown(f"*Generated Search Terms:* {query}")
        
        return query
    except Exception as e:
        st.warning(f"Error generating search query: {e}")
        # Fallback to a simplified version of the original question
        simple_query = " ".join(topic.split()[:10])
        return simple_query

def analyze_findings(topic, current_findings, iteration):
    """Analyze current findings and identify knowledge gaps."""
    try:
        language = st.session_state.fixed_language
        
        if language == 'zh':
            prompt = f"""基於我們對 '{topic}' 的研究，分析以下發現：

{current_findings}

對於迭代 {iteration}，請：
1. 總結我們學到的要點
2. 確定特定的知識差距或需要更多細節的領域
3. 建議我們接下來應該調查哪些具體方面

請將您的回應格式化為JSON，使用鍵：'summary'（摘要）、'gaps'（差距）、'next_focus'（下一個焦點）"""
        else:
            prompt = f"""Based on our research about '{topic}', analyze these findings:

{current_findings}

For iteration {iteration}, please:
1. Summarize the key points we've learned
2. Identify specific knowledge gaps or areas that need more detail
3. Suggest what specific aspects we should investigate next

Format your response as JSON with keys: 'summary', 'gaps', 'next_focus'"""

        response = st.session_state.ollama_client.chat(
            model=st.session_state.selected_model,
            messages=[
                {"role": "system", "content": "你是一個研究分析師，識別知識差距。" if language == 'zh' else "You are a research analyst identifying knowledge gaps."},
                {"role": "user", "content": prompt}
            ],
            format='json'
        )
        
        analysis_result = json.loads(response['message']['content'])
        
        # Ensure summary is a string
        summary = analysis_result.get('summary', '')
        if not isinstance(summary, str):
            summary = str(summary)  # Convert to string if not
        
        return {
            "summary": summary,
            "gaps": analysis_result.get('gaps', "沒有識別出差距" if language == 'zh' else "No gaps identified"),
            "next_focus": analysis_result.get('next_focus', topic)
        }
    except Exception as e:
        st.warning(f"Error analyzing findings: {e}")
        return {
            "summary": "分析失敗" if st.session_state.fixed_language == 'zh' else "Analysis failed",
            "gaps": "分析失敗" if st.session_state.fixed_language == 'zh' else "Analysis failed",
            "next_focus": topic
        }
    
def deep_research(query, language='en'):
    """Perform iterative deep research on a topic."""
    all_findings = []
    current_summary = ""
    all_sources = []  # Track all sources used
    max_attempts = 6  # Maximum number of attempts to get meaningful results
    
    # Create a container to capture the research process
    research_container = st.empty()
    with research_container, st.expander("🔍 Research Process", expanded=True):
        iteration = 0
        while iteration < SEARCH_CONFIG['max_iterations']:  # Changed from RESEARCH_CONFIG
            attempt = 0
            meaningful_iteration = False
            
            while not meaningful_iteration and attempt < max_attempts:
                attempt += 1
                st.markdown(f"**Iteration {iteration + 1} (Attempt {attempt})**")
                
                # Generate search query
                search_query = generate_search_query(
                    query, 
                    current_summary if current_summary else None,
                    all_findings[-1].get('gaps', None) if all_findings else None
                )
                
                st.markdown(f"*Search Query:* {search_query}")
                
                # Perform search and display sources
                search_results = get_web_search_results(
                    search_query,
                    language=language
                )
                
                if not search_results:
                    st.warning("No information found. Trying a different query...")
                    continue
                
                # Extract sources
                sources = []
                for line in search_results.split('\n'):
                    if 'Source:' in line or '來源:' in line:
                        url = line.split(': ')[-1].strip()
                        if url not in all_sources:  # Only add unique sources
                            sources.append(url)