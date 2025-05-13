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