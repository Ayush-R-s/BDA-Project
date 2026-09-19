"""
AI Crime Analytics Assistant — Backend Engine for Streamlit Dashboard.
Handles OpenAI API integration, data grounding, and strict topic guardrails.
"""

import json
import os
from typing import Iterator

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (e.g., from .env)
load_dotenv()


def get_openai_client() -> openai.Client | None:
    """Initialize and return the OpenAI client."""
    # Check session state first, then env var
    api_key = st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        return None
    try:
        return openai.Client(api_key=api_key)
    except Exception:
        return None


def generate_system_prompt(df_clusters: pd.DataFrame, df_metro: pd.DataFrame) -> str:
    """Generate the grounded system prompt based on project data."""
    
    total_districts = len(df_clusters)
    total_metros = len(df_metro)
    total_crime = int(df_clusters["total_crime_burden"].sum())
    total_anomalies = df_clusters["is_anomaly"].sum()
    
    prompt = f"""You are the NCRB 2024 Crime Analytics Intelligence Assistant, an expert AI embedded within a dashboard.
Your primary role is to answer questions strictly related to the NCRB 2024 Crime Analytics project.

### Dataset Overview
- **Total Districts Analyzed:** {total_districts}
- **Metropolitan Cities (Pop > 1M):** {total_metros}
- **Total Crime Burden (IPC + SLL):** {total_crime:,}
- **DBSCAN Anomalies Detected:** {total_anomalies}

### Key Typologies (K-Means Clusters)
- **Cluster 0:** Muted Terracotta — High Violent Crime & Women Vulnerability
- **Cluster 1:** Steel Blue — Agrarian Belts
- **Cluster 2:** Muted Cyan / Sage — Low-Intensity Stable
- **Cluster 3:** Warm Sand / Amber — Major Commercial Hubs
- **Cluster 4:** Soft Olive / Green — Specialized Cyber Wings

### Metropolitan Policing Quadrants
- **Q1:** Critical Bottleneck (High Crime, Low Chargesheet)
- **Q2:** Active Enforcement (High Crime, High Chargesheet)
- **Q3:** Effective Containment (Low Crime, High Chargesheet)
- **Q4:** Latent Risk (Low Crime, Low Chargesheet)

### STRICT GUARDRAILS
1. You must ONLY answer questions related to the NCRB 2024 crime analytics data, spatial clusters, district vulnerability footprints, metropolitan policing quadrants, motives, and anomalies.
2. If the user asks ANY question outside this scope (e.g., coding help, general knowledge, weather, unrelated politics, creative writing), you MUST reply with exactly: "I am a dedicated crime analytics intelligence assistant focused solely on the NCRB 2024 Crime Analytics project. I can only assist with questions regarding district crime patterns, spatial typologies, metropolitan policing efficiency, and anomaly diagnostics."
3. Always be professional, concise, and refer to data accurately.
"""
    return prompt


def generate_chat_response(
    client: openai.Client,
    messages: list[dict],
    df_clusters: pd.DataFrame,
    df_metro: pd.DataFrame
) -> Iterator[str]:
    """Generate a streaming chat response using OpenAI."""
    
    system_prompt = generate_system_prompt(df_clusters, df_metro)
    
    api_messages = [{"role": "system", "content": system_prompt}]
    
    # Process user history. If user asks about a specific city/district, we could theoretically inject context here.
    # For now, we rely on the summary in the system prompt.
    for msg in messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            stream=True,
            temperature=0.3,
            max_tokens=1000
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Error communicating with OpenAI: {str(e)}"
