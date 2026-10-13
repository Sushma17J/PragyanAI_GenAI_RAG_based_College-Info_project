
import os
import re
import json
import time
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Karnataka College Finder",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
    color: #172033;
}
h1, h2, h3 {
    color: #173b70;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Karnataka College Finder")
st.write(
    "Upload a college list, research college information online, "
    "estimate college tiers, and ask questions using AI."
)

# --------------------------------------------------
# GROQ API KEY: UPDATED TO GROQ_API_KEY_1
# --------------------------------------------------
API_KEY = st.secrets.get(
    "GROQ_API_KEY_1",
    os.getenv("GROQ_API_KEY_1", "")
)

MODEL_NAME = st.secrets.get(
    "GROQ_MODEL",
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

client = Groq(api_key=API_KEY) if API_KEY else None

if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = []
if "messages" not in st.session_state:
    st.session_state.messages = []


def clean_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def find_college_column(df):
    accepted = {
        "college name", "college", "institution name",
        "institute name", "name of college",
        "name of the college", "institution"
    }

    for col in df.columns:
        normalized = re.sub(
            r"[^a-z0-9 ]", "", str(col).lower()
        ).strip()

        if normalized in accepted:
            return col

    return None


def ask_groq(system_prompt, user_prompt):
    if not client:
        raise RuntimeError(
            "Configure GROQ_API_KEY_1 in Streamlit Secrets."
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# --------------------------------------------------
# WEB SEARCH
# --------------------------------------------------
def search_web(query, max_results=5):
    url = (
        "https://html.duckduckgo.com/html/?q="
        + quote_plus(query)
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(
            url, headers=headers, timeout=20
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for item in soup.select(".result"):
            title_tag = item.select_one(".result__title")
            snippet_tag = item.select_one(".result__snippet")

            if not title_tag or not snippet_tag:
                continue

            anchor = title_tag.find("a")
            if not anchor:
                continue

            results.append({
                "title": clean_text(anchor.get_text(" ", strip=True)),
                "url": anchor.get("href", ""),
                "snippet": clean_text(
                    snippet_tag.get_text(" ", strip=True)
                )
            })

            if len(results) >= max_results:
                break

        return results

    except requests.RequestException:
        return []


def fetch_page_text(url, max_chars=3000):
    if not url.startswith(("http://", "https://")):
        return ""

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12
        )
        response.raise_for_status()

        if "text/html" not in response.headers.get(
            "Content-Type", ""
        ).lower():
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "noscript"
        ]):
            tag.decompose()

        return clean_text(
            soup.get_text(" ", strip=True)
        )[:max_chars]

    except requests.RequestException:
        return ""


# --------------------------------------------------
# RESEARCH COLLEGE
# --------------------------------------------------
def research_college(college_name, spreadsheet_info=""):
    queries = [
        f'"{college_name}" official website affiliation',
        f'"{college_name}" NIRF ranking NAAC NBA accreditation',
        f'"{college_name}" autonomous university VTU affiliation'
    ]

    sources = []

    for query in queries:
        sources.extend(search_web(query, max_results=4))
        time.sleep(0.3)

    unique_sources = []
    seen = set()

    for source in sources:
        key = source.get("url") or source.get("title")

        if key and key not in seen:
            seen.add(key)
            unique_sources.append(source)

    evidence_parts = []

    for source in unique_sources[:8]:
        page_text = fetch_page_text(source.get("url", ""))

        evidence_parts.append(
            f"Title: {source.get('title', '')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Snippet: {source.get('snippet', '')}\n"
            f"Page content: {page_text}"
        )

    evidence = "\n\n".join(evidence_parts)

    if not evidence:
        evidence = "No usable web evidence was retrieved."

    prompt = f"""
Research this college in Karnataka using the evidence below.

College: {college_name}
Spreadsheet information: {spreadsheet_info}

Evidence:
{evidence[:14000]}

Rules:
- Tier 1, Tier 2 and Tier 3 are estimates, not official categories.
- Do not invent rankings, affiliations, accreditation or autonomy.
- Prefer official college websites and official NIRF or accreditation
  sources when available in the evidence.
- If evidence is insufficient, say Unable to determine.
- Distinguish autonomous colleges, universities and affiliated colleges.
- Include only source URLs supplied in the evidence.

Return a JSON object with:
tier_estimate, confidence, evidence_summary,
institution_status, affiliation, nirf_evidence,
limitations, sources_used.
"""

    try:
        answer = ask_groq(
            "You are a careful college research assistant. "
            "Never invent evidence. Return valid JSON only.",
            prompt
        )

        answer = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "", answer.strip()
        )

        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError("The model did not return valid JSON.")

    except Exception as error:
        data = {
            "tier_estimate": "Unable to determine",
            "confidence": "Low",
            "evidence_summary": "Research could not be verified.",
            "institution_status": "Not verified",
            "affiliation": "Not verified",
            "nirf_evidence": "Not verified",
            "limitations": str(error),
            "sources_used": []
        }

    data["college_name"] = college_name
    data["search_results"] = unique_sources
    data["evidence"] = evidence
    data["spreadsheet_info"] = spreadsheet_info

    return data


# --------------------------------------------------
# RAG RETRIEVAL
# --------------------------------------------------
def build_documents(df, results):
    documents = []

    for _, row in df.iterrows():
        row_text = " | ".join(
            f"{col}: {row[col]}"
            for col in df.columns
            if pd.notna(row[col])
        )

        documents.append({
            "text": row_text,
            "source": "Uploaded spreadsheet"
        })

    for result in results:
        documents.append({
            "text": (
                f"College: {result.get('college_name')}\n"
                f"Tier estimate: {result.get('tier_estimate')}\n"
                f"Confidence: {result.get('confidence')}\n"
                f"Status: {result.get('institution_status')}\n"
                f"Affiliation: {result.get('affiliation')}\n"
                f"NIRF evidence: {result.get('nirf_evidence')}\n"
                f"Evidence summary: {result.get('evidence_summary')}\n"
                f"Limitations: {result.get('limitations')}\n"
                f"Research evidence: {result.get('evidence', '')}"
            ),
            "source": f"Research: {result.get('college_name')}"
        })

    return documents


def retrieve_documents(question, documents, top_k=5):
    if not documents:
        return []

    texts = [doc["text"] for doc in documents]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )
        matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector, matrix
        ).flatten()

        indices = scores.argsort()[::-1][:top_k]

        return [
            {
                **documents[i],
                "score": float(scores[i])
            }
            for i in indices if scores[i] > 0
        ]

    except ValueError:
        return []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Region: Karnataka, India")
    st.write("AI provider: Groq")

    if client:
        st.success("Groq API key detected")
    else:
        st.warning("GROQ_API_KEY_1 is not configured.")

    st.caption(
        "Never commit your API key to GitHub."
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# UPLOAD EXCEL OR CSV
# --------------------------------------------------
st.header("1. Upload college spreadsheet")

uploaded_file = st.file_uploader(
    "Upload an Excel or CSV file",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        college_column = find_college_column(df)

        if college_column is None:
            st.error(
                "Please name the college column "
                "'College Name' or 'Institution Name'."
            )
            st.write("Available columns:", list(df.columns))
        else:
            df[college_column] = (
                df[college_column].astype(str).str.strip()
            )
            df = df[
                df[college_column].ne("")
                & df[college_column].str.lower().ne("nan")
            ].drop_duplicates(subset=[college_column])

            st.session_state.df = df
            st.session_state.college_column = college_column
            st.session_state.results = []
            st.session_state.messages = []

    except Exception as error:
        st.error(f"Could not read file: {error}")


df = st.session_state.df

if df is not None:
    college_column = st.session_state.college_column

    st.success(f"Loaded {len(df)} colleges.")
    st.dataframe(df, use_container_width=True)

    college_names = df[college_column].tolist()

    selected_college = st.selectbox(
        "Select a college",
        college_names
    )

    row = df[df[college_column] == selected_college].iloc[0]

    with st.expander("College details from spreadsheet"):
        st.json({
            str(col): str(row[col])
            for col in df.columns if pd.notna(row[col])
        })

    # --------------------------------------------------
    # RESEARCH
    # --------------------------------------------------
    st.header("2. Research college information")

    research_scope = st.radio(
        "Research scope",
        ["Selected college", "All colleges"],
        horizontal=True
    )

    max_colleges = st.number_input(
        "Maximum colleges per run",
        min_value=1,
        max_value=100,
        value=min(10, len(df))
    )

    if st.button("🔎 Research tier and affiliation", type="primary"):
        if not client:
            st.error(
                "Set GROQ_API_KEY_1 in Streamlit Cloud Secrets first."
            )
        else:
            if research_scope == "Selected college":
                targets = [selected_college]
            else:
                targets = college_names[:int(max_colleges)]

            existing = {
                item["college_name"]: item
                for item in st.session_state.results
            }

            progress = st.progress(0)
            status = st.empty()

            for i, name in enumerate(targets):
                status.write(f"Researching {name}...")

                college_row = df[
                    df[college_column] == name
                ].iloc[0]

                extra = " | ".join(
                    f"{col}: {college_row[col]}"
                    for col in df.columns
                    if col != college_column
                    and pd.notna(college_row[col])
                )

                existing[name] = research_college(name, extra)
                progress.progress((i + 1) / len(targets))

            st.session_state.results = list(existing.values())
            status.success("Research finished.")
            st.rerun()

    # --------------------------------------------------
    # DISPLAY RESEARCH RESULTS
    # --------------------------------------------------
    results = st.session_state.results

    if results:
        st.header("3. College research results")

        output_df = pd.DataFrame([
            {
                "College Name": r.get("college_name", ""),
                "Estimated Tier": r.get(
                    "tier_estimate", "Unable to determine"
                ),
                "Confidence": r.get("confidence", "Low"),
                "Institution Status": r.get(
                    "institution_status", "Not verified"
                ),
                "Affiliation": r.get("affiliation", "Not verified"),
                "NIRF Evidence": r.get("nirf_evidence", "Not verified")
            }
            for r in results
        ])

        st.dataframe(output_df, use_container_width=True)

        st.download_button(
            "Download research results CSV",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="karnataka_college_research.csv",
            mime="text/csv"
        )

        names = [r["college_name"] for r in results]

        detail_name = st.selectbox(
            "View research evidence",
            names,
            key="research_detail"
        )

        detail = next(
            r for r in results
            if r["college_name"] == detail_name
        )

        st.subheader(detail_name)
        st.metric(
            "Estimated tier",
            detail.get("tier_estimate", "Unable to determine")
        )
        st.write("**Confidence:**", detail.get("confidence", "Low"))
        st.write("**Evidence:**", detail.get("evidence_summary", ""))
        st.write("**Institution status:**", detail.get(
            "institution_status", "Not verified"
        ))
        st.write("**Affiliation:**", detail.get("affiliation", ""))
        st.write("**NIRF evidence:**", detail.get("nirf_evidence", ""))
        st.write("**Limitations:**", detail.get("limitations", ""))

        st.markdown("### Search sources")

        for source in detail.get("search_results", []):
            title = source.get("title", "Source")
            url = source.get("url", "")
            snippet = source.get("snippet", "")

            if url.startswith(("http://", "https://")):
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")

            if snippet:
                st.caption(snippet)

        st.warning(
            "Tier is an AI-generated estimate, not an official ranking. "
            "Confirm important details using official sources."
        )

    # --------------------------------------------------
    # RAG CHATBOT
    # --------------------------------------------------
    st.header("4. Ask the college chatbot")

    documents = build_documents(df, st.session_state.results)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about colleges, tiers or affiliations..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not client:
                answer = "Configure GROQ_API_KEY_1 to use the chatbot."
                st.error(answer)

            else:
                relevant = retrieve_documents(question, documents)

                context = "\n\n".join(
                    f"Source: {doc['source']}\n{doc['text'][:3000]}"
                    for doc in relevant
                )

                if not context:
                    context = (
                        "No matching information was retrieved. "
                        "Do not invent an answer."
                    )

                try:
                    answer = ask_groq(
                        "You are Karnataka College Finder. "
                        "Answer using the supplied context only. "
                        "Do not invent rankings, affiliations or facts. "
                        "If information is missing, say so. "
                        "Tier labels are estimates, not official categories.",
                        f"Question: {question}\n\nContext:\n{context}"
                    )

                    st.markdown(answer)

                    with st.expander("Retrieved source information"):
                        for doc in relevant:
                            st.markdown(f"**{doc['source']}**")
                            st.write(doc["text"][:1200])

                except Exception as error:
                    answer = f"Could not generate an answer: {error}"
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info(
        "Upload an Excel or CSV file to begin. "
        "Include a column named College Name."
    )

    st.markdown("### Example Excel columns")
    st.dataframe(pd.DataFrame({
        "College Name": [
            "P D A College of Engineering",
            "BMS College of Engineering",
            "RV College of Engineering"
        ],
        "City": [
            "Kalaburagi",
            "Bengaluru",
            "Bengaluru"
        ]
    }), use_container_width=True)
