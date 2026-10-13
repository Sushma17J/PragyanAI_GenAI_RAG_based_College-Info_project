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


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Karnataka College Finder",
    page_icon="🎓",
    layout="wide"
)

# ==================================================
# UI DESIGN - IMPROVED VISIBILITY
# ==================================================
st.markdown("""
<style>
.stApp {
    background: #f5f7fb;
    color: #172033;
}
h1, h2, h3, h4 {
    color: #173b70 !important;
}
.stApp label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp .stMarkdown p {
    color: #172033 !important;
}
.stApp input,
.stApp textarea {
    background: #ffffff !important;
    color: #172033 !important;
    border: 1px solid #cbd5e1 !important;
}
.stApp [data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #172033 !important;
    border-color: #cbd5e1 !important;
}
.stApp [data-baseweb="select"] * {
    color: #172033 !important;
}
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stRadio"] label p {
    color: #172033 !important;
}
.stApp .stButton > button,
.stApp .stDownloadButton > button {
    background: #ff4b4b !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}
.stApp .stButton > button:hover,
.stApp .stDownloadButton > button:hover {
    background: #df3535 !important;
}
section[data-testid="stSidebar"] {
    background: #262730 !important;
}
section[data-testid="stSidebar"] * {
    color: #ffffff;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
    padding: 15px;
}
[data-testid="stDataFrame"] {
    background: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Karnataka College Finder")
st.write(
    "Research Karnataka colleges, estimate their tiers, "
    "check university/autonomous/VTU status, and ask questions."
)
st.divider()


# ==================================================
# GROQ CONFIGURATION
# ==================================================
API_KEY = st.secrets.get(
    "GROQ_API_KEY_1",
    os.getenv("GROQ_API_KEY_1", "")
)

MODEL_NAME = st.secrets.get(
    "GROQ_MODEL",
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

client = Groq(api_key=API_KEY) if API_KEY else None


# ==================================================
# SESSION STATE
# ==================================================
defaults = {
    "df": None,
    "college_column": None,
    "research_results": {},
    "chat_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# HELPERS
# ==================================================
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


def call_groq(system_prompt, user_prompt):
    if not client:
        raise RuntimeError(
            "GROQ_API_KEY_1 is not configured."
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


def parse_json(answer):
    answer = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        answer.strip()
    )

    match = re.search(r"\{.*\}", answer, re.DOTALL)

    if not match:
        raise ValueError("AI did not return valid JSON.")

    return json.loads(match.group())


# ==================================================
# WEB SEARCH
# ==================================================
def web_search(query, limit=5):
    url = (
        "https://html.duckduckgo.com/html/?q="
        + quote_plus(query)
    )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
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

            if len(results) >= limit:
                break

        return results

    except requests.RequestException:
        return []


def fetch_page(url):
    if not url.startswith(("http://", "https://")):
        return ""

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
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
        )[:3500]

    except requests.RequestException:
        return ""


# ==================================================
# RESEARCH AND CLASSIFICATION
# ==================================================
def research_college(college_name, spreadsheet_info=""):
    queries = [
        f'"{college_name}" official website Karnataka',
        f'"{college_name}" VTU affiliation autonomous university',
        f'"{college_name}" NIRF NAAC NBA accreditation ranking',
        f'"{college_name}" official affiliation university'
    ]

    all_sources = []

    for query in queries:
        all_sources.extend(web_search(query, limit=4))
        time.sleep(0.3)

    unique_sources = []
    seen = set()

    for source in all_sources:
        key = source.get("url") or source.get("title")

        if key and key not in seen:
            seen.add(key)
            unique_sources.append(source)

    evidence_parts = []

    for source in unique_sources[:10]:
        page_text = fetch_page(source.get("url", ""))

        evidence_parts.append(
            f"Title: {source.get('title', '')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Snippet: {source.get('snippet', '')}\n"
            f"Page text: {page_text}"
        )

    evidence = "\n\n".join(evidence_parts)

    if not evidence:
        evidence = "No usable web evidence was retrieved."

    prompt = f"""
Research this college in Karnataka.

COLLEGE: {college_name}
SPREADSHEET INFORMATION: {spreadsheet_info}

WEB EVIDENCE:
{evidence[:16000]}

Return the following information:

1. tier_estimate:
   Tier 1, Tier 2, Tier 3, or Unable to determine.

2. institution_type:
   University, Autonomous College, VTU-affiliated College,
   Other Affiliated College, Deemed-to-be University,
   or Unable to determine.

3. university_name:
   Name of the university, if verified.

4. affiliation:
   Verified affiliating university, or Unable to determine.

5. nirf_information:
   Rank, category and year only if supported by the evidence.

6. accreditation:
   Verified NAAC/NBA information, or Unable to determine.

7. confidence:
   High, Medium or Low.

8. evidence_summary:
   Explain why you selected the tier and institution type.

9. limitations:
   Explain missing or conflicting evidence.

10. sources_used:
    List source URLs that appear in the supplied evidence.

IMPORTANT:
- Tier 1/2/3 is an estimate, not an official universal classification.
- Do not assume every engineering college is affiliated to VTU.
- Do not call a college a university merely because its name sounds similar.
- Do not invent accreditation, NIRF rankings or affiliations.
- Give priority to official institution, university, NIRF and accreditation
  sources present in the evidence.
- If the evidence is insufficient, use "Unable to determine".

Return ONLY valid JSON with these exact keys:
tier_estimate, institution_type, university_name, affiliation,
nirf_information, accreditation, confidence, evidence_summary,
limitations, sources_used.
"""

    try:
        answer = call_groq(
            "You are a careful college research assistant. "
            "Use evidence only. Never fabricate facts. Return JSON.",
            prompt
        )
        result = parse_json(answer)

    except Exception as error:
        result = {
            "tier_estimate": "Unable to determine",
            "institution_type": "Unable to determine",
            "university_name": "Unable to determine",
            "affiliation": "Unable to determine",
            "nirf_information": "Unable to determine",
            "accreditation": "Unable to determine",
            "confidence": "Low",
            "evidence_summary": "Research could not be verified.",
            "limitations": str(error),
            "sources_used": []
        }

    result["college_name"] = college_name
    result["search_results"] = unique_sources
    result["web_evidence"] = evidence
    result["spreadsheet_info"] = spreadsheet_info

    return result


# ==================================================
# BUILD RAG DOCUMENTS
# ==================================================
def build_documents(df, college_column, results):
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

    for result in results.values():
        documents.append({
            "text": (
                f"College: {result.get('college_name')}\n"
                f"Tier: {result.get('tier_estimate')}\n"
                f"Institution type: {result.get('institution_type')}\n"
                f"University: {result.get('university_name')}\n"
                f"Affiliation: {result.get('affiliation')}\n"
                f"NIRF: {result.get('nirf_information')}\n"
                f"Accreditation: {result.get('accreditation')}\n"
                f"Confidence: {result.get('confidence')}\n"
                f"Evidence: {result.get('evidence_summary')}\n"
                f"Limitations: {result.get('limitations')}\n"
                f"Web evidence: {result.get('web_evidence', '')}"
            ),
            "source": f"Research: {result.get('college_name')}"
        })

    return documents


def retrieve_documents(question, documents, top_k=5):
    if not documents:
        return []

    try:
        texts = [doc["text"] for doc in documents]

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
            {**documents[i], "score": float(scores[i])}
            for i in indices if scores[i] > 0
        ]

    except ValueError:
        return []


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Region: Karnataka, India")
    st.write("AI provider: Groq")

    if client:
        st.success("Groq API key detected")
    else:
        st.error("GROQ_API_KEY_1 is missing")

    st.caption(
        "Never upload your API key to GitHub."
    )

    if st.button("Clear chatbot history"):
        st.session_state.chat_history = []
        st.rerun()


# ==================================================
# UPLOAD COLLEGE LIST
# ==================================================
st.header("1. Upload college list")

uploaded_file = st.file_uploader(
    "Upload Excel (.xlsx/.xls) or CSV (.csv)",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)

        column = find_college_column(new_df)

        if column is None:
            st.error(
                "College name column not found. Rename it to "
                "'College Name' or 'Institution Name'."
            )
            st.write("Your columns:", list(new_df.columns))

        else:
            new_df[column] = new_df[column].astype(str).str.strip()
            new_df = new_df[
                new_df[column].ne("")
                & new_df[column].str.lower().ne("nan")
            ].drop_duplicates(subset=[column])

            st.session_state.df = new_df
            st.session_state.college_column = column
            st.session_state.research_results = {}
            st.session_state.chat_history = []

    except Exception as error:
        st.error(f"File error: {error}")


# ==================================================
# COLLEGE SELECTION
# ==================================================
df = st.session_state.df

if df is not None:
    column = st.session_state.college_column
    names = df[column].tolist()

    st.success(f"Loaded {len(df)} colleges.")

    with st.expander("Preview uploaded spreadsheet"):
        st.dataframe(df, use_container_width=True)

    st.header("2. Select a college")

    selected = st.selectbox(
        "Select a college",
        names,
        key="selected_college"
    )

    selected_row = df[df[column] == selected].iloc[0]

    with st.expander("College details from spreadsheet"):
        st.json({
            str(col): str(selected_row[col])
            for col in df.columns
            if pd.notna(selected_row[col])
        })

    # ==================================================
    # RESEARCH CONTROLS
    # ==================================================
    st.header("3. Research tier and institution type")

    scope = st.radio(
        "Choose research scope",
        ["Selected college", "All colleges"],
        horizontal=True
    )

    max_count = st.number_input(
        "Maximum colleges to research",
        min_value=1,
        max_value=100,
        value=min(10, len(df))
    )

    if st.button(
        "🔎 Research tier, university and affiliation",
        type="primary"
    ):
        if not client:
            st.error(
                "Add GROQ_API_KEY_1 in Streamlit Cloud Secrets."
            )
        else:
            targets = (
                [selected]
                if scope == "Selected college"
                else names[:int(max_count)]
            )

            progress = st.progress(0)
            status = st.empty()

            for i, name in enumerate(targets):
                status.write(
                    f"Researching {i + 1}/{len(targets)}: {name}"
                )

                row = df[df[column] == name].iloc[0]

                extra = " | ".join(
                    f"{col}: {row[col]}"
                    for col in df.columns
                    if col != column and pd.notna(row[col])
                )

                st.session_state.research_results[name] = (
                    research_college(name, extra)
                )

                progress.progress((i + 1) / len(targets))

            status.success("Research completed.")
            st.rerun()

    # ==================================================
    # RESULTS TABLE AND FILTERS
    # ==================================================
    results = st.session_state.research_results

    if results:
        st.header("4. College tier and university results")

        result_rows = []

        for name, r in results.items():
            result_rows.append({
                "College Name": name,
                "Estimated Tier": r.get(
                    "tier_estimate", "Unable to determine"
                ),
                "Institution Type": r.get(
                    "institution_type", "Unable to determine"
                ),
                "University": r.get(
                    "university_name", "Unable to determine"
                ),
                "Affiliation": r.get(
                    "affiliation", "Unable to determine"
                ),
                "NIRF": r.get(
                    "nirf_information", "Unable to determine"
                ),
                "Accreditation": r.get(
                    "accreditation", "Unable to determine"
                ),
                "Confidence": r.get("confidence", "Low")
            })

        result_df = pd.DataFrame(result_rows)

        col1, col2 = st.columns(2)

        with col1:
            tier_filter = st.selectbox(
                "Filter by tier",
                [
                    "All tiers",
                    "Tier 1",
                    "Tier 2",
                    "Tier 3",
                    "Unable to determine"
                ]
            )

        with col2:
            type_filter = st.selectbox(
                "Filter by institution type",
                [
                    "All types",
                    "University",
                    "Autonomous College",
                    "VTU-affiliated College",
                    "Other Affiliated College",
                    "Deemed-to-be University",
                    "Unable to determine"
                ]
            )

        filtered = result_df.copy()

        if tier_filter != "All tiers":
            filtered = filtered[
                filtered["Estimated Tier"] == tier_filter
            ]

        if type_filter != "All types":
            filtered = filtered[
                filtered["Institution Type"] == type_filter
            ]

        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "⬇️ Download all research results",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="karnataka_college_results.csv",
            mime="text/csv"
        )

        # ==================================================
        # DETAILED RESULT
        # ==================================================
        st.subheader("Detailed college report")

        detail_name = st.selectbox(
            "Choose a researched college",
            list(results.keys()),
            key="detail_college"
        )

        detail = results[detail_name]

        a, b = st.columns(2)

        with a:
            st.metric(
                "Estimated tier",
                detail.get("tier_estimate", "Unable to determine")
            )
            st.markdown("**Institution type**")
            st.info(
                detail.get(
                    "institution_type", "Unable to determine"
                )
            )
            st.markdown("**Affiliating university**")
            st.write(
                detail.get("university_name", "Unable to determine")
            )

        with b:
            st.metric(
                "Confidence",
                detail.get("confidence", "Low")
            )
            st.markdown("**Affiliation**")
            st.write(
                detail.get("affiliation", "Unable to determine")
            )
            st.markdown("**NIRF information**")
            st.write(
                detail.get("nirf_information", "Not verified")
            )

        st.markdown("**Accreditation**")
        st.write(detail.get("accreditation", "Not verified"))

        st.markdown("**Evidence summary**")
        st.write(detail.get("evidence_summary", "No summary available."))

        st.markdown("**Limitations**")
        st.write(detail.get("limitations", "Not available."))

        st.markdown("### Research sources")

        for source in detail.get("search_results", []):
            url = source.get("url", "")
            title = source.get("title", "Source")

            if url.startswith(("http://", "https://")):
                st.markdown(f"- [{title}]({url})")
            else:
                st.write(f"- {title}")

            st.caption(source.get("snippet", ""))

        st.warning(
            "Tier is an AI estimate, not an official classification. "
            "Confirm affiliation and institution status through official "
            "university or college records before making decisions."
        )

    # ==================================================
    # RAG CHATBOT
    # ==================================================
    st.divider()
    st.header("5. Ask the college chatbot")

    documents = build_documents(
        df,
        column,
        st.session_state.research_results
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Example: Show researched autonomous colleges"
    )

    if question:
        st.session_state.chat_history.append({
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
                    f"Source: {d['source']}\n{d['text'][:3000]}"
                    for d in relevant
                )

                if not context:
                    context = (
                        "No matching information was found. "
                        "Do not invent an answer."
                    )

                try:
                    answer = call_groq(
                        "You are Karnataka College Finder. "
                        "Answer using the supplied evidence only. "
                        "Do not invent tiers, rankings or affiliations. "
                        "Say when information is missing.",
                        f"Question: {question}\n\nEvidence:\n{context}"
                    )

                    st.markdown(answer)

                    with st.expander("Retrieved information"):
                        for d in relevant:
                            st.markdown(f"**{d['source']}**")
                            st.write(d["text"][:1500])

                except Exception as error:
                    answer = f"Could not answer: {error}"
                    st.error(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info(
        "Upload an Excel or CSV file to start. "
        "Your spreadsheet must contain a College Name column."
    )

    st.markdown("### Example spreadsheet format")

    st.dataframe(pd.DataFrame({
        "College Name": [
            "K. V. G. College of Engineering",
            "P D A College of Engineering",
            "BMS College of Engineering"
        ],
        "City": [
            "Sullia",
            "Kalaburagi",
            "Bengaluru"
        ]
    }), use_container_width=True)

    st.caption(
        "These are example input names, not verified tier classifications."
    )
