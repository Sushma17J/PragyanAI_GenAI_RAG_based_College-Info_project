import os
import re
import json
import time
import requests
import pandas as pd
import streamlit as st

from bs4 import BeautifulSoup
from groq import Groq
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Karnataka College Finder",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# 2. UI DESIGN
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
    color: #172033;
}
h1, h2, h3, h4 {
    color: #173b70 !important;
}
p, label, li, .stMarkdown {
    color: #172033;
}
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p {
    color: #172033 !important;
}
.stApp input, .stApp textarea {
    background-color: white !important;
    color: #172033 !important;
    border: 1px solid #cbd5e1 !important;
}
.stApp [data-baseweb="select"] > div {
    background-color: white !important;
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
.stButton > button, .stDownloadButton > button {
    background-color: #173b70 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] {
    background-color: #202938 !important;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dbe3ef;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Karnataka College Finder")
st.write(
    "Research colleges, explore estimated tiers and affiliations, "
    "and ask questions using a retrieval-augmented chatbot."
)

# =========================================================
# 3. GROQ CONFIGURATION
# =========================================================
API_KEY = st.secrets.get(
    "GROQ_API_KEY_1",
    os.getenv("GROQ_API_KEY_1", "")
)

MODEL_NAME = st.secrets.get(
    "GROQ_MODEL",
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

client = Groq(api_key=API_KEY) if API_KEY else None


# =========================================================
# 4. SESSION STATE
# =========================================================
defaults = {
    "college_df": None,
    "college_column": None,
    "research": {},
    "chat_history": [],
    "loaded_filename": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================
def clean_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def find_college_column(df):
    accepted = {
        "college name",
        "college",
        "institution name",
        "institute name",
        "name of college",
        "name of the college",
        "institution"
    }

    for col in df.columns:
        normalized = re.sub(
            r"[^a-z0-9 ]", "", str(col).lower()
        ).strip()

        if normalized in accepted:
            return col

    return None


def groq_answer(system_prompt, user_prompt):
    if not client:
        raise RuntimeError(
            "GROQ_API_KEY_1 is missing from Streamlit Secrets."
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


def parse_json_response(answer):
    answer = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        answer.strip()
    )

    match = re.search(r"\{.*\}", answer, re.DOTALL)

    if not match:
        raise ValueError("The AI did not return valid JSON.")

    return json.loads(match.group())


# =========================================================
# 6. WEB SEARCH - NO TAVILY KEY REQUIRED
# =========================================================
def search_web(query, max_results=5):
    """
    Uses the DDGS library. No separate search API key required.
    Results depend on search service availability.
    """
    results = []

    try:
        with DDGS() as search:
            items = search.text(
                query,
                max_results=max_results
            )

            for item in items:
                results.append({
                    "title": clean_text(
                        item.get("title", "")
                    ),
                    "url": item.get("href", ""),
                    "snippet": clean_text(
                        item.get("body", "")
                    )
                })

    except Exception:
        # Search can fail because of rate limits or network restrictions.
        return []

    return results


def fetch_page_text(url):
    """
    Attempts to retrieve readable text from a result page.
    Some websites block automated requests.
    """
    if not url.startswith(("http://", "https://")):
        return ""

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/130 Safari/537.36"
                )
            },
            timeout=12
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "noscript", "svg"
        ]):
            tag.decompose()

        return clean_text(
            soup.get_text(" ", strip=True)
        )[:4500]

    except requests.RequestException:
        return ""


# =========================================================
# 7. COLLEGE RESEARCH
# =========================================================
def research_college(college_name, spreadsheet_info=""):
    queries = [
        f'"{college_name}" Karnataka official website',
        f'"{college_name}" affiliation university VTU',
        f'"{college_name}" autonomous status official',
        f'"{college_name}" NIRF ranking NAAC NBA'
    ]

    sources = []
    seen_urls = set()

    for query in queries:
        found = search_web(query, max_results=5)

        for item in found:
            url = item.get("url", "")

            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(item)

        time.sleep(0.25)

    # Collect page content and search snippets.
    evidence_parts = []

    for source in sources[:12]:
        page_text = fetch_page_text(source.get("url", ""))

        evidence_parts.append(
            f"Title: {source.get('title', '')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Search snippet: {source.get('snippet', '')}\n"
            f"Page content: {page_text}"
        )

    evidence = "\n\n".join(evidence_parts)

    if not evidence.strip():
        evidence = (
            "No web evidence could be retrieved. "
            "Do not infer the college's status from its name."
        )

    if not client:
        result = {
            "tier_estimate": "Unable to determine",
            "institution_type": "Unable to determine",
            "university_name": "Unable to determine",
            "affiliation": "Unable to determine",
            "nirf_information": "Not verified",
            "accreditation": "Not verified",
            "confidence": "Low",
            "evidence_summary": (
                "Groq API key is not configured."
            ),
            "limitations": "AI analysis was not performed.",
            "sources_used": []
        }
    else:
        prompt = f"""
Research this Karnataka college using the evidence supplied below.

COLLEGE: {college_name}

SPREADSHEET DETAILS:
{spreadsheet_info}

RETRIEVED WEB EVIDENCE:
{evidence[:18000]}

Return JSON with exactly these keys:
tier_estimate
institution_type
university_name
affiliation
nirf_information
accreditation
confidence
evidence_summary
limitations
sources_used

Allowed tier_estimate values:
Tier 1, Tier 2, Tier 3, Unable to determine.

Allowed institution_type values:
University
Autonomous College
VTU-affiliated College
Other Affiliated College
Deemed-to-be University
Unable to determine

Rules:
1. Tier 1/2/3 is an estimate, not a universal official classification.
2. Never invent a tier, NIRF rank, accreditation or affiliation.
3. Prioritize official college/university websites and official ranking
   or accreditation sources when such evidence is present.
4. Do not assume every engineering college is affiliated to VTU.
5. A college may be autonomous while remaining affiliated to a university.
6. Use Unable to determine when the evidence is insufficient.
7. Explain which evidence supports the classification.
8. Use only URLs present in the retrieved evidence for sources_used.
9. If search results are empty, state that web research was unsuccessful.

Return valid JSON only.
"""

        try:
            answer = groq_answer(
                "You are a careful college research assistant. "
                "Use retrieved evidence, not guesses. Return valid JSON.",
                prompt
            )
            result = parse_json_response(answer)

        except Exception as error:
            result = {
                "tier_estimate": "Unable to determine",
                "institution_type": "Unable to determine",
                "university_name": "Unable to determine",
                "affiliation": "Unable to determine",
                "nirf_information": "Not verified",
                "accreditation": "Not verified",
                "confidence": "Low",
                "evidence_summary": "AI research could not be completed.",
                "limitations": str(error),
                "sources_used": []
            }

    result["college_name"] = college_name
    result["search_results"] = sources
    result["web_evidence"] = evidence
    result["spreadsheet_info"] = spreadsheet_info

    return result


# =========================================================
# 8. BUILD THE RAG KNOWLEDGE BASE
# =========================================================
def build_documents(df, college_column, research_results):
    documents = []

    # Spreadsheet documents.
    for _, row in df.iterrows():
        college = str(row[college_column])

        row_text = "\n".join(
            f"{col}: {row[col]}"
            for col in df.columns
            if pd.notna(row[col])
        )

        documents.append({
            "text": f"College: {college}\n{row_text}",
            "source": "Uploaded spreadsheet"
        })

    # Research documents with source URLs and retrieved evidence.
    for name, result in research_results.items():
        documents.append({
            "text": (
                f"College: {name}\n"
                f"Tier estimate: {result.get('tier_estimate')}\n"
                f"Institution type: {result.get('institution_type')}\n"
                f"University: {result.get('university_name')}\n"
                f"Affiliation: {result.get('affiliation')}\n"
                f"NIRF: {result.get('nirf_information')}\n"
                f"Accreditation: {result.get('accreditation')}\n"
                f"Confidence: {result.get('confidence')}\n"
                f"Evidence summary: {result.get('evidence_summary')}\n"
                f"Limitations: {result.get('limitations')}\n"
                f"Sources and retrieved evidence:\n"
                f"{result.get('web_evidence', '')}"
            ),
            "source": f"Web research: {name}"
        })

    return documents


def retrieve_documents(question, documents, top_k=6):
    if not documents:
        return []

    texts = [doc["text"] for doc in documents]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000
        )

        matrix = vectorizer.fit_transform(texts)
        question_vector = vectorizer.transform([question])

        scores = cosine_similarity(
            question_vector, matrix
        ).flatten()

        indices = scores.argsort()[::-1][:top_k]

        return [
            {
                **documents[i],
                "score": float(scores[i])
            }
            for i in indices
            if scores[i] > 0
        ]

    except ValueError:
        return []


# =========================================================
# 9. SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Location: Karnataka, India")
    st.write("AI: Groq")
    st.write("Web search: DuckDuckGo")
    st.write("Retrieval: TF-IDF")

    if client:
        st.success("Groq API key detected")
    else:
        st.warning("GROQ_API_KEY_1 is missing")

    st.caption(
        "Never commit your API keys to GitHub."
    )

    if st.button("Clear chatbot conversation"):
        st.session_state.chat_history = []
        st.rerun()


# =========================================================
# 10. UPLOAD EXCEL OR CSV
# =========================================================
st.header("1. Upload college list")

uploaded_file = st.file_uploader(
    "Choose an Excel or CSV file",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file:
    if (
        st.session_state.loaded_filename
        != uploaded_file.name
    ):
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            college_column = find_college_column(df)

            if college_column is None:
                st.error(
                    "College name column not found. "
                    "Rename it to 'College Name' or "
                    "'Institution Name'."
                )
                st.write("Detected columns:", list(df.columns))

            else:
                df[college_column] = (
                    df[college_column].astype(str).str.strip()
                )

                df = df[
                    df[college_column].ne("")
                    & df[college_column].str.lower().ne("nan")
                ].drop_duplicates(subset=[college_column])

                st.session_state.college_df = df
                st.session_state.college_column = college_column
                st.session_state.research = {}
                st.session_state.chat_history = []
                st.session_state.loaded_filename = uploaded_file.name

        except Exception as error:
            st.error(f"Could not read the file: {error}")


# =========================================================
# 11. SELECT COLLEGE
# =========================================================
df = st.session_state.college_df

if df is not None:
    college_column = st.session_state.college_column
    college_names = df[college_column].tolist()

    st.success(f"Loaded {len(df)} colleges.")

    with st.expander("Preview uploaded Excel data"):
        st.dataframe(df, use_container_width=True)

    st.header("2. Select a college")

    selected_college = st.selectbox(
        "College name",
        college_names
    )

    selected_row = df[
        df[college_column] == selected_college
    ].iloc[0]

    with st.expander("View spreadsheet details"):
        st.json({
            str(col): str(selected_row[col])
            for col in df.columns
            if pd.notna(selected_row[col])
        })

    # =====================================================
    # 12. RESEARCH CONTROLS
    # =====================================================
    st.header("3. Research college information")

    research_scope = st.radio(
        "Research scope",
        ["Selected college", "All colleges"],
        horizontal=True
    )

    max_colleges = st.number_input(
        "Maximum colleges to research",
        min_value=1,
        max_value=100,
        value=min(10, len(df))
    )

    if st.button(
        "🔎 Research tier and university status",
        type="primary"
    ):
        if not client:
            st.error(
                "Add GROQ_API_KEY_1 in Streamlit Cloud Secrets."
            )

        else:
            targets = (
                [selected_college]
                if research_scope == "Selected college"
                else college_names[:int(max_colleges)]
            )

            progress = st.progress(0)
            status = st.empty()

            for index, name in enumerate(targets):
                status.write(
                    f"Researching {index + 1}/{len(targets)}: {name}"
                )

                row = df[
                    df[college_column] == name
                ].iloc[0]

                spreadsheet_info = "\n".join(
                    f"{col}: {row[col]}"
                    for col in df.columns
                    if pd.notna(row[col])
                )

                result = research_college(
                    name,
                    spreadsheet_info
                )

                st.session_state.research[name] = result

                progress.progress(
                    (index + 1) / len(targets)
                )

            status.success("Research process finished.")
            st.rerun()

    # =====================================================
    # 13. RESULTS AND FILTERS
    # =====================================================
    results = st.session_state.research

    if results:
        st.header("4. Tier and institution classification")

        rows = []

        for name, result in results.items():
            rows.append({
                "College Name": name,
                "Estimated Tier": result.get(
                    "tier_estimate", "Unable to determine"
                ),
                "Institution Type": result.get(
                    "institution_type", "Unable to determine"
                ),
                "University": result.get(
                    "university_name", "Unable to determine"
                ),
                "Affiliation": result.get(
                    "affiliation", "Unable to determine"
                ),
                "NIRF": result.get(
                    "nirf_information", "Not verified"
                ),
                "Accreditation": result.get(
                    "accreditation", "Not verified"
                ),
                "Confidence": result.get("confidence", "Low")
            })

        result_df = pd.DataFrame(rows)

        first, second = st.columns(2)

        with first:
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

        with second:
            institution_filter = st.selectbox(
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

        filtered_df = result_df.copy()

        if tier_filter != "All tiers":
            filtered_df = filtered_df[
                filtered_df["Estimated Tier"] == tier_filter
            ]

        if institution_filter != "All types":
            filtered_df = filtered_df[
                filtered_df["Institution Type"] == institution_filter
            ]

        # HTML table styling via pandas Styler.
        styled_df = filtered_df.style.set_properties(
            **{
                "background-color": "#ffffff",
                "color": "#172033",
                "border-color": "#d1d5db",
                "text-align": "left"
            }
        ).set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#173b70"),
                    ("color", "#ffffff"),
                    ("font-weight", "bold")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", "#172033")
                ]
            }
        ])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download college research CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="karnataka_college_research.csv",
            mime="text/csv"
        )

        # =================================================
        # 14. DETAILED REPORT AND SOURCES
        # =================================================
        st.subheader("Detailed research report")

        detail_name = st.selectbox(
            "Choose a researched college",
            list(results.keys()),
            key="detail_college"
        )

        detail = results[detail_name]

        left, right = st.columns(2)

        with left:
            st.metric(
                "Estimated Tier",
                detail.get("tier_estimate", "Unable to determine")
            )
            st.markdown("**Institution Type**")
            st.info(
                detail.get(
                    "institution_type", "Unable to determine"
                )
            )
            st.markdown("**University Name**")
            st.write(
                detail.get("university_name", "Unable to determine")
            )

        with right:
            st.metric(
                "Confidence",
                detail.get("confidence", "Low")
            )
            st.markdown("**Affiliation**")
            st.write(
                detail.get("affiliation", "Unable to determine")
            )
            st.markdown("**NIRF Information**")
            st.write(
                detail.get("nirf_information", "Not verified")
            )

        st.markdown("**Accreditation**")
        st.write(detail.get("accreditation", "Not verified"))

        st.markdown("**Evidence Summary**")
        st.write(detail.get("evidence_summary", ""))

        st.markdown("**Limitations**")
        st.write(detail.get("limitations", ""))

        st.markdown("### Retrieved web sources")

        if detail.get("search_results"):
            for source in detail["search_results"]:
                title = source.get("title", "Source")
                url = source.get("url", "")
                snippet = source.get("snippet", "")

                if url.startswith(("http://", "https://")):
                    st.markdown(f"- [{title}]({url})")
                else:
                    st.write(title)

                if snippet:
                    st.caption(snippet)
        else:
            st.warning(
                "No search results were retrieved. "
                "Try again later or verify the college manually."
            )

        with st.expander("View collected research evidence"):
            st.text(detail.get("web_evidence", ""))

        st.warning(
            "Tier is an estimate, not an official universal category. "
            "Verify affiliations and rankings through official sources."
        )

    # =====================================================
    # 15. RAG CHATBOT
    # =====================================================
    st.divider()
    st.header("5. Ask the college chatbot")

    documents = build_documents(
        df,
        college_column,
        st.session_state.research
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Example: Which researched colleges are autonomous?"
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
                answer = (
                    "Configure GROQ_API_KEY_1 in Streamlit Secrets."
                )
                st.error(answer)

            else:
                relevant_docs = retrieve_documents(
                    question,
                    documents,
                    top_k=6
                )

                context_parts = []

                for doc in relevant_docs:
                    context_parts.append(
                        f"Source label: {doc['source']}\n"
                        f"Evidence:\n{doc['text'][:3500]}"
                    )

                context = "\n\n---\n\n".join(context_parts)

                if not context:
                    context = (
                        "No matching documents were retrieved. "
                        "Do not invent an answer."
                    )

                try:
                    answer = groq_answer(
                        """
You are Karnataka College Finder, a RAG research assistant.

Answer using only the supplied retrieved context.
Answer the question directly and clearly.
Distinguish verified facts from estimates.
Never invent rankings, tiers, accreditation or affiliation.
Tier 1/2/3 is an estimate unless an explicit source defines it.
If evidence is missing, say so.
Include relevant source URLs found in the supplied context.
Do not claim to have searched the web during this chat;
the web evidence was collected in the research step.
""",
                        f"USER QUESTION:\n{question}\n\n"
                        f"RETRIEVED CONTEXT:\n{context}"
                    )

                    st.markdown(answer)

                    with st.expander("View retrieved RAG evidence"):
                        for doc in relevant_docs:
                            st.markdown(f"**{doc['source']}**")
                            st.write(doc["text"][:2000])

                except Exception as error:
                    answer = f"Could not generate an answer: {error}"
                    st.error(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info(
        "Upload an Excel or CSV file containing a column named "
        "'College Name' to begin."
    )

    st.markdown("### Example spreadsheet")

    st.dataframe(
        pd.DataFrame({
            "College Name": [
                "P. A. College of Engineering",
                "Canara Engineering College",
                "Malnad College of Engineering"
            ],
            "City": [
                "Mangaluru",
                "Bantwal",
                "Hassan"
            ]
        }),
        use_container_width=True
    )
