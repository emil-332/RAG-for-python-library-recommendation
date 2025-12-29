import streamlit as st
import ollama
import json
from pathlib import Path
from indexing_and_retrieval import retrieve_libraries

# --- Configuration ---
MODEL_NAME = "llama3.1"
ENRICHED_DIR = Path("enriched")

# --- Prompts ---

# 1. Query Analysis Prompt (The "Expert" Step)
ANALYSIS_PROMPT = """
You are a senior software architect. Analyze the user's project idea.
Determine if the description is specific enough to recommend precise Python libraries.

Return a JSON object with this structure:
{
    "status": "vague" or "specific",
    "content": [
        // If "vague": Provide 2-3 short, specific technical questions to narrow down the requirements.
        // If "specific": Provide a string of 5-10 technical keywords to expand the search query.
    ]
}

Example 1 (Vague):
Input: "I want to scrape websites."
Output: {"status": "vague", "content": ["Are the sites static HTML or dynamic (JavaScript)?", "Do you need to bypass captchas?"]}

Example 2 (Specific):
Input: "I want to scrape React websites using a headless browser."
Output: {"status": "specific", "content": "selenium playwright pyppeteer dynamic-content headless browser automation javascript-rendering"}
"""

# 2. RAG Generation Prompt (The "Recommender" Step)
RAG_SYSTEM_PROMPT = """
You are a software architecture assistant. Your goal is to recommend libraries based on the user's project description.
Use the provided Context to recommend the best libraries and justify your recommendations.
Focus on the specific technical needs identified in the user's query.
If the Context is empty or irrelevant, strictly state that you cannot find suitable libraries in the database.

Structure your answer:
1. **Top Recommendation**: The best fit.
2. **Alternatives**: 2 other options.
3. **Why**: specific technical reasons linking the library features to the user's goal.

Use a bullet point list of advantages and disadvantages for each recommendation.
List at most 3 advantages and disadvantages, and use most relevant advantages/disadvantages in the given context.
Start with the best recommendation, then 2nd place, 3rd place.
Then follow up with a final sentence or two that recommends a specific library based on the context the user gave.
Talk to the user directly, not in the third person about the user.
"""

def analyze_user_query(user_input):
    """
    Step 1: User Modeling / Query Expansion.
    Decides if we need to ask questions or if we can search immediately.
    """
    try:
        response = ollama.chat(model=MODEL_NAME, format='json', messages=[
            {'role': 'system', 'content': ANALYSIS_PROMPT},
            {'role': 'user', 'content': user_input},
        ])
        return json.loads(response['message']['content'])
    except Exception as e:
        # Fallback if JSON parsing fails
        return {"status": "specific", "content": user_input}

def load_library_details(library_name):
    """Load enriched data for transparency."""
    safe_name = library_name.replace(" ", "-")
    file_path = ENRICHED_DIR / f"{safe_name}.json"
    
    if not file_path.exists():
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", [])
        full_text = "\n\n".join([c.get("text", "") for c in chunks])
        return {
            "summary": data.get("summary", ""),
            "usage_description": data.get("usage_description", ""),
            "full_readme": full_text
        }
    except:
        return None

def generate_rag_response(search_query, original_intent=""):
    """
    Step 2: Retrieval & Generation.
    Uses the 'search_query' (which might be the expanded keywords) to find docs,
    but answers the 'original_intent' to keep the conversation natural.
    """
    # 1. Retrieve
    try:
        retrieved_items = retrieve_libraries(search_query, top_x=4)
    except Exception as e:
        return f"Error connecting to Vector DB: {str(e)}", []

    # 2. Build Context
    context_str = ""
    final_results = []
    
    for item in retrieved_items:
        details = load_library_details(item['library'])
        if details:
            item.update(details)
        else:
            item.update({"full_readme": "N/A", "usage_description": "N/A", "summary": ""})
        
        final_results.append(item)
        
        context_str += f"""
        === Library: {item['library']} ===
        [Tags]: {', '.join(item['tags'])}
        [Enriched Analysis]: {item['usage_description']}
        [Snippet]: {item['context_chunk']}
        \n
        """

    # 3. Generate Answer
    final_prompt = f"""
    User's Original Goal: "{original_intent}"
    Technical Search Context: "{search_query}"

    Available Libraries:
    {context_str}

    Recommend the best libraries based on the context. Explain WHY they fit the technical context.
    """

    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': RAG_SYSTEM_PROMPT},
            {'role': 'user', 'content': final_prompt},
        ])
        return response['message']['content'], final_results
    except Exception as e:
        return f"Error: {e}", []

# --- UI Logic ---
st.set_page_config(page_title="GenAI Library Expert", layout="wide")
st.title("🧠 Adaptive Library Architect")

# Session State for "Conversation Flow"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "clarification_mode" not in st.session_state:
    st.session_state.clarification_mode = False
if "original_query" not in st.session_state:
    st.session_state.original_query = ""

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Describe your project idea..."):
    
    # User message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- FLOW A: Handling Clarification Answers ---
    if st.session_state.clarification_mode:
        # The user just answered the questions.
        # We combine original query + their answers into a super-query.
        combined_query = f"Project: {st.session_state.original_query}. Requirements: {prompt}"
        
        with st.spinner("Synthesizing requirements and retrieving libraries..."):
            # We treat this combined query as "Specific" now
            ai_response, sources = generate_rag_response(combined_query, original_intent=st.session_state.original_query)
        
        # Reset mode
        st.session_state.clarification_mode = False
        st.session_state.original_query = ""
        
        # Display Results
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            if sources:
                st.divider()
                st.subheader("📚 Analyzed Libraries")
                for s in sources:
                     with st.expander(f"📄 {s['library']} (Score: {s['score']:.2f})"):
                        tab1, tab2 = st.tabs(["Analysis", "Full Readme"])
                        with tab1:
                            st.info(s['usage_description'])
                            st.caption(s['context_chunk'])
                        with tab2:
                            st.markdown(s['full_readme'])
                            
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # --- FLOW B: New Query Analysis ---
    else:
        with st.spinner("Analyzing complexity..."):
            analysis = analyze_user_query(prompt)
        
        if analysis['status'] == 'vague':
            # Case: Query is vague -> Ask Questions
            st.session_state.clarification_mode = True
            st.session_state.original_query = prompt
            
            # Format questions nicely
            questions_text = "**I need a bit more detail to give you the best advice:**\n\n"
            for q in analysis['content']:
                questions_text += f"- {q}\n"
            
            with st.chat_message("assistant"):
                st.markdown(questions_text)
            
            st.session_state.messages.append({"role": "assistant", "content": questions_text})
            
        else:
            # Case: Query is specific -> Expand Keywords & Search
            expanded_terms = analysis['content']
            # We append the keywords to the prompt for the vector search
            enhanced_search_query = f"{prompt} {expanded_terms}"
            
            with st.spinner(f"Searching with enhanced context: '{expanded_terms}'..."):
                ai_response, sources = generate_rag_response(enhanced_search_query, original_intent=prompt)
            
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                if sources:
                    st.divider()
                    st.subheader("📚 Analyzed Libraries")
                    for s in sources:
                         with st.expander(f"📄 {s['library']} (Score: {s['score']:.2f})"):
                            tab1, tab2 = st.tabs(["Analysis", "Full Readme"])
                            with tab1:
                                st.info(s['usage_description'])
                                st.caption(s['context_chunk'])
                            with tab2:
                                st.markdown(s['full_readme'])
                                
            st.session_state.messages.append({"role": "assistant", "content": ai_response})