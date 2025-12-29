import streamlit as st
import ollama
import json
from pathlib import Path
from indexing_and_retrieval import retrieve_libraries

# --- Configuration ---
MODEL_NAME = "llama3.1"
ENRICHED_DIR = Path("enriched")

SYSTEM_PROMPT = """
You are a software architecture assistant. Your goal is to recommend libraries based on the user's project description.
Use the provided Context (retrieved library data) to justify your recommendations.
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

def load_library_details(library_name):
    """
    Loads the full enriched data for a given library from the JSON file.
    Handles spaces in library names by replacing them with hyphens.
    """
    # Construct filename: "library name" -> "library-name.json"
    safe_name = library_name.replace(" ", "-")
    file_path = ENRICHED_DIR / f"{safe_name}.json"
    
    if not file_path.exists():
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Reconstruct the full readme by joining all text chunks
        chunks = data.get("chunks", [])
        full_text = "\n\n".join([c.get("text", "") for c in chunks])
        
        return {
            "summary": data.get("summary", "No summary available."),
            "usage_description": data.get("usage_description", "No enriched description available."),
            "full_readme": full_text
        }
    except Exception as e:
        return {"summary": "", "usage_description": f"Error loading file: {e}", "full_readme": ""}

def generate_rag_response(user_query):
    # 1. Retrieval
    try:
        # Retrieve top matches based on vector similarity
        retrieved_items = retrieve_libraries(user_query, top_x=6)
    except Exception as e:
        return f"Error connecting to Vector DB: {str(e)}", []

    # 2. Context Construction & Data Loading
    context_str = ""
    final_results = []
    
    for item in retrieved_items:
        # Load the full enriched JSON for this library
        details = load_library_details(item['library'])
        
        if details:
            # Add loaded details to the item dictionary for the UI
            item['full_readme'] = details['full_readme']
            item['usage_description'] = details['usage_description']
            item['summary'] = details['summary']
        else:
            # Fallback if file not found
            item['full_readme'] = "Could not find enriched data file."
            item['usage_description'] = "N/A"
            item['summary'] = item.get('summary', '')

        final_results.append(item)

        # Add to the Prompt Context
        # We include the Enriched Description and Summary for better LLM reasoning
        context_str += f"""
        === Library: {item['library']} ===
        [Tags]: {', '.join(item['tags'])}
        [Summary]: {item['summary']}
        [Enriched Analysis]: {item['usage_description']}
        [Relevant Readme Snippet]:
        {item['context_chunk']}
        ==================================
        \n
        """

    final_prompt = f"""
    User Project Goal: "{user_query}"

    Available Libraries (Context):
    {context_str}

    Based ONLY on the context above, recommend the best libraries.
    """

    # 3. Generation (Ollama)
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': final_prompt},
        ])
        return response['message']['content'], final_results
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Is 'ollama serve' running?", []

# --- UI Layout ---
st.set_page_config(page_title="Library Recommender", layout="wide") # Wide layout for better reading
st.title("📚 Intelligent Library Recommender")
st.caption("Describe your project goal (e.g., 'I want to build a web scraper for news sites')")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Area
if prompt := st.chat_input("What are you building?"):
    # Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Response
    with st.spinner("Analyzing requirements & retrieving libraries..."):
        ai_response, sources = generate_rag_response(prompt)

    # Display AI Message
    with st.chat_message("assistant"):
        st.markdown(ai_response)
        
        # Sources Section
        if sources:
            st.divider()
            st.subheader("📚 Analyzed Libraries")
            for s in sources:
                # Expander for each library
                with st.expander(f"📄 {s['library']} (Score: {s['score']:.2f})"):
                    # Tab structure to keep it clean
                    tab1, tab2 = st.tabs(["Analysis & Summary", "Full Readme"])
                    
                    with tab1:
                        st.markdown(f"**Summary:** {s['summary']}")
                        st.info(f"**💡 Enriched Usage Description:**\n\n{s['usage_description']}")
                        st.caption(f"**Tags:** {', '.join(s['tags'])}")
                    
                    with tab2:
                        st.markdown(s['full_readme'])

    st.session_state.messages.append({"role": "assistant", "content": ai_response})