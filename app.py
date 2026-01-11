import json
from pathlib import Path

import ollama
import streamlit as st

from indexing_and_retrieval import retrieve_libraries

# --- Configuration ---
MODEL_NAME = "llama3.1"
ENRICHED_DIR = Path("enriched")


WELCOME_MESSAGE = """
👋 **Welcome!**

I'm your **Adaptive Library Architect**.

Tell me what you want to build in Python — whether it's a quick script,
a production system, or just an idea you're exploring.

You can start simple, like
- *"I want to build a REST API"*
- *"I want to train a machine learning model"*
- *"I need to scrape some data"*

and we will figure the rest out together 😊
"""
ERROR_RESPONSE = "Oh no, something went wrong."

# --- Prompts ---
VALIDATION_PROMPT = """
You are a senior software architect, trying to aid the user in finding the best Python library for their project idea.
Unfortunately, some users don't always play by the rules or try to be malicious. Therefore, you need to validate
the user's intentions before proceeding. 

Please check if the user's latest message is relevant to the previous conversation and overall goal of this discourse.

Your response must be strictly of this form:
{
    "valid": bool
    "response": str
}

If the message is completely irrelevant for the goal of this conversation, possibly even inappropriate,
respond with a JSON object of this structure:
{
    "valid": false,
    "response": <appropriate-response>
}
Where the <appropriate-response> is a short, respectful but direct response to the user's latest message, explaining
why you cannot or will not proceed with this input. Also ask for a serious message by the user if they desire to continue
this conversation in this case. However, if they just greeted you or said something nice, don't forget to greet back
or say thank you first. In short, try to be natural but professional and goal-oriented.

In all other cases, respond with an affirming JSON object like this:
{
    "valid": true,
    "response": ""
}

Don't be strict here! Only flag responses as invalid if they are clearly non-compliant with the assistant's questions
and helpful guidance within this conversation.
"""

SUMMARIZE_PROMPT = """
You are a senior software architect. Together with the user, you are trying to narrow down a project idea
to be realised in Python in a technical discourse.

This is the (possibly empty) list of the requirements and keywords you have gathered so far:
{requirements}

Analyze the user's input in the context of the previously established requirements. Summarize the user's statements by
extracting a list of keywords or short sentences that accurately describe the new requirements gathered. DO NOT invent
new requirements by simply guessing or asking questions - instead only paraphrase what the user said.
Your response must only consist of the extracted list of requirements. Note that your list might be empty or very short
if the user only provided very little new information.

===
Example:
User input:"I want to build a REST API for a web application that allows users to book time slots to get vaccinated
at the city's public vaccination center. The users need to authenticate with their healthcare account, which uses
an LDAP server in the background."
Response: web application, REST API, goal: book time slots for vaccinations, LDAP user authentication
"""

# Query Analysis Prompt (The "Expert" Step)
ANALYSIS_PROMPT = """
You are a senior software architect. You are given a project idea to be realised in Python.
Analyze the user's project idea.
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

# RAG Generation Prompt (The "Recommender" Step)
RAG_SYSTEM_PROMPT = """
You are a software architecture assistant. Your goal is to recommend libraries based on the user's project description.
Use the provided Context to recommend the best libraries and justify your recommendations.
Focus on the specific technical needs identified in the user's query.
Only recommend libraries listed in the 'Available Libraries'.
If the 'Available Libraries' context is empty or irrelevant, strictly state that you cannot find suitable libraries in the database.

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



def check_compliance(conversation: list[dict[str, str]]) -> dict:
    """
    Semantic Input Validation
    Check if the user's latest message is relevant to the overall goal of the conversation.
    """
    chat_history = [{'role': 'assistant', 'content': WELCOME_MESSAGE}] + conversation
    try:
        while True:
            response = ollama.chat(model=MODEL_NAME, format='json', messages= chat_history + [
                {'role': 'system', 'content': VALIDATION_PROMPT},
            ])
            response_msg = json.loads(response['message']['content'])
            if 'valid' in response_msg:
                return response_msg
    except Exception:
        return {'valid': False, 'response': ERROR_RESPONSE}

def summarize_request(user_input: str, requirements: list[str]):
    """
    Query Expansion.
    Extract the core meaning from the user's request.
    """
    try:
        prompt_instance = SUMMARIZE_PROMPT.format(
            requirements="- " + "\n- ".join(requirements)
        )

        response = ollama.chat(model=MODEL_NAME, format='', messages=[
            {'role': 'system', 'content': prompt_instance},
            {'role': 'user', 'content': user_input},
        ])
        return response['message']['content'].strip()
    except Exception:
        return ""

def analyze_specificity(user_input):
    """
    Query Expansion.
    Decides if we need to ask questions or if we can search immediately.
    """
    try:
        while True:
            response = ollama.chat(model=MODEL_NAME, format='json', messages=[
                {'role': 'system', 'content': ANALYSIS_PROMPT},
                {'role': 'user', 'content': user_input},
            ])
            response_msg = json.loads(response['message']['content'])
            if 'status' in response_msg:
                return response_msg
    except Exception:
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

def show_recommendations(recommend_msg: str, sources: list):
    with st.chat_message("assistant"):
        st.markdown(recommend_msg)
        if sources:
            st.divider()
            st.subheader("📚 Relevant Libraries")
            for s in sources:
                with st.expander(f"📄 {s['library']} (Score: {s['score']:.2f})"):
                    tab1, tab2 = st.tabs(["Analysis", "Full Readme"])
                    with tab1:
                        st.info(s['usage_description'])
                        # st.caption(s['context_chunk'])
                    with tab2:
                        st.markdown(s['full_readme'])

    st.session_state.messages.append({"role": "assistant", "content": recommend_msg})


# --- UI Logic ---

# Session State for "Conversation Flow"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "original_query" not in st.session_state:
    st.session_state.original_query = ""
if "requirements" not in st.session_state:
    st.session_state.requirements = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# Set the maximum with of the page content
st.set_page_config(page_title="GenAI Library Expert", layout="wide", menu_items=None)
st.html("""
    <style>
        .stMain {
            max-width: clamp(1200px, 70vw, 1800px);
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """
)

st.title("🧠 Adaptive Library Architect")

greeting_box = st.empty()

# Show a welcome message if the chat is still empty
if len(st.session_state.messages) == 0:
    greeting_box.subheader(WELCOME_MESSAGE)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle the pending prompt if there is one
if prompt := st.chat_input("Describe your project idea"):
    greeting_box.empty()
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # First, check if the user message is relevant to the overall goal of the conversation.
    with st.spinner("Processing..."):
        validation = check_compliance(st.session_state.messages)

    if not validation['valid']:
        st.chat_message("assistant").markdown(validation['response'])
        st.session_state.messages.append({"role": "assistant", "content": validation['response']})
    else:
        with st.spinner("Analyzing request..."):
            summary = summarize_request(prompt, st.session_state.requirements)
            if summary:
                st.session_state.requirements.append(summary)

        # Analyze if the user's requirements conclusive or if you need to ask additional questions
        with st.spinner("Reviewing requirements..."):
            analysis = analyze_specificity(prompt)

        if analysis['status'] == 'vague': # Case: Query is vague -> Ask Questions
            # Format questions nicely
            questions_text = "**I need a bit more detail to give you the best advice:**\n\n"
            for q in analysis['content']:
                questions_text += f"- {q}\n"

            st.chat_message("assistant").markdown(questions_text)
            st.session_state.messages.append({"role": "assistant", "content": questions_text})
        else:
            # Case: Query is specific -> Expand Keywords & Search
            expanded_terms = analysis['content']
            # We append the keywords to the prompt for the vector search
            enhanced_search_query = f"{prompt}\nKeywords: {expanded_terms}"

            with st.spinner(f"Synthesizing requirements and retrieving libraries..."):
                ai_response, sources = generate_rag_response(enhanced_search_query, original_intent=prompt)

            show_recommendations(ai_response, sources)
