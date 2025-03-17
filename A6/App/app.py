# app.py
import dash
from dash import dcc, html, Input, Output, State
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.prompts import PromptTemplate

# Initialize Dash app
app = dash.Dash(__name__)

# Define custom prompt
PROMPT = PromptTemplate.from_template(
    "I'm a friendly chatbot here to answer questions about Ishika based on provided documents.\n"
    "I'll give gentle and informative responses about Ishika's life, education, work, and beliefs.\n"
    "If I use a document to answer, I’ll cite it (e.g., resume, bio).\n"
    "    {context}\n"
    "    Question: {question}\n"
    "    Answer:"
)

# Load Groq Llama3-70B
print("Loading Groq Llama3-70B...")
llm_groq = ChatGroq(
    model_name="llama3-70b-8192",
    api_key="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"
)
print("Groq Llama3-70B loaded successfully.")

# Load FAISS vector store
print("Loading FAISS vector store...")
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
vectordb = FAISS.load_local(
    folder_path="../vector-store/ishika_data",
    embeddings=embeddings,
    index_name="personal",
    allow_dangerous_deserialization=True
)
retriever = vectordb.as_retriever()
print("FAISS vector store loaded successfully.")

# Define Groq chain
memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True, output_key='answer')
question_generator_groq = LLMChain(llm=llm_groq, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
doc_chain_groq = load_qa_chain(llm=llm_groq, chain_type='stuff', prompt=PROMPT, verbose=False)
chain_groq = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_groq,
    combine_docs_chain=doc_chain_groq,
    return_source_documents=True,
    memory=memory,
    verbose=False,
    get_chat_history=lambda h: h
)

# Dash layout with your UI design, including the description
app.layout = html.Div(
    style={
        'display': 'flex',
        'justifyContent': 'center',  # Center horizontally
        'alignItems': 'center',  # Center vertically
        'height': '100vh',  # Full viewport height
        'backgroundColor': '#F8FAFC',  # Light gray background
    },
    children=[
        html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',  # Center children horizontally
                'justifyContent': 'center',  # Center children vertically
                'backgroundColor': '#FFFFFF',  # White background for inner container
                'padding': '20px',  # Padding inside
                'borderRadius': '10px',  # Rounded corners
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Shadow effect
                'width': '50%',  # Width of inner container
                'maxWidth': '600px',  # Max width limit
            },
            children=[
                html.H1(
                    "IshikaBot: Chatbot",
                    style={'fontFamily': 'Gill Sans, sans-serif', 'fontSize': '24px'}
                ),
                html.P(
                    "Ask a question and get responses based on Ishika's bio!",
                    style={
                        'fontFamily': 'Gill Sans, sans-serif',
                        'fontSize': '16px',
                        'color': '#333',
                        'margin': '5px 0'
                    }
                ),
                dcc.Textarea(
                    id='input-box',
                    placeholder='Type your question here...',
                    style={
                        'width': '80%',  # Adjust width
                        'margin': '10px 0',
                        'padding': '10px',  # Padding inside textarea
                        'border': '2px solid #F6F5F2',  # Light border
                        'borderRadius': '5px',  # Rounded corners
                        'fontSize': '16px',  # Font size
                        'fontFamily': 'Gill Sans, sans-serif',
                        'outline': 'none',  # No outline on focus
                        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',  # Subtle shadow
                        'transition': 'border-color 0.3s ease',  # Smooth border transition
                        'height': '100px',  # Fixed height for textarea
                    }
                ),
                html.Button(
                    'Generate Response',
                    id='submit-button',
                    n_clicks=0,
                    style={
                        "backgroundColor": "#008CBA",  # Blue button
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "padding": "10px 20px",
                        "margin": "10px auto",
                        "display": "block",
                        "cursor": "pointer",
                        'fontSize': '16px',  # Match font size
                    }
                ),
                html.Div(
                    id='chat-output',
                    style={
                        'fontFamily': 'Gill Sans, sans-serif',
                        'fontWeight': 'bold',
                        'fontSize': '18px',
                        'margin': '10px 0',
                        'color': '#333',  # Dark text for readability
                        'whiteSpace': 'pre-line'  # Preserve line breaks
                    }
                )
            ]
        )
    ]
)

# Callback to update chat output
@app.callback(
    Output('chat-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-box', 'value')
)
def update_output(n_clicks, question):
    if n_clicks > 0 and question:
        try:
            result_groq = chain_groq({"question": question})
            groq_answer = result_groq["answer"].strip()  # Just the raw answer
            unique_pages = set(doc.metadata['page'] + 1 for doc in result_groq["source_documents"])
            groq_sources = [f"cv_bio.pdf (Page {page})" for page in sorted(unique_pages)]
            return html.Div([
                html.H3("Groq Llama3-70B Response:"),
                html.P(groq_answer),
                html.P("Sources: " + ", ".join(groq_sources))
            ])
        except Exception as e:
            return html.P(f"Error: {str(e)}")
    return html.P("Ask a question to get started!")

# Run the app
if __name__ == '__main__':
    print("Starting Dash server...")
    app.run_server(debug=True)