import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer from Hugging Face
model_name = "ishikapradhan/nlp_A5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        'display': 'flex',
        'justifyContent': 'center',  # Center horizontally
        'alignItems': 'center',  # Center vertically
        'height': '100vh',  # Full viewport height
        'backgroundColor': '#F8FAFC',  # Background color for the outer container
    },
    children=[
        html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',  # Center children horizontally
                'justifyContent': 'center',  # Center children vertically
                'backgroundColor': '#FFFFFF',  # White background for the inner container
                'padding': '20px',  # Add padding
                'borderRadius': '10px',  # Rounded corners
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Add shadow
                'width': '50%',  # Set width of the inner container
                'maxWidth': '600px',  # Limit maximum width
            },
            children=[
                html.H1("Optimization Human Preference", style={'fontFamily': 'Gill Sans, sans-serif', 'fontSize': '24px'}),
                dcc.Textarea(
                    id='user-input',
                    placeholder='Enter your text here...',
                    style={
                        'width': '80%',  # Adjust width
                        'margin': '10px 0',
                        'padding': '10px',  # Add padding inside the input
                        'border': '2px solid #F6F5F2',  # Border color
                        'borderRadius': '5px',  # Rounded corners
                        'fontSize': '16px',  # Increase font size
                        'fontFamily': 'Gill Sans, sans-serif',
                        'outline': 'none',  # Remove default outline
                        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',  # Add subtle shadow
                        'transition': 'border-color 0.3s ease',  # Smooth transition for focus
                        'height': '100px',  # Keep textarea height
                    }
                ),
                html.Button(
                    'Generate Response',
                    id='submit-btn',
                    n_clicks=0,
                    style={
                        "backgroundColor": "#008CBA",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "padding": "10px 20px",
                        "margin": "10px auto",
                        "display": "block",
                        "cursor": "pointer",
                        'fontSize': '16px',  # Match font size with input
                    }
                ),
                html.Div(
                    id='model-response',
                    style={
                        'fontFamily': 'Gill Sans, sans-serif',
                        'fontWeight': 'bold',
                        'fontSize': '18px',
                        'margin': '10px 0',
                        'color': '#333',  # Darker text color for better readability
                        'whiteSpace': 'pre-line'
                    }
                )
            ]
        )
    ]
)

@app.callback(
    Output('model-response', 'children'),
    Input('submit-btn', 'n_clicks'),
    Input('user-input', 'value')
)
def generate_response(n_clicks, user_input):
    if n_clicks > 0 and user_input:
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=100, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
