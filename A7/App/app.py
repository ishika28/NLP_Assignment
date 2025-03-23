# Import necessary libraries
from dash import Dash, html, dcc, Input, Output, callback
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize Dash app
app = Dash(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Adjust if saved differently
model_path = "./model/train_even_model"  # Path where train_even_model is saved
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set to evaluation mode

# Define label mapping (assuming 0 = non-toxic, 1 = toxic)
id2label = {0: "Non-Toxic", 1: "Toxic"}


# Dash layout with your UI design
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
                    "Toxicity Classifier",
                    style={'fontFamily': 'Gill Sans, sans-serif', 'fontSize': '24px'}
                ),
                html.P(
                    "Enter text to check if it's toxic or non-toxic!",
                    style={
                        'fontFamily': 'Gill Sans, sans-serif',
                        'fontSize': '16px',
                        'color': '#333',
                        'margin': '5px 0'
                    }
                ),
                dcc.Textarea(
                    id='text-input',
                    placeholder='Type your text here...',
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
                    'Classify Text',
                    id='classify-button',
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
                    id='output-result',
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

# Define callback to update output based on input
@app.callback(
    Output("output-result", "children"),
    Input("classify-button", "n_clicks"),
    Input("text-input", "value")
)
def classify_text(n_clicks, text):
    if n_clicks > 0 and text:
        # Tokenize the input text
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        # Move inputs to the same device as the model (CPU assumed here; adjust if using GPU)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        
        # Return the classification result
        result = id2label[prediction]
        return html.P(f"Classification: {result}")
    return html.P("Please enter text and click 'Classify'.")

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)