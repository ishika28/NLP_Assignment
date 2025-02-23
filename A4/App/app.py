import dash
from dash import dcc, html, Input, Output, State
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the custom-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Load the state dictionary
state_dict = torch.load('model/bert_sentence.pth', map_location=device)[0]  # Extract the state dictionary from the list
model.load_state_dict(state_dict)
model.eval()

# Initialize the classifier head
classifier_head = torch.nn.Linear(768 * 3, 3).to(device)

# Define mean pooling function
def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
# Layout
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
                
                html.H3("Do You Agree?", style={'fontFamily': 'Gill Sans, sans-serif', 'fontSize': '24px'}),
                dcc.Input(
                    id='premise-input',
                    type='text',
                    placeholder='Enter premise...',
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
                    }
                ),
                dcc.Input(
                    id='hypothesis-input',
                    type='text',
                    placeholder='Enter hypothesis...',
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
                    }
                ),
                html.Button(
                    'Predict',
                    id='predict-button',
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
                html.Br(),
                html.Div(
                    id='output',
                    style={
                        'fontFamily': 'Gill Sans, sans-serif',
                        'fontWeight': 'bold', 
                        'fontSize': '18px', 
                        'margin': '10px 0',
                        'color': '#333',  # Darker text color for better readability
                    }
                )
            ]
        )
    ]
)
# Define the callback to handle predictions
@app.callback(
    Output('output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('premise-input', 'value'),
     State('hypothesis-input', 'value')]
)
def predict_nli(n_clicks, premise, hypothesis):
    if n_clicks > 0 and premise and hypothesis:
        # Tokenize the inputs
        premise_inputs = tokenizer(premise, return_tensors='pt', padding=True, truncation=True, max_length=128)
        hypothesis_inputs = tokenizer(hypothesis, return_tensors='pt', padding=True, truncation=True, max_length=128)

        # Move inputs to the device
        premise_inputs = {key: val.to(device) for key, val in premise_inputs.items()}
        hypothesis_inputs = {key: val.to(device) for key, val in hypothesis_inputs.items()}

        # Get the embeddings
        with torch.no_grad():
            u = model(**premise_inputs).last_hidden_state
            v = model(**hypothesis_inputs).last_hidden_state

        # Mean pooling
        u_mean_pool = mean_pool(u, premise_inputs['attention_mask'])
        v_mean_pool = mean_pool(v, hypothesis_inputs['attention_mask'])

        # Concatenate u, v, |u-v|
        abs_uv = torch.abs(u_mean_pool - v_mean_pool)
        x = torch.cat([u_mean_pool, v_mean_pool, abs_uv], dim=-1)

        # Pass through classifier head
        logits = classifier_head(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        # Map prediction to label
        labels = ["entailment", "neutral", "contradiction"]
        predicted_label = labels[preds.item()]

        # Get the probability of the predicted label
        predicted_prob = probs[0][preds.item()].item()

        return html.Div([
            html.H3(f"Label: {predicted_label}"),
            html.P(f"Cosine Similarity {predicted_label}: {predicted_prob:.4f}")
        ])
    return html.Div()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)