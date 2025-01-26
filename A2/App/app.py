from dash import Dash, html, dcc, Input, Output, State
import torch
import torch.nn as nn
import pickle
from torchtext.data.utils import get_tokenizer

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer initialization
tokenizer = get_tokenizer('basic_english')

# Define the LSTM model class
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super(LSTMLanguageModel, self).__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, src, hidden):
        embedding = self.embedding(src)
        output, hidden = self.lstm(embedding, hidden)
        prediction = self.fc(output)
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device))

# Load the vocabulary (assuming it's a torchtext Vocab object)
with open("model/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)


# Model parameters
vocab_size = len(loaded_vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65

# Load the trained LSTM model
lstm_model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model_path = "model/best-val-lstm_lm.pt"
lstm_model.load_state_dict(torch.load(model_path, map_location=device))
lstm_model.eval()

# Generate text function
def generate_text(prompt, max_seq, temperature, model, tokenizer, stoi, itos, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    tokens = tokenizer(prompt)
    indices = [stoi.get(token, stoi["<unk>"]) for token in tokens]
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        for _ in range(max_seq):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == stoi["<eos>"]:
                break
            indices.append(next_token)

    return ' '.join(itos[idx] for idx in indices)

# Dash App
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Language Model Demo", style={'textAlign': 'center'}),
    html.Div([
        dcc.Input(id='search-query', type='text', placeholder='Enter your text...', style={'width': '70%', 'margin': '10px auto'}),
        html.Button('Generate', id='search-button', n_clicks=0, style={'padding': '10px 20px', 'background-color': '#007BFF', 'color': 'white'}),
    ], style={'textAlign': 'center'}),
    html.Div(id='search-results', style={'margin-top': '20px', 'textAlign': 'center'})
])

@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('search-query', 'value')]
)
def update_output(n_clicks, query):
    if n_clicks > 0:
        if not query:
            return html.Div("Please enter a prompt.", style={'color': 'red'})
        
        temperatures = [0.5, 0.7, 1.0]
        results = []
        for temp in temperatures:
            generated = generate_text(
                prompt=query,
                max_seq=30,
                temperature=temp,
                model=lstm_model,
                tokenizer=tokenizer,
                stoi=stoi,
                itos=itos,
                device=device,
                seed=42
            )
            results.append(html.Div([
                html.H4(f"Temperature {temp}:"),
                html.P(generated, style={'color': 'black', 'textAlign': 'left'})
            ]))
        return results
    return html.Div("Enter a prompt to generate text.", style={'color': 'gray'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
