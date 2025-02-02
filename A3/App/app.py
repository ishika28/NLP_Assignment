import torch
import dash
import pickle
from dash import dcc, html, Input, Output, State
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nepalitokenizers import WordPiece
from model_definitions import Encoder, Decoder, Seq2SeqTransformer, AdditiveAttention, EncoderLayer, DecoderLayer,MultiHeadAttentionLayer, PositionwiseFeedforwardLayer 

 # Import necessary classes

# Use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize tokenizers
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'ne'

token_transform = {
    "en": get_tokenizer('spacy', language='en_core_web_sm'),
    "ne": WordPiece()
}

# Load vocab from model folder
vocab_transform = torch.load('model/vocab')

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Define model parameters
input_dim = len(vocab_transform[SRC_LANGUAGE])
output_dim = len(vocab_transform[TRG_LANGUAGE])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1
attn_variant = 'additive'

#  Create a fresh model instance
encoder = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, attn_variant, device)
decoder = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, attn_variant, device)
# model = Seq2SeqTransformer(encoder, decoder, PAD_IDX, PAD_IDX, device).to(device)

model_path = 'model/additive_Seq2SeqTransformer.pt'
params, state = torch.load(model_path)
model = Seq2SeqTransformer(**params, device=device).to(device)
model.load_state_dict(state)
model.eval()


#  Function to translate text
def translate_sentence(sentence, src_language, trg_language, model, device, max_length=100):
    model.eval()
    tokenized = token_transform[src_language](sentence.lower())
    tokenized = [SOS_IDX] + [vocab_transform[src_language][token] for token in tokenized if token in vocab_transform[src_language]] + [EOS_IDX]
    src_tensor = torch.LongTensor(tokenized).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    trg_indexes = [SOS_IDX]
    for _ in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break


    trg_tokens = [vocab_transform[trg_language].get_itos()[i] for i in trg_indexes if i not in [UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX] and vocab_transform[trg_language].get_itos()[i] != '[CLS]']
   
    return ' '.join(trg_tokens)

# ✅ Initialize Dash app
app = dash.Dash(__name__)


app.layout = html.Div(
    style={
        'display': 'flex',
        'justifyContent': 'center',  # Center horizontally
        'alignItems': 'center',  # Center vertically
        'height': '100vh',  # Full viewport height
        'backgroundColor': 'white',  # Background color for the outer container
    },
    children=[
        html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',  # Center children horizontally
                'justifyContent': 'center',  # Center children vertically
                'backgroundColor': 'F8FAFC',  
                'padding': '20px',  # Add padding
                'borderRadius': '10px',  # Optional: Rounded corners
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Optional: Add shadow
                'width': '50%',  # Set width of the inner container
                'maxWidth': '600px',  # Optional: Limit maximum width
            },
            children=[
                # Add an image
              html.Img(
                src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhp8Tmdtl-RZ_-JLc-y9y88jsCZhaDW-TKuEGDlxGFDVLaUsHVQv_Og-7yX9vEoJtF8LGHbNqvYZFFeXCbuOCWuoP11RqfZLWsY8eOZW9eJL2NNpdw8BHJ7NvCeK3pxPCCnYoBEX43g-PY/s1600/Flag_of_Nepal.gif",
                style={
                  'width': '110px',
                  'height': 'auto',
                  'marginBottom': '20px',
                 'borderRadius': '10px',
    }
),
                html.H3("English to Nepali Translator",
                        style={'fontFamily': 'Gill Sans, sans-serif'}),
                dcc.Input(
                    id='input-text',
                    type='text',
                    className='form-control',
                    placeholder='Enter text in English...',
                    style={
                        'width': '60%',  # Adjust width
                        'margin': '10px 0',
                        'padding': '10px',  # Add padding inside the input
                        'border': '2px solid F6F5F2',  # Border color
                        'borderRadius': '5px',  # Rounded corners
                        'fontSize': '16px',  # Increase font size
                        'fontfamily':'Gill Sans, sans-serif',
                        'outline': 'none',  # Remove default outline
                        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',  # Add subtle shadow
                        'transition': 'border-color 0.3s ease',  # Smooth transition for focus
                    }
                ),
                html.Button(
                    'Translate',
                    id='translate-btn',
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
                html.Br(),
                html.Div(
    id='output-text',
    style={
        'fontFamily': 'Gill Sans, sans-serif',  # Corrected property name
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



@app.callback(
    Output('output-text', 'children'),
    [Input('translate-btn', 'n_clicks')],
    [State('input-text', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks == 0 or not value:
        return "Enter text and click 'Translate'."
    
    translated_text = translate_sentence(value, SRC_LANGUAGE, TRG_LANGUAGE, model, device)
    return f'Translated text: {translated_text}'

if __name__ == '__main__':
    app.run_server(debug=True)
