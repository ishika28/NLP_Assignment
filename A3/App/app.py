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
print("Model loaded successfully!")

# #  Load only the state dictionary instead of the full model
# model_state_dict = torch.load('model/additive_Seq2SeqTransformer.pt', map_location=device)

# #  Apply state dictionary to the model
# model.load_state_dict(model_state_dict)
# model.eval()

# print(" Model loaded successfully!")

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

    # trg_tokens = [vocab_transform[trg_language].get_itos()[i] for i in trg_indexes if i not in [UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX]]
    trg_tokens = [vocab_transform[trg_language].get_itos()[i] for i in trg_indexes if i not in [UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX] and vocab_transform[trg_language].get_itos()[i] != '[CLS]']
    print(f"Translated tokens: {trg_tokens}")
    return ' '.join(trg_tokens)

# ✅ Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("English to Nepali Translator"),
    dcc.Input(id='input-text', type='text', placeholder='Enter text in English...', style={'width': '60%'}),
    html.Button('Translate', id='translate-btn', n_clicks=0, style={'margin-left': '10px'}),
    html.Br(), html.Br(),
    html.Div(id='output-text', style={'font-weight': 'bold', 'font-size': '18px'})
])

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
