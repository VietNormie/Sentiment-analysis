import streamlit as st
import torch.nn as nn
import torch
import torchtext.vocab as vocab
import torch.nn.functional as F
import pandas as pd
import numpy as np
from underthesea import word_tokenize
import unicodedata
import re
from tqdm import tqdm

# Dictionary for common Vietnamese slang/abbreviations
abbreviations = {
    "ko": "không",
    "sp": "sản phẩm",
    "k": "không",
    "m": "mình",
    "đc": "được",
    "dc": "được",
    "h": "giờ",
    "trloi": "trả lời",
    "cg": "cũng",
    "bt": "bình thường",
    "dt": "điện thoại",
    "mt": "máy tính",
    "m.n": "mọi người"
    # add more slang mappings
}

# Regex patterns
url_pattern = r"http\S+|www\S+"  # URLs
user_pattern = r"@\w+"  # usernames
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]", flags=re.UNICODE)
emoticon_pattern = r"[:;=8][\-o\*']?[\)\]\(\[dDpP/:}\{@\|\\]"
repeat_pattern = re.compile(r"(.)\1{2,}")


def clean_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(url_pattern, '', text)
    text = re.sub(user_pattern, '', text)
    text = emoji_pattern.sub(' ', text)
    text = re.sub(emoticon_pattern, ' ', text)

    if abbreviations:
        def expand(match):
            word = match.group(0)
            return abbreviations.get(word, word)
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, abbreviations.keys())) + r")\b")
        text = pattern.sub(expand, text)

    text = repeat_pattern.sub(r"\1", text)
    text = re.sub(r"[^\w\s\u00C0-\u024F]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# Vocabulary class unchanged...
class Vocabulary:
    def __init__(self):
        self.word2id = {'<pad>': 0, '<unk>': 1}
        self.unk_id = 1
        self.id2word = {0: '<pad>', 1: '<unk>'}
    def __getitem__(self, word): return self.word2id.get(word, self.unk_id)
    def __contains__(self, word): return word in self.word2id
    def __len__(self): return len(self.word2id)
    def add(self, word):
        if word not in self.word2id:
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word
            return idx
        return self[word]
    @staticmethod
    def tokenize_corpus(corpus):
        tokenized = []
        for doc in tqdm(corpus):
            tokenized.append([w.replace(' ', '_') for w in word_tokenize(doc)])
        return tokenized
    def corpus_to_tensor(self, corpus, is_tokenized=False):
        tok = corpus if is_tokenized else self.tokenize_corpus(corpus)
        tensors = []
        for doc in tqdm(tok):
            idxs = list(map(lambda w: self[w], doc))
            tensors.append(torch.tensor(idxs, dtype=torch.int64).to(device))
        return tensors

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, bidir, dropout, pad_idx, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                           bidirectional=bidir, dropout=dropout if n_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * (2 if bidir else 1), n_classes)
    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.to('cpu'), enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        if self.rnn.bidirectional:
            h = torch.cat((h[-2], h[-1]), dim=1)
        else:
            h = h[-1]
        return self.fc(self.dropout(h))

# Load pretrained embeddings and build vocab
word_embedding = vocab.Vectors(
    name='vi_word2vec.txt',
    unk_init=torch.Tensor.normal_
)
vocab = Vocabulary()
for w in word_embedding.stoi.keys(): vocab.add(w)

# Model hyperparams
input_dim = word_embedding.vectors.shape[0]
emb_dim = 100
hid_dim = 256
n_layers = 2
bidir = True
dropout = 0.5
pad_idx = vocab['<pad>']

label_map = {0: 'tiêu cực', 1: 'bình thường', 2: 'tích cực'}

@st.cache_resource
# Ensure model and its weights moved to correct device
def load_model(path: str):
    model = RNN(input_dim, emb_dim, hid_dim, n_layers, bidir, dropout, pad_idx, len(label_map))
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model('model.pt')

# Prediction helper
def predict_sentiment(model, sentence, vocab, label_mapping=None):
    tensor = vocab.corpus_to_tensor([sentence])[0]
    length = torch.LongTensor([tensor.size(0)]).to(device)
    tensor = tensor.unsqueeze(1)  # seq_len x batch
    with torch.no_grad():
        logits = model(tensor, length).squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().tolist()
    idx = int(torch.tensor(probs).argmax())
    return (label_mapping[idx], probs) if label_mapping else (idx, probs)

def main():
    # Streamlit UI
    st.title('Sentiment Analysis')
    st.subheader('Nhập comment (mỗi dòng một comment):')
    text_input = st.text_area('', height=150)
    st.subheader('Hoặc upload file txt/csv chứa comments:')
    file = st.file_uploader('', type=['txt', 'csv'])

    comments = []
    if text_input: comments += [l.strip() for l in text_input.splitlines() if l.strip()]
    if file:
        if file.type == 'text/plain':
            txt = file.read().decode('utf-8')
            comments += [l.strip() for l in txt.splitlines() if l.strip()]
        else:
            df = pd.read_csv(file)
            comments += df.iloc[:, 0].dropna().astype(str).tolist()

    if st.button('Predict'):
        if not comments:
            st.warning('Vui lòng nhập ít nhất một comment hoặc upload file.')
        else:
            results = []
            for c in comments:
                lbl, ps = predict_sentiment(model, c, vocab, label_map)
                results.append({'Comment': c, 'Dự đoán': lbl, 'Xác suất': ps})
            st.dataframe(pd.DataFrame(results), use_container_width=True)

if __name__ == '__main__':
    main()
