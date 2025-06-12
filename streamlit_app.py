import streamlit as st
import torch
import psycopg2
import re
import numpy as np
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download('punkt')

# --------------------------------------
# 1. Pre-load on first run using session
# --------------------------------------

@st.cache_resource
def load_all_models_and_data():
    # DB Fetch
    conn = psycopg2.connect("postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
    cur = conn.cursor()
    cur.execute("""
        SELECT i.title, i.url, i.score, i.by, u.karma
        FROM hacker_news.items i
        JOIN hacker_news.users u ON i.by = u.id
        WHERE i.title IS NOT NULL AND i.score IS NOT NULL AND i.by IS NOT NULL
        LIMIT 20000;
    """)
    rows = cur.fetchall()
    conn.close()
    titles, urls, scores, by, karmas = zip(*rows)

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]+', '', text)
        return text.split()

    checkpoint = torch.load("umuts_cbow_full.pth")
    word_to_ix = checkpoint['word_to_ix']

    tokenized_titles = [preprocess(title) for title in titles]
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    vocab_size = len(word_to_ix)

    class CBOW(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, embed_dim)
            self.linear = nn.Linear(embed_dim, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs).mean(dim=1)
            return self.linear(embeds)

    embed_dim = 5
    cbow_model = CBOW(vocab_size, embed_dim)
    cbow_model.load_state_dict(torch.load("umuts_cbow.pth"))
    cbow_model.eval()

    title_embeddings = []
    valid_indices = []
    for i, tokens in enumerate(tokenized_titles):
        token_ids = [word_to_ix[t] for t in tokens if t in word_to_ix]
        if token_ids:
            with torch.no_grad():
                vectors = cbow_model.embeddings(torch.tensor(token_ids))
                avg_vector = vectors.mean(dim=0)
            title_embeddings.append(avg_vector)
            valid_indices.append(i)

    X_title = torch.stack(title_embeddings)
    y = torch.tensor([scores[i] for i in valid_indices], dtype=torch.float32).unsqueeze(1)

    parsed_domains = []
    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or 'unknown'
        except:
            domain = 'unknown'
        parsed_domains.append(domain)

    le = LabelEncoder()
    domain_ids = le.fit_transform(parsed_domains)
    domain_ids_tensor = torch.tensor(domain_ids, dtype=torch.long)[valid_indices]
    domain_vocab_size = len(le.classes_)
    domain_embed_dim = 3

    karmas_tensor = torch.tensor([karmas[i] for i in valid_indices], dtype=torch.float32).unsqueeze(1)
    user_ids = [by[i] for i in valid_indices]
    user_karma_lookup = {user_ids[i]: karmas_tensor[i].item() for i in range(len(user_ids))}

    class UpvotePredictor(nn.Module):
        def __init__(self, title_embed_dim, domain_vocab_size, domain_embed_dim):
            super().__init__()
            self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embed_dim)
            self.model = nn.Sequential(
                nn.Linear(title_embed_dim + domain_embed_dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, title_embed, domain_id, karma):
            domain_vec = self.domain_embedding(domain_id)
            x = torch.cat([title_embed, domain_vec, karma], dim=1)
            return self.model(x)

    model = UpvotePredictor(embed_dim, domain_vocab_size, domain_embed_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    class HNDataset(Dataset):
        def __init__(self, title_embeds, domain_ids, karmas, labels):
            self.title_embeds = title_embeds
            self.domain_ids = domain_ids
            self.karmas = karmas
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.title_embeds[idx], self.domain_ids[idx], self.karmas[idx], self.labels[idx]

    dataset = HNDataset(X_title, domain_ids_tensor, karmas_tensor, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):  # shorter train in Streamlit
        total_loss = 0
        for title_embed, domain_id, karma, label in dataloader:
            pred = model(title_embed, domain_id, karma)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model, cbow_model, word_to_ix, le, user_karma_lookup


# --------------------------------------
# 2. Streamlit UI
# --------------------------------------
st.title("🔮 Predict Hacker News Upvotes")
st.write("Enter a Hacker News post title, URL, and user ID. The model will predict expected upvotes.")

title_input = st.text_input("Post Title", "Show HN: AI Hacker generates $1 billion")
url_input = st.text_input("URL", "https://openai.com")
user_id_input = st.text_input("Username", "ingve")

if st.button("Predict"):
    with st.spinner("Running model..."):
        model, cbow_model, word_to_ix, le, user_karma_lookup = load_all_models_and_data()

        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^a-z0-9 ]+', '', text)
            return text.split()

        tokens = preprocess(title_input)
        token_ids = [word_to_ix.get(t) for t in tokens if t in word_to_ix]
        if not token_ids:
            st.error("No valid words in title for prediction.")
        else:
            with torch.no_grad():
                vectors = cbow_model.embeddings(torch.tensor(token_ids))
                avg_embed = vectors.mean(dim=0)

            try:
                parsed = urlparse(url_input)
                domain = parsed.netloc or 'unknown'
            except:
                domain = 'unknown'

            try:
                domain_id = le.transform([domain])[0]
            except:
                domain_id = 0

            domain_tensor = torch.tensor([domain_id], dtype=torch.long)
            karma_value = user_karma_lookup.get(user_id_input, 0)
            karma = torch.tensor([[karma_value]], dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                prediction = model(avg_embed.unsqueeze(0), domain_tensor, karma).item()
                st.success(f"Predicted Upvotes: {prediction:.2f}")
