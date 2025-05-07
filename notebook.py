import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium", app_title="Presidential Speech Analysis")


@app.cell
def _():
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import CountVectorizer
    import json
    import marimo as mo
    return (
        BERTopic,
        CountVectorizer,
        KeyBERTInspired,
        WordNetLemmatizer,
        json,
        mo,
        pd,
        sent_tokenize,
        stopwords,
        word_tokenize,
    )


@app.cell
def _(json, pd):
    def load_json(file):
        with open(file) as f:
            data = json.load(f)
            return data

    speeches = pd.DataFrame(load_json("speeches.json"))
    speeches.head()
    return (speeches,)


@app.cell
def _(speeches):
    speeches.drop(columns=['doc_name'], inplace=True)
    speeches
    return


@app.cell
def _(speeches):
    print(speeches['president'].unique())
    return


@app.cell
def _(pd):
    presidents = pd.read_csv("presidents.csv")
    presidents
    return (presidents,)


@app.cell
def _(presidents, speeches):
    pres_names = presidents['President Name'].str.replace(' ', '', regex=True)
    speeches_clean = speeches[speeches['president'].str.replace(' ', '', regex=True).isin(pres_names)]
    speeches_clean = speeches_clean.sort_values('date').reset_index(drop=True)
    speeches_clean = speeches_clean[['date', 'president','title','transcript']]
    speeches_clean['doc_id'] = speeches_clean.index
    cols = ['doc_id'] + [col for col in speeches_clean.columns if col != 'doc_id']
    speeches_clean = speeches_clean[cols]
    speeches_clean.set_index('doc_id', inplace=True)
    speeches_clean
    return (speeches_clean,)


@app.cell
def _(speeches_clean):
    print(speeches_clean['president'].unique())
    return


@app.cell
def _(speeches_clean, word_tokenize):
    speeches_clean['tokenized_text'] = speeches_clean['transcript'].apply(word_tokenize)
    return


@app.cell
def _(WordNetLemmatizer, speeches, speeches_clean, stopwords, word_tokenize):
    import re
    def clean_text(text):
        # Remove special characters and numbers, but keep sentence-ending punctuation and apostrophes
        text = re.sub(r'[^A-Za-z\s.?!,:\']', '', text) 
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize whitespace (remove leading/trailing and extra spaces within)
        text = ' '.join(text.split())

        return text

    def preprocess_text(text):
        text = clean_text(text)
        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        return lemmatized_tokens

    def token_to_text(tokens):
        preprocessed_text = " ".join(tokens)

        return preprocessed_text

    speeches_clean['lem_tokenized_text'] = speeches_clean['transcript'].apply(preprocess_text)
    speeches['transcript_clean'] = speeches['transcript'].apply(clean_text)
    speeches_clean['transcript_clean'] = speeches_clean['transcript'].apply(clean_text)
    return


@app.cell
def _(BERTopic, CountVectorizer, KeyBERTInspired, speeches):
    vectorizer_model = CountVectorizer(stop_words="english")
    representation_model = KeyBERTInspired(random_state=66)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, representation_model=representation_model, min_topic_size=5)
    topics, probs = topic_model.fit_transform(speeches['transcript_clean'].to_list())
    topic_model.get_topic_info()
    return


@app.cell
def _(pd, sent_tokenize, speeches_clean):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    all_sentences = []

    for doc_id, row in speeches_clean.iterrows():
        sentences = sent_tokenize(row['transcript_clean'])
        for idx, sentence in enumerate(sentences):
            score = sia.polarity_scores(sentence)['compound']
            all_sentences.append({
                'doc_id': doc_id,
                'sentence_index': idx,
                'sentence_text': sentence,
                'sentiment_score': score
            })


    sentence_sentiment = pd.DataFrame(all_sentences)
    sentence_sentiment.head()
    return


@app.cell
def _(pd, speeches_clean, word_tokenize):
    import textstat

    def lexical_diversity(text):
        tokens = word_tokenize(text.lower())
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0


    def get_readability_metrics(text):
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'smog_index': textstat.smog_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'dale_chall_score': textstat.dale_chall_readability_score(text),
            'difficult_words': textstat.difficult_words(text),
            'readability_consensus': textstat.text_standard(text)
        }

    def add_speech_metrics(row):
        text = row['transcript_clean']
        metrics = get_readability_metrics(text)
        metrics['lexical_diversity'] = lexical_diversity(text)
        return pd.Series(metrics)

    df_metrics = speeches_clean.apply(add_speech_metrics, axis=1)
    df_overall = pd.concat([speeches_clean, df_metrics], axis=1)

    df_overall
    return (df_overall,)


@app.cell
def _(df_overall):
    from collections import defaultdict
    from itertools import combinations
    import spacy
    import networkx as nx

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_lg")

    entity_edges = defaultdict(int)

    for text in df_overall['transcript_clean']:
        doc = nlp(text)
        entities = set(ent.text for ent in doc.ents if ent.label_ in {'PERSON', 'GPE', 'ORG'})
        for ent1, ent2 in combinations(entities, 2):
            key = tuple(sorted((ent1, ent2)))
            entity_edges[key] += 1

    G = nx.Graph()

    # Add weighted edges
    for (ent1, ent2), weight in entity_edges.items():
        if weight >= 2:  # Filter low-frequency connections
            G.add_edge(ent1, ent2, weight=weight)

    return G, nx


@app.cell
def _(G, nx):
    import plotly.graph_objects as go

    # Compute layout
    pos = nx.spring_layout(G, k=0.5)

    # Build edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Build node traces
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'{node}\nConnections: {len(list(G[node]))}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition='top center',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[5 + len(G[node]) for node in G.nodes()],
            color=[len(G[node]) for node in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Final figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Entity Co-occurrence Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)))

    fig.show()


    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Plot Ideas
    - Topics changes over time (lines)
    - Sentiment by President
    - Topic by president
    - Speech Complexity, readability, lexicon, etc. by President / Time
    """
    )
    return


if __name__ == "__main__":
    app.run()
