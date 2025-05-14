import marimo

__generated_with = "0.13.8"
app = marimo.App(
    width="medium",
    app_title="Presidential Speech Analysis",
    layout_file="layouts/notebook.slides.json",
    auto_download=["html"],
)


@app.cell
def _(mo):
    mo.md(r"""# Utilizing NLP and ML to Analyze Presidential Speeches""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Natural language processing and machine learning both are tools that are used widely in data science. This article aims to apply these techniques both as a way of showing off an interesting application of these techniques through exploratory analysis as well as to share some interesting findings and educate on how to interpret results from using these techiniques. This project is done in Marimo an excellent tool for data science that solves some issues with Jupyter Notebooks. Stay tuned for another article about Marimo Notebooks soon. Here are the tools that will be used in this article:

    * Sklearn: PCA
    * Pandas: Handling data organization and basic manipulation
    * NLTK: Tokenization, Sentiment
    * BERTopic: Topic Modeling
    * Plotly, Matplotlib: Plotting
    * Textstat: Readibility Metrics
    """
    )
    return


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
def _(mo):
    mo.md(
        r"""
    ## Initial Data Processing
    All the text data in the speeches file is fetched from the Miller center. It could be fetched in code using the documentation [here.](https://data.millercenter.org/api.html) The presidents file also contains some additional information about presidents that have approval ratings. This data will be used later. As this project was iterated over a few moths a fixed data file was downloaded and saved and will be loaded for this analysis. In order to do good text analysis, you need to first preprocess your text data. Since this data contains transcripts sometimes from television there are many unneeded symbols in the text. Regex is a tool that allows us to remove characters from strings using pattern matching. After removing this excess data there are further preprocessing steps we can do. One step is to remove stop words. These are words that are frequent and don't really have much meaning depending on what your end goal is. Another thing you can do is something call lemmatization. This is a process which essentially shortens words to their roots and allows words that are similar like do and doing to show up as identical words. This can make models simpler since you don't have as many different components. Finally when this is all done you can then tokenize text. This is when text is broken up into much smaller parts. These parts can be thought of as words although that's not necessarily what the tokens actually end up being. It's important to note that different tokenizers can process text differently.
    """
    )
    return


@app.cell
def _(json, pd):
    # Load speech data
    def load_json(file):
        with open(file) as f:
            data = json.load(f)
            return data

    speeches = pd.DataFrame(load_json("speeches.json"))
    return (speeches,)


@app.cell
def _(pd):
    # Load president approval data
    presidents = pd.read_csv("presidents.csv")
    presidents
    return (presidents,)


@app.cell
def _(presidents, speeches):
    #Clean president names
    pres_names = presidents['President Name'].str.replace(' ', '', regex=True)
    #Only use names from 
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
def _(mo):
    mo.md(
        r"""
    ## Topic Modeling
    Now that our data is all preprocessed we can use a transformer model to get different topics from our data. Topic modeling essentially involves taking all the words present in a sample (note that in this case we are using **all** speeches not just those relating to our group of interest because generally you need more data for good topic modeling or neural networks in general) and using a pretrained transformer (a type of machine learning model) you can group words into general topics. There are other ways to do this but using transformers gives much more human understandable topic aggregates than an algorithm like lda. As you can see below there are some clear trends, like topic 2 for examples clearly relates to money and banking.
    """
    )
    return


@app.cell
def _(BERTopic, CountVectorizer, KeyBERTInspired, speeches):
    # Create a vectorizer that will remove english stop words
    vectorizer_model = CountVectorizer(stop_words="english")
    representation_model = KeyBERTInspired(random_state=66)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, representation_model=representation_model, min_topic_size=5)
    topics, probs = topic_model.fit_transform(speeches['transcript_clean'].to_list())
    out = topic_model.visualize_barchart()
    out.update_layout(
        title = "Top 8 Topic Representative Words"
    )
    out
    return topic_model, topics


@app.cell
def _(mo):
    mo.md(r"""Now that we have our topics from our model and our data, we can extract those topics and their probablities and visualize how often those topics are a predominant topic in each president's speech. Plotly is excellent for this as it allows us to visualize almost all of these topics per president without overloading with data on first glance. Go ahead and explore the plot with your mouse! You can see what words make up different topics. Notice any trends or anythign interesting? Almost every president here uses some speech related to topic -1 which seems to revolve around government and legislation as general terminology. Isn't that interesting? Don't worry about the code too much for this one, plotting is always a bit messy.""")
    return


@app.cell
def _(px, speeches, topic_model, topics):
    speeches['topic'] = topics

    topic_counts = speeches.groupby(['president', 'topic']).size().reset_index(name='count')

    topic_pivot = topic_counts.pivot(index='president', columns='topic', values='count').fillna(0).reset_index()



    topic_long = topic_pivot.melt(id_vars='president', var_name='topic', value_name='count')

    topic_labels = {
        topic_id: ", ".join([word for word, _ in topic_model.get_topic(topic_id)])
        for topic_id in topic_model.get_topics().keys()
    }
    topic_long['topic_label'] = topic_long['topic'].map(topic_labels)
    pres_needed = [
        'Harry S. Truman',
        'Richard M. Nixon',
        'Dwight D. Eisenhower',
        'John F. Kennedy',
        'Lyndon B. Johnson',
        'Ronald Reagan',
        'Gerald Ford',
        'Jimmy Carter',
        'George H. W. Bush',
        'Bill Clinton',
        'George W. Bush',
        'Barack Obama',
        'Donald Trump',
        'Joe Biden'
    ]

    topic_long_subset = topic_long[topic_long['president'].isin(pres_needed)]

    president_terms = {
        'Harry S. Truman': (1945, 1953),
        'Dwight D. Eisenhower': (1953, 1961),
        'John F. Kennedy': (1961, 1963),
        'Lyndon B. Johnson': (1963, 1969),
        'Richard M. Nixon': (1969, 1974),
        'Gerald Ford': (1974, 1977),
        'Jimmy Carter': (1977, 1981),
        'Ronald Reagan': (1981, 1989),
        'George H. W. Bush': (1989, 1993),
        'Bill Clinton': (1993, 2001),
        'George W. Bush': (2001, 2009),
        'Barack Obama': (2009, 2017),
        'Donald Trump': (2017, 2021),
        'Joe Biden': (2021, 2025)
    }
    president_parties = {
        'Harry S. Truman': '(D)',
        'Dwight D. Eisenhower': '(R)',
        'John F. Kennedy': '(D)',
        'Lyndon B. Johnson': '(D)',
        'Richard M. Nixon': '(R)',
        'Gerald Ford': '(R)',
        'Jimmy Carter': '(D)',
        'Ronald Reagan': '(R)',
        'George H. W. Bush': '(R)',
        'Bill Clinton': '(D)',
        'George W. Bush': '(R)',
        'Barack Obama': '(D)',
        'Donald Trump': '(R)',
        'Joe Biden': '(D)'
    }

    def get_term(president):
        return f"{president_terms[president][0]}-{president_terms[president][1]}"

    topic_long_subset['term'] = topic_long_subset['president'].apply(get_term)


    term_order = [
        '(D) Harry S. Truman (1945-1953)',
        '(R) Dwight D. Eisenhower (1953-1961)',
        '(D) John F. Kennedy (1961-1963)',
        '(D) Lyndon B. Johnson (1963-1969)',
        '(R) Richard M. Nixon (1969-1974)',
        '(R) Gerald Ford (1974-1977)',
        '(D) Jimmy Carter (1977-1981)',
        '(R) Ronald Reagan (1981-1989)',
        '(R) George H. W. Bush (1989-1993)',
        '(D) Bill Clinton (1993-2001)',
        '(R) George W. Bush (2001-2009)',
        '(D) Barack Obama (2009-2017)',
        '(R) Donald Trump (2017-2021)',
        '(D) Joe Biden (2021-2025)'
    ]
    topic_long_subset['term_president_party'] = topic_long_subset['president'].map(president_parties) + ' ' + topic_long_subset['president'] + ' (' + topic_long_subset['term'] + ')'

    fig_3 = px.bar(
        topic_long_subset,
        x='term_president_party',
        y='count',
        color='topic',
        text='count',
        hover_data=['president', 'topic_label'],
        title='Topic Distribution per President(Stacked Bar Chart)'
    )

    fig_3.update_layout(
        barmode='stack',
        xaxis={'categoryorder': 'array', 'categoryarray': term_order, 'title': 'Presidential Term'},
        yaxis={'title':'Topic Distribution'},
        legend_title_text='Topic ID'
    )

    fig_3
    return topic_labels, topic_long


@app.cell
def _(mo):
    mo.md(r"""We also can look at topics and how republicans and democrats vary in their usage in their speeches. Since we're using plotly again feel free to explore with your mouse again! An example of something interesting is that republicans in the sample spoke of agriculture and labor far more than democrats. What else do you notice?""")
    return


@app.cell
def _(px, topic_labels, topic_long):
    president_parties_new = {
        'Harry S. Truman': 'Democrat',
        'Dwight D. Eisenhower': 'Republican',
        'John F. Kennedy': 'Democrat',
        'Lyndon B. Johnson': 'Democrat',
        'Richard M. Nixon': 'Republican',
        'Gerald Ford': 'Republican',
        'Jimmy Carter': 'Democrat',
        'Ronald Reagan': 'Republican',
        'George H. W. Bush': 'Republican',
        'Bill Clinton': 'Democrat',
        'George W. Bush': 'Republican',
        'Barack Obama': 'Democrat',
        'Donald Trump': 'Republican',
        'Joe Biden': 'Democrat'
    }
    topic_long['party'] = topic_long['president'].map(president_parties_new)

    def truncate_label(label, max_words=5):
        words = label.split(', ')
        return ', '.join(words[:max_words]) if len(words) > max_words else label

    party_topic_counts = topic_long.groupby(['topic', 'party'])['count'].sum().reset_index(name='total_count')


    # Merge with topic labels
    party_topic_counts['topic_label'] = party_topic_counts['topic'].map(topic_labels)
    party_topic_counts['truncated_topic_label'] = party_topic_counts['topic_label'].apply(truncate_label)
    # Create the comparative bar chart
    fig_party_comparison = px.bar(
        party_topic_counts.sort_values(by='total_count', ascending=False),
        y='truncated_topic_label',
        x='total_count',
        color='party',
        barmode='group',
        title='Topic Distribution: Democrats vs. Republicans',
        labels={'truncated_topic_label': 'Topic', 'total_count': 'Total Mentions', 'party': 'Party'}
    )

    fig_party_comparison.update_layout(xaxis={'categoryorder': 'total descending'})
    fig_party_comparison
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Sentiment and Readability Metrics
    Now that we've talked about the topics that presidents are speaking about how about what distinguishes presidents from each other? One aspect of text is something called sentiment. This is exactly what you expect, its how postive or negative the text is. There are various ways to do this but the method we've used here is Vader. It is a dictionary based method. In simplistic terms words are assigned to a specific value that is either postive or negative and then that dictionary is mapped to your text. This can be accessed through nltk which has a vader module. This is one metric but how about some others? Well fortunately this is where textstat comes in. It is a package that allows us to calculate various agreed upon readability metrics. If you want to learn more about individual metrics check out the textstat package or look up the indiviudal metrics on wikipedia.
    """
    )
    return


@app.cell
def _(pd, sent_tokenize, speeches_clean):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    all_sentences = []

    # Get the sentiment for each sentence in a document
    # Note that you could do this for the whole document but there's some interesting
    # stuff you could do like speech cadences if you have individual sentences!
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
    return (sentence_sentiment,)


@app.cell
def _(pd, speeches_clean, word_tokenize):
    import textstat

    # Our own derived lexical diversity metric
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
            'readability_consensus': textstat.text_standard(text, float_output = True)
        }

    def add_speech_metrics(row):
        text = row['transcript_clean']
        metrics = get_readability_metrics(text)
        metrics['lexical_diversity'] = lexical_diversity(text)
        return pd.Series(metrics)

    df_metrics = speeches_clean.apply(add_speech_metrics, axis=1)
    df_overall = pd.concat([speeches_clean, df_metrics], axis=1)
    return (df_overall,)


@app.cell
def _(sentence_sentiment):
    df_polarity = sentence_sentiment[['doc_id','sentiment_score']]
    df_polarity_avg = df_polarity.groupby('doc_id')['sentiment_score'].mean().reset_index()
    return (df_polarity_avg,)


@app.cell
def _(presidents):
    pres_info = presidents
    pres_info.rename(columns={'President Name': 'president'}, inplace=True)
    return (pres_info,)


@app.cell
def _(df_overall, df_polarity_avg, pres_info):
    metrics = [
        'president',
        'flesch_reading_ease',
        'flesch_kincaid_grade', 
        'smog_index', 
        'coleman_liau_index',
        'automated_readability_index', 
        'dale_chall_score', 
        'difficult_words',
        'readability_consensus', 
        'lexical_diversity'   
    ]
    df_subset = df_overall[metrics]
    df_subset = df_subset.merge(df_polarity_avg, on='doc_id', how='left')
    pres_speech_means = df_subset.groupby('president').mean().reset_index()
    pres_speech_means = pres_speech_means.merge(pres_info, on='president', how='left')
    pres_speech_means = pres_speech_means.drop(columns=['Number','Years In Office'])
    return (pres_speech_means,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## PCA
    Ok great so now we've got all these metrics, but how the heck do we visualize them or make any comparisons? Fortunately this is where Principal Component Analysis (PCA) comes in. Scikit-learn a machine learning package allows us to do this. So what is PCA? Well it's a dimensionality reduction technique. Gee that sounds confusing what does it mean? Well think of each metric we've calculated as a dimension in space. You can imagine a central sphere and that each metric is a random line coming of that sphere. Now each actual observation or number we have for each of those arrows is how far we go in that direction to create a new point. It's just like how on a regular 2d plot you have a number for X and a number for Y which are your two dimensions. So [1,1] would be 1 in the direction or dimenions of x which is left and right and 1 in the y dimension which is up and down. Now just pretend that each one of those directions actually represents one of our metrics we've calculated. If that makes sense then you're on the way to understanding high dimensions! I'm sure you're now asking though, how do you graph that? Well fortunately there  are ways to reduce a bunch of dimension into a 2d one. To really understand PCA you need to get into a little bit of linear algebra but essentially it breaks down the data into little bits that represent as much of the data as it can. We can then use the first two bits to make a plot that will actually contain information about all of the data we calculated but in a way that we can visualize. That is what the plot below is. This plot is a called a biplot! All you need to really know to interpret it is that if you look at the red lines, those represent exacctly what they are labeled as. Now imagine for example that difficult_words is a line that extends all the way up and down from where it is now. Any of the presidents that are closer to the top of the line in this plot would be higher in the amount of difficult words they use and vice versa. Now apply that to all of the other lines and you can see what the plot is telling you. If this is confusing try thinking about each line or dimension one at a time. Let's take flesch reading ease as an example. Fortunately it aligns almost exactly with the x axis so its a bit easier to understand. Those presidents to the right such as Joe Biden, Donald Trump, have extremely easy to read speeches. Those on the left like Dwight D. Eisenhower, and John F. Kennedy are much harder to read. If you want to think about this a level higher you might notice that sentiment score and reading ease are *nearly* antiparallel which means that as one increases the other decreases in an almost 1:1 ratio. So someone who makes speeches that are highly readable also tends to have less postive speeches. Interesting huh? Unfortunately our visualization is in matplotlib this time so no intereatability.
    """
    )
    return


@app.cell
def _(pd, pres_speech_means):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    labels = ['president','Party']
    drops = labels + ['doc_id']
    identifiers = pres_speech_means[labels].copy()
    X = pres_speech_means.drop(drops, axis=1)
    reduced_data = X.copy()
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=66)
    principalComponents = pca.fit_transform(X_scaled)
    loadings = pca.components_.T  # Each row corresponds to a feature

    # Build PCA scores DataFrame
    df_pca = pd.DataFrame(principalComponents, columns=['PC1', 'PC2'])
    df_pca = pd.concat([identifiers.reset_index(drop=True), df_pca], axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Scatter points (presidents)
    for party in df_pca['Party'].unique():
        if party == "Republican":
            color = ['red']
        else:
            color = ['blue']
        subset = df_pca[df_pca['Party'] == party]
        color = color * len(subset)
        ax.scatter(subset['PC1'], subset['PC2'], label=party, c=color)

    # Draw variable loadings as arrows
    for i, feature in enumerate(reduced_data):
        ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
                 color='red', alpha=0.7, head_width=0.1)
        ax.text(loadings[i, 0]*3.5, loadings[i, 1]*3.5, feature, color='red', ha='center', va='center')

    # Add text labels to each point (president)
    for _, row_num in df_pca.iterrows():
        ax.text(row_num['PC1'] + 0.1, row_num['PC2'] + 0.1, row_num['president'],
                fontsize=9, alpha=0.7, color='white')


    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_xlim(df_pca['PC1'].min() - 1, df_pca['PC1'].max() + 1)
    ax.set_ylim(df_pca['PC2'].min() - 1, df_pca['PC2'].max() + 1)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title('PCA Biplot of Presidential Speech Features')
    ax.legend()
    plt.grid(True)
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""Phew that was a lot. Let's finish up with one more plot. This is a plot of all the speeches of our presidents we are interested in in regards to the sentiment over time. This is once again done with plotly so feel free to explore. This one doesn't necessarily have clear conclusions to draw from but maybe you'll notice something interesting. Perhaps you could try to map some of these speech periods to historical periods in your mind quickly. See? Text analysis and machine learning isn't that hard! Now you hopefully have some tools to be able to do some of this on your own.""")
    return


@app.cell
def _(df_polarity_avg, speeches_clean):
    import plotly.express as px
    import plotly.colors as pc

    # Merge sentiment with speech metadata
    sentiment_timeline = speeches_clean.reset_index()[['doc_id', 'date', 'president']].merge(df_polarity_avg, on='doc_id')
    color_seq = pc.qualitative.Pastel

    fig_2 = px.line(
        sentiment_timeline,
        x='date',
        y='sentiment_score',
        labels = {
            'sentiment_score':'Speech Sentiment',
            'date':'Date'
        },
        color='president',
        color_discrete_sequence=color_seq,
        line_group='president',
        title='Presidential Speech Sentiment Over Time',
        markers=True
    )
    fig_2.update_layout(showlegend=True,legend_title_text='President')
    fig_2
    return (px,)


if __name__ == "__main__":
    app.run()
