import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def generate_c_tf_idf(docs_per_topic, data_len, stop_words="english", ngram_range=(1, 1)):
    count = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range).fit(
        docs_per_topic["text"].values)
    t = count.transform(docs_per_topic["text"].values).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(data_len, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count


def extract_top_n_words_per_topic(c_tf_idf, grouped_docs, count, n=20):
    words = count.get_feature_names_out()
    labels = sorted(list(grouped_docs["cluster"]))
    tf_idf_transposed = c_tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)
    }
    return top_n_words


async def get_top_k_words(raw_data, cluster_labels, n=10):
    clustered_embeds = pd.DataFrame({
        "cluster": cluster_labels,
        "text": raw_data,
    }).reset_index()

    grouped_docs = clustered_embeds.groupby(
        ['cluster'], as_index=False).agg({'text': ' '.join})

    c_tf_idf, count = generate_c_tf_idf(
        grouped_docs, len(clustered_embeds))

    return extract_top_n_words_per_topic(c_tf_idf, grouped_docs, count, n)


async def find_popular_word(raw_text, cluster_labels):
    vectorizer = CountVectorizer(max_features=32, stop_words="english")
    vectorizer.fit(raw_text)

    feature_names = vectorizer.get_feature_names_out()
    feature_matrix = pd.DataFrame(
        vectorizer.transform(raw_text).toarray(),
        columns=vectorizer.get_feature_names_out()
    ).T

    popular_words = {}
    for feature in feature_names:
        feature_doc_topics = cluster_labels[feature_matrix.loc[feature].to_numpy().nonzero()]
        unique, counts = np.unique(feature_doc_topics, return_counts=True)
        unique_counts = sorted(list(zip(unique.tolist(), counts.tolist())),
                               key=lambda x: x[1], reverse=True)[:10]

        popular_words[feature] = {
            "topic_counts": dict(unique_counts),
            "n_docs": len(feature_doc_topics)
        }

    return popular_words


def pretty_print_top_k_words(raw_data, cluster_labels, n_words=10, n_cols=3):
    top_words = get_top_k_words(raw_data, cluster_labels, n_words)
    c_ids = list(sorted(top_words.keys()))

    if -1 in top_words.keys():
        c_ids = c_ids[1:]
        c_ids.append(-1)

    for start in range(0, len(c_ids), n_cols):
        current_ids = c_ids[start:(start + n_cols)]

        print("-" * (33 * len(current_ids) + n_cols))
        for i, c_id in enumerate(current_ids):
            num_docs = len(cluster_labels[cluster_labels == c_id])
            percent = round((num_docs / len(cluster_labels)) * 100, 2)
            idx = "|  " if i == 0 else ""
            end = "\n" if i == len(current_ids) - 1 else ""
            msg = f"Topic #{c_id}" if c_id != -1 else "Outliers (-1)"
            msg += f": {num_docs} Docs ({percent}%)"
            print(idx, msg.ljust(29), "| ", end=end)
        print("-" * (33 * len(current_ids) + 2))

        for i, word_scores in enumerate(zip(*[top_words[i] for i in current_ids])):
            for j, (word, score) in enumerate(word_scores):
                end = " " if j < len(word_scores) else "\n"
                idx = str(i + 1).ljust(3) if j == 0 else ""
                print(idx, word.ljust(20), str(
                    score.round(5)).ljust(8), "|", end=end)
            print()
        print()


def plot_top_k_words(raw_data, cluster_labels, n_words=10, n_cols=3):
    top_words = get_top_k_words(raw_data, cluster_labels, n_words)
    df = {"word": [], "score": [], "topic": []}

    for topic, words in top_words.items():
        for word, score in words:
            df["word"].append(word)
            df["score"].append(score)
            df["topic"].append(topic)

    df = pd.DataFrame(df)

    base = alt.Chart(df).mark_bar().encode(
        x=alt.X('word:N', axis=alt.Axis(labelAngle=-45)),
        y='score:Q',
        color=alt.Color('topic:N').scale(
            scheme="category10" if len(cluster_labels) <= 10 else "category20"
        )
    ).properties(
        width=120,
        height=160,
    )

    # List to hold rows of charts
    rows = []

    # Temporary list to hold charts for each row
    temp_row = []

    for i, topic in enumerate(df["topic"].unique()):
        # Add a chart to the temporary row
        temp_row.append(base.transform_filter(alt.datum.topic == topic).properties(
            title=f"Topic {topic}"
        ))

        # When we have `n_cols` charts, or we're at the last chart, add the row to the rows list
        if (i + 1) % n_cols == 0 or i == len(df["topic"].unique()) - 1:
            rows.append(alt.hconcat(*temp_row))
            temp_row = []

    return alt.vconcat(*rows).properties(
        title=f"Top {n_words} Words in Each Topic"
    )


def plot_embedding_space(data, embeds, cluster_labels, width=500, height=500, summary_col="title"):
    x = embeds[:, 0]
    y = embeds[:, 1]

    chart_data = pd.DataFrame({
        "x": x,
        "y": y,
        "topic": cluster_labels,
        "title": data[summary_col],
        # "date": data["date"]
    }).reset_index()

    topic_selection = alt.selection_point(fields=["topic"])

    chart = alt.Chart(chart_data).mark_point(
        filled=True,
        size=112,
        stroke="#FFF",
        strokeWidth=1,
    ).encode(
        x=alt.Y('x').scale(zero=False),
        y=alt.Y('y').scale(zero=False),
        color=alt.condition(
            topic_selection,
            alt.Color('topic:N'),
            alt.value('lightgray'),
        ),
        tooltip=["title", "index",
                 #  "date",
                 "topic"]
    ).add_params(
        topic_selection
    ).properties(
        width=width,
        height=height,
        title="Low Dimensional Embedding Space"
    ).interactive()

    return chart, topic_selection, chart_data


def plot_img_embedding_space(images, embeds, cluster_labels, width=500, height=500):
    x = embeds[:, 0]
    y = embeds[:, 1]

    chart_data = pd.DataFrame({
        "x": x,
        "y": y,
        "topic": cluster_labels,
        "image": images,
    }).reset_index()

    topic_selection = alt.selection_point(fields=["topic"])

    chart = alt.Chart(chart_data).mark_point(
        filled=True,
        size=112,
        stroke="#FFF",
        strokeWidth=1,
    ).encode(
        x=alt.Y('x').scale(zero=False),
        y=alt.Y('y').scale(zero=False),
        color=alt.condition(
            topic_selection,
            alt.Color('topic:N'),
            alt.value('lightgray'),
        ),
        tooltip=["index", "image", "topic"]
    ).add_params(
        topic_selection
    ).properties(
        width=width,
        height=height,
        title="Low Dimensional Embedding Space"
    ).interactive()

    return chart, topic_selection, chart_data


def plot_highlighted_nodes(scatter_plot: alt.Chart, landmark):
    scatter_plot = scatter_plot.encode(
        opacity=alt.condition(
            alt.datum.highlight,
            alt.value(1),
            alt.value(0.05)
        ),
        shape=alt.condition(
            alt.datum.index == landmark,
            alt.value(
                "M0,.5L.6,.8L.5,.1L1,-.3L.3,-.4L0,-1L-.3,-.4L-1,-.3L-.5,.1L-.6,.8L0,.5Z"),
            alt.value("circle")
        ),
        stroke=alt.condition(
            alt.datum.index == landmark,
            alt.value("#000"),
            alt.value("#fff")
        )
    )

    return scatter_plot


def plot_topic_count(cluster_labels, doc_indices, selected_topic):
    topics = cluster_labels if doc_indices == True else cluster_labels[doc_indices]

    bar_chart = alt.Chart(
        pd.DataFrame(
            {"topic": topics}
        ).value_counts().reset_index()
    ).mark_bar().encode(
        x='topic:N',
        y="count",
        color=alt.condition(
            selected_topic,
            alt.Color('topic:N').scale(
                scheme="category10"
            ),
            alt.value('lightgray')
        )
    ).add_params(
        selected_topic
    )

    return bar_chart


def find_topk_tfidf(df, k=10):
    nlp = spacy.load("en_core_web_sm")

    narrative_df = df.apply(lambda x: " ".join(
        [token.lemma_ for token in nlp(x.lower()) if not token.is_stop]
    ))

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(narrative_df)

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=narrative_df.index
    )

    word_relevance = tfidf_df.sum(axis=0).sort_values(ascending=False)
    top_k_words = word_relevance.head(k)
    top_k_tfidf = tfidf_df[top_k_words.index]

    adj = np.zeros((k, k))
    for word_idx, _ in enumerate(top_k_words.index):
        for _, doc in top_k_tfidf.iterrows():
            if doc.iloc[word_idx] != 0:
                d = np.array(doc.tolist())
                d_val = d[word_idx]
                d[word_idx] = 0
                adj[word_idx] += (1 - d) * (1 - d_val)

    return top_k_tfidf, top_k_words, adj


def rank_topk_word(topk_words, adj):
    # Initialize an empty undirected graph
    G = nx.Graph()

    # Add nodes with the word as the node ID
    for word in topk_words.index:
        G.add_node(word)

    # Add edges based on the adjacency matrix
    for i, word1 in enumerate(topk_words.index):
        for j, word2 in enumerate(topk_words.index):
            # Check if there's a connection (and avoid adding an edge with 0 weight or self-loops)
            if i != j and adj[i, j] != 0:
                G.add_edge(word1, word2, weight=adj[i, j])

    return nx.eigenvector_centrality_numpy(G, weight='weight')


def plot_topk_tfidf(df, k=10, width_multiplier=1):
    top_k_tfidf, top_k_words, adj = find_topk_tfidf(df, k)
    node_ranks = rank_topk_word(top_k_words, adj)

    sorted_words_by_eigencentrality = sorted(
        node_ranks.keys(),
        key=lambda word: node_ranks[word],
        reverse=True
    )

    sorted_top_k_tfidf = top_k_tfidf[sorted_words_by_eigencentrality]
    sorted_top_k_tfidf_transposed = sorted_top_k_tfidf.T
    n_rows, n_cols = sorted_top_k_tfidf_transposed.shape
    base_cell_size = 0.5

    fig_width = n_cols * base_cell_size * width_multiplier  # Total figure width
    fig_height = n_rows * base_cell_size - 1  # Total figure height

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        sorted_top_k_tfidf_transposed,
        cmap="viridis",
        annot=False,
        fmt=".2f",
        cbar_kws={'label': 'TF-IDF Score'},
        annot_kws={"size": 5}
    )
    # plt.title('TF-IDF Scores of Top k Words')
    plt.xlabel('Document Index')
    plt.ylabel('Words')
    plt.xticks(rotation=45)
    plt.show()
