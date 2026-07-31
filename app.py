import random
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec


st.set_page_config(
    page_title="マンガレコメンド",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 画面デザイン
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --pink: #f7a9c4;
        --pink-soft: #fff0f5;
        --lavender: #b9a7ef;
        --lavender-soft: #f1edff;
        --mint: #bfe9dd;
        --mint-soft: #effbf7;
        --purple: #7456c8;
        --text: #342f45;
        --muted: #746f83;
        --line: rgba(116, 86, 200, 0.14);
        --surface: rgba(255, 255, 255, 0.94);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            "Hiragino Sans", "Yu Gothic UI", "Yu Gothic", sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(247, 169, 196, 0.28), transparent 26rem),
            radial-gradient(circle at 92% 14%, rgba(185, 167, 239, 0.30), transparent 28rem),
            radial-gradient(circle at 72% 92%, rgba(191, 233, 221, 0.30), transparent 28rem),
            linear-gradient(135deg, #fffafd 0%, #f8f5ff 48%, #f3fcf9 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero-card {
        padding: 1.7rem 1.9rem;
        margin-bottom: 1.25rem;
        border: 1px solid var(--line);
        border-radius: 28px;
        background: linear-gradient(
            135deg,
            rgba(255, 255, 255, 0.98) 0%,
            rgba(255, 240, 245, 0.92) 48%,
            rgba(241, 237, 255, 0.94) 100%
        );
        box-shadow: 0 16px 38px rgba(100, 75, 150, 0.10);
    }

    .hero-title {
        margin: 0;
        color: var(--text);
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.2;
        letter-spacing: -0.02em;
    }

    .hero-description {
        margin: 0.7rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
    }

    .home-stats {
        margin: -0.25rem 0 1.25rem;
        color: var(--muted);
        font-size: 0.86rem;
        text-align: right;
    }

    .home-feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 0.4rem;
    }

    .home-feature-card {
        min-height: 175px;
        padding: 1.45rem;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: var(--surface);
        box-shadow: 0 12px 30px rgba(100, 75, 150, 0.09);
    }

    .home-feature-card:nth-child(1) {
        background: linear-gradient(145deg, #ffffff 0%, var(--pink-soft) 100%);
    }

    .home-feature-card:nth-child(2) {
        background: linear-gradient(145deg, #ffffff 0%, var(--lavender-soft) 100%);
    }

    .home-feature-card:nth-child(3) {
        background: linear-gradient(145deg, #ffffff 0%, var(--mint-soft) 100%);
    }

    .home-feature-title {
        margin: 0 0 0.7rem;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 800;
    }

    .home-feature-text {
        margin: 0;
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.7;
    }

    .section-label {
        margin: 1.8rem 0 0.8rem;
        color: var(--text);
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff3f8 0%, #f5f1ff 54%, #f0fbf7 100%);
        border-right: 1px solid var(--line);
    }

    div[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .sidebar-brand {
        padding: 1rem 1.05rem;
        margin-bottom: 0.8rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 10px 24px rgba(100, 75, 150, 0.08);
    }

    .sidebar-brand-title {
        margin: 0;
        color: var(--purple);
        font-size: 1.15rem;
        font-weight: 800;
    }

    .sidebar-brand-text {
        margin: 0.35rem 0 0;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        min-height: 112px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 10px 26px rgba(100, 75, 150, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--purple);
    }

    .stButton > button,
    .stDownloadButton > button,
    button[kind="primary"] {
        min-height: 3rem;
        padding: 0.65rem 1.15rem;
        border: 0 !important;
        border-radius: 999px !important;
        background: linear-gradient(135deg, #f28db4 0%, #8f72df 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 18px rgba(139, 102, 203, 0.20);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    button[kind="primary"]:hover {
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(139, 102, 203, 0.28);
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(0);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        min-height: 3rem;
        border-radius: 16px !important;
        background: rgba(255, 255, 255, 0.96) !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.35rem;
        border-radius: 18px;
        background: linear-gradient(90deg, #ffeaf2 0%, #eee9ff 100%);
    }

    button[data-baseweb="tab"] {
        min-height: 2.9rem;
        padding: 0 1rem;
        border-radius: 14px;
        font-weight: 750;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff;
        color: var(--purple);
        box-shadow: 0 5px 14px rgba(100, 75, 150, 0.12);
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 8px 22px rgba(100, 75, 150, 0.06);
    }

    div[data-testid="stAlert"] {
        border-radius: 18px;
    }

    hr {
        margin: 2rem 0;
        border-color: var(--line);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
        }

        .hero-card {
            padding: 1.25rem;
            border-radius: 22px;
        }

        .home-stats {
            text-align: left;
        }

        .home-feature-grid {
            grid-template-columns: 1fr;
        }

        .home-feature-card {
            min-height: 0;
        }

        div[data-testid="stMetric"] {
            min-height: 100px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_page_header(title, description):
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-description">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title):
    st.markdown(
        f'<div class="section-label">{title}</div>',
        unsafe_allow_html=True,
    )


# ==================================================
# 作品名の表記統一
# ==================================================


def title_key(title):
    text = unicodedata.normalize("NFKC", str(title)).strip()
    text = text.replace("×", "x")
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text.lower()


RAW_TITLE_ALIASES = {
    "ハイキュー": "ハイキュー!!",
    "ハイキュー!": "ハイキュー!!",
    "ハイキュー!!": "ハイキュー!!",
    "ハイキュー!!!": "ハイキュー!!",
    "HUNTER×HUNTER": "HUNTER×HUNTER",
    "Hunter×Hunter": "HUNTER×HUNTER",
    "HUNTER X HUNTER": "HUNTER×HUNTER",
    "ハンターハンター": "HUNTER×HUNTER",
    "ONE PIECE": "ONE PIECE",
    "ONEPIECE": "ONE PIECE",
    "ワンピース": "ONE PIECE",
    "SKET DANCE": "SKET DANCE",
    "SKETDANCE": "SKET DANCE",
    "SLAM DUNK": "SLAM DUNK",
    "SLAMDUNK": "SLAM DUNK",
    "スラムダンク": "SLAM DUNK",
    "SPY×FAMILY": "SPY×FAMILY",
    "SPY X FAMILY": "SPY×FAMILY",
    "SPYXFAMILY": "SPY×FAMILY",
    "スパイファミリー": "SPY×FAMILY",
    "DEATH NOTE": "DEATH NOTE",
    "DEATHNOTE": "DEATH NOTE",
    "デスノート": "DEATH NOTE",
    "呪術回戦": "呪術廻戦",
    "呪術廻戦": "呪術廻戦",
    "コナン": "名探偵コナン",
    "名探偵コナン": "名探偵コナン",
    "WHICHWATCH": "ウィッチウォッチ",
    "WITCH WATCH": "ウィッチウォッチ",
    "ウィッチウォッチ": "ウィッチウォッチ",
    "魔入りました入間くん": "魔入りました！入間くん",
    "魔入りました!入間くん": "魔入りました！入間くん",
    "魔入りました！入間くん": "魔入りました！入間くん",
    "僕のヒーローアカデミア": "僕のヒーローアカデミア",
    "ヒロアカ": "僕のヒーローアカデミア",
    "転生したらスライムだった件": "転生したらスライムだった件",
    "転スラ": "転生したらスライムだった件",
    "鬼滅": "鬼滅の刃",
    "鬼滅の刃": "鬼滅の刃",
}

TITLE_ALIASES = {
    title_key(original): canonical
    for original, canonical in RAW_TITLE_ALIASES.items()
}

INVALID_ANSWERS = {
    title_key(value)
    for value in [
        "",
        "なし",
        "無し",
        "特になし",
        "特にない",
        "ない",
        "ありません",
        "未回答",
        "なし。",
    ]
}


def prepare_title(value):
    if pd.isna(value):
        return None

    title = unicodedata.normalize("NFKC", str(value)).strip()
    title = re.sub(r"\s+", " ", title)
    key = title_key(title)

    if not key or key in INVALID_ANSWERS:
        return None

    return key, title


# ==================================================
# アンケートデータの読み込み
# ==================================================


@st.cache_data
def load_survey():
    try:
        df = pd.read_csv(
            "data/manga_survey.csv",
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            "data/manga_survey.csv",
            encoding="cp932",
        )

    manga_columns = [
        column
        for column in df.columns
        if "マンガ" in str(column)
    ]

    if not manga_columns:
        st.error("CSVに「マンガ」を含む列がありません。")
        st.stop()

    variant_counts = defaultdict(Counter)

    for column in manga_columns:
        for value in df[column]:
            prepared = prepare_title(value)
            if prepared is None:
                continue

            key, display_title = prepared
            if key not in TITLE_ALIASES:
                variant_counts[key][display_title] += 1

    canonical_titles = dict(TITLE_ALIASES)

    for key, counts in variant_counts.items():
        canonical_titles[key] = counts.most_common(1)[0][0]

    responses = []

    for _, row in df.iterrows():
        titles = []

        for column in manga_columns:
            prepared = prepare_title(row[column])
            if prepared is None:
                continue

            key, _ = prepared
            canonical_title = canonical_titles[key]

            if canonical_title not in titles:
                titles.append(canonical_title)

        if titles:
            responses.append(titles)

    return responses


# ==================================================
# item2vecモデル
# ==================================================


@st.cache_resource
def train_preference_model(sentences):
    training_data = [list(sentence) for sentence in sentences]

    return Word2Vec(
        sentences=training_data,
        vector_size=32,
        window=10,
        min_count=1,
        sg=1,
        negative=10,
        sample=0,
        epochs=300,
        seed=123,
        workers=1,
    )


# ==================================================
# 共通計算
# ==================================================


def create_average_vector(model, titles):
    valid_titles = [
        title
        for title in titles
        if title in model.wv.key_to_index
    ]

    if not valid_titles:
        return None

    vectors = [model.wv.get_vector(title) for title in valid_titles]
    return np.mean(vectors, axis=0)


def cosine_similarity(first_vector, second_vector):
    if first_vector is None or second_vector is None:
        return 0.0

    first_norm = np.linalg.norm(first_vector)
    second_norm = np.linalg.norm(second_vector)

    if first_norm == 0 or second_norm == 0:
        return 0.0

    similarity = float(
        np.dot(first_vector, second_vector)
        / (first_norm * second_norm)
    )

    return max(0.0, min(1.0, similarity))


def item_similarity(model, first_title, second_title):
    if first_title == second_title:
        return 1.0

    if (
        first_title not in model.wv.key_to_index
        or second_title not in model.wv.key_to_index
    ):
        return 0.0

    similarity = float(model.wv.similarity(first_title, second_title))
    return max(0.0, min(1.0, similarity))


# ==================================================
# 推薦機能
# ==================================================


def create_group_recommendations(
    responses,
    selected_title,
    excluded_titles=None,
):
    excluded_titles = set(excluded_titles or [])

    selected_responses = [
        titles
        for titles in responses
        if selected_title in titles
    ]

    together_counts = Counter()

    for titles in selected_responses:
        for title in titles:
            if title != selected_title and title not in excluded_titles:
                together_counts[title] += 1

    results = []

    for title, together_count in together_counts.items():
        title_user_count = sum(
            title in titles
            for titles in responses
        )

        union_count = (
            len(selected_responses)
            + title_user_count
            - together_count
        )

        recommendation_score = (
            together_count / union_count
            if union_count > 0
            else 0
        )

        results.append(
            {
                "マンガ": title,
                "関連回答数": together_count,
                "相性スコア": recommendation_score,
                "おすすめ理由": (
                    f"同じ回答データで{together_count}件選択"
                ),
            }
        )

    results.sort(
        key=lambda item: (
            item["相性スコア"],
            item["関連回答数"],
        ),
        reverse=True,
    )

    return pd.DataFrame(results)


def create_preference_recommendations(
    model,
    responses,
    selected_title,
    result_count,
    excluded_titles=None,
):
    excluded_titles = set(excluded_titles or [])

    if selected_title not in model.wv.key_to_index:
        return pd.DataFrame()

    vocabulary_size = len(model.wv.index_to_key)
    if vocabulary_size <= 1:
        return pd.DataFrame()

    candidate_count = min(
        max(result_count * 5, 30),
        vocabulary_size - 1,
    )

    similar_titles = model.wv.most_similar(
        positive=[selected_title],
        topn=candidate_count,
    )

    results = []

    for title, similarity in similar_titles:
        if title in excluded_titles:
            continue

        together_count = sum(
            selected_title in titles and title in titles
            for titles in responses
        )

        similarity = max(0.0, min(1.0, float(similarity)))

        results.append(
            {
                "マンガ": title,
                "好みの近さ": similarity,
                "関連回答数": together_count,
                "おすすめ理由": (
                    f"好みの近さ {similarity:.2f}"
                ),
            }
        )

        if len(results) >= result_count:
            break

    return pd.DataFrame(results)


def create_multi_recommendations(
    model,
    responses,
    selected_titles,
    result_count,
    excluded_titles=None,
):
    excluded_titles = set(excluded_titles or [])
    selected_set = set(selected_titles)
    selected_vector = create_average_vector(model, selected_titles)

    if selected_vector is None:
        return pd.DataFrame()

    vocabulary_size = len(model.wv.index_to_key)
    candidate_count = min(
        max(result_count * 8, 50),
        vocabulary_size,
    )

    similar_titles = model.wv.similar_by_vector(
        selected_vector,
        topn=candidate_count,
    )

    results = []

    for title, similarity in similar_titles:
        if title in selected_set or title in excluded_titles:
            continue

        selected_cooccurrence = sum(
            title in response
            and any(selected in response for selected in selected_set)
            for response in responses
        )

        similarity = max(0.0, min(1.0, float(similarity)))

        support_score = min(
            1.0,
            selected_cooccurrence / 3,
        )

        combined_score = (
            0.80 * similarity
            + 0.20 * support_score
        )

        results.append(
            {
                "マンガ": title,
                "相性スコア": combined_score,
                "好みの近さ": similarity,
                "関連回答数": selected_cooccurrence,
                "おすすめ理由": (
                    "複数作品の平均的な好みに近い"
                    if selected_cooccurrence == 0
                    else (
                        f"好みが近く、関連回答が"
                        f"{selected_cooccurrence}件"
                    )
                ),
            }
        )

        if len(results) >= result_count:
            break

    results.sort(
        key=lambda item: item["相性スコア"],
        reverse=True,
    )

    return pd.DataFrame(results)


# ==================================================
# 異端度・嗜好診断
# ==================================================


DIRECT_MATCH_WEIGHT = 0.60
ITEM2VEC_WEIGHT = 0.40
REFERENCE_NEIGHBORS = 3
CLOSE_SIMILARITY_THRESHOLD = 0.70


def analyze_rarity(
    responses,
    selected_titles,
    model,
):
    selected_set = set(selected_titles)
    selected_count = len(selected_set)

    if selected_count < 2:
        return None

    selected_vector = create_average_vector(
        model,
        selected_titles,
    )

    comparison_results = []

    for index, response_titles in enumerate(responses):
        response_set = set(response_titles)
        matched_titles = selected_set & response_set
        exact_match_count = len(matched_titles)
        direct_match_ratio = exact_match_count / selected_count

        response_vector = create_average_vector(
            model,
            response_titles,
        )

        vector_similarity = cosine_similarity(
            selected_vector,
            response_vector,
        )

        combined_similarity = (
            DIRECT_MATCH_WEIGHT * direct_match_ratio
            + ITEM2VEC_WEIGHT * vector_similarity
        )

        comparison_results.append(
            {
                "回答番号": index + 1,
                "一致数": exact_match_count,
                "直接一致率": direct_match_ratio,
                "作品構成の近さ": vector_similarity,
                "総合的な近さ": combined_similarity,
                "一致した作品": " / ".join(sorted(matched_titles)),
                "回答作品": " / ".join(response_titles),
                "選んだ作品をすべて含む": (
                    selected_set.issubset(response_set)
                ),
            }
        )

    comparison_results.sort(
        key=lambda result: result["総合的な近さ"],
        reverse=True,
    )

    neighbor_count = min(
        REFERENCE_NEIGHBORS,
        len(comparison_results),
    )

    closest_results = comparison_results[:neighbor_count]

    neighborhood_similarity = sum(
        result["総合的な近さ"]
        for result in closest_results
    ) / neighbor_count

    rarity_score = round(
        100 * (1 - neighborhood_similarity)
    )
    rarity_score = max(0, min(100, rarity_score))

    complete_match_count = sum(
        result["選んだ作品をすべて含む"]
        for result in comparison_results
    )

    close_match_count = sum(
        result["総合的な近さ"]
        >= CLOSE_SIMILARITY_THRESHOLD
        for result in comparison_results
    )

    return {
        "selected_count": selected_count,
        "rarity_score": rarity_score,
        "best_similarity": comparison_results[0]["総合的な近さ"],
        "neighborhood_similarity": neighborhood_similarity,
        "close_match_count": close_match_count,
        "complete_match_count": complete_match_count,
        "closest_results": comparison_results[:10],
    }


def rarity_message(score):
    if score <= 15:
        return "かなり王道。近い好みの人が見つかりました。"
    if score <= 30:
        return "やや王道。似た好みの人がいます。"
    if score <= 50:
        return "ほどよく個性的な組み合わせです。"
    if score <= 70:
        return "かなり個性的な組み合わせです。"
    return "かなりレアなマンガ嗜好です。"


def pairwise_cohesion(model, titles):
    similarities = []

    for first_index in range(len(titles)):
        for second_index in range(first_index + 1, len(titles)):
            similarities.append(
                item_similarity(
                    model,
                    titles[first_index],
                    titles[second_index],
                )
            )

    if not similarities:
        return 0.0

    return sum(similarities) / len(similarities)


def diagnose_taste(
    selected_titles,
    popularity,
    model,
):
    maximum_popularity = max(popularity.values())

    popularity_level = sum(
        popularity[title] / maximum_popularity
        for title in selected_titles
    ) / len(selected_titles)

    cohesion = pairwise_cohesion(
        model,
        selected_titles,
    )

    if popularity_level >= 0.45 and cohesion >= 0.45:
        label = "王道集中型"
        message = (
            "人気作を中心に、近い系統を選ぶ好みです。"
        )
    elif popularity_level >= 0.45:
        label = "王道横断型"
        message = (
            "人気作を押さえつつ、幅広い系統を楽しむ好みです。"
        )
    elif cohesion >= 0.45:
        label = "発掘集中型"
        message = (
            "知る人ぞ知る作品を、近い系統で深掘りする好みです。"
        )
    else:
        label = "独自横断型"
        message = (
            "レアな作品を、系統をまたいで楽しむ好みです。"
        )

    return {
        "label": label,
        "message": message,
        "popularity_level": popularity_level,
        "cohesion": cohesion,
    }


# ==================================================
# セッション内リスト
# ==================================================


def initialize_session_state():
    defaults = {
        "read_later": [],
        "not_interested": [],
        "gacha_title": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_unique_to_state(key, title):
    current = list(st.session_state[key])
    if title not in current:
        current.append(title)
        st.session_state[key] = current


def remove_from_state(key, title):
    st.session_state[key] = [
        item
        for item in st.session_state[key]
        if item != title
    ]


def render_title_actions(title, key_prefix):
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🔖 あとで読む",
            key=f"{key_prefix}_read_later",
            use_container_width=True,
        ):
            add_unique_to_state("read_later", title)
            remove_from_state("not_interested", title)
            st.success("「あとで読む」に追加しました。")

    with col2:
        if st.button(
            "🙅 興味なし",
            key=f"{key_prefix}_not_interested",
            use_container_width=True,
        ):
            add_unique_to_state("not_interested", title)
            remove_from_state("read_later", title)
            st.success("おすすめ候補から外しました。")

    with col3:
        if st.button(
            "解除",
            key=f"{key_prefix}_clear",
            use_container_width=True,
        ):
            remove_from_state("read_later", title)
            remove_from_state("not_interested", title)
            st.success("登録を外しました。")


def render_recommendation_actions(
    candidate_titles,
    key_prefix,
):
    if not candidate_titles:
        return

    st.markdown("#### 気になる作品を保存")

    action_title = st.selectbox(
        "保存する作品",
        candidate_titles,
        key=f"{key_prefix}_action_title",
    )

    render_title_actions(
        action_title,
        key_prefix,
    )


# ==================================================
# データ準備
# ==================================================


responses = load_survey()
training_sentences = tuple(
    tuple(titles)
    for titles in responses
)
preference_model = train_preference_model(
    training_sentences
)

all_titles = sorted(
    {
        title
        for titles in responses
        for title in titles
    },
    key=lambda title: title.lower(),
)

popularity = Counter(
    title
    for titles in responses
    for title in titles
)

popularity_rank = {
    title: rank
    for rank, (title, _) in enumerate(
        popularity.most_common(),
        start=1,
    )
}

initialize_session_state()
excluded_titles = set(st.session_state.not_interested)


# ==================================================
# サイドバー
# ==================================================


st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <p class="sidebar-brand-title">📚 マンガレコメンド</p>
        <p class="sidebar-brand-text">
            好きなマンガから、次に読む1冊を探せます。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "メニュー",
    [
        "🏠 ホーム",
        "📖 1冊から探す",
        "📚 2〜5冊から探す",
        "🔎 作品検索",
        "⚖️ 作品相性チェック",
        "🪐 マンガ嗜好診断",
        "🎲 マンガガチャ",
        "🏆 ランキング",
        "🔖 あとで読む",
    ],
)

st.sidebar.caption(
    f"回答データ：{len(responses)}人 ／ "
    f"登録作品：{len(all_titles)}冊"
)

if st.session_state.read_later:
    with st.sidebar.expander(
        f"🔖 あとで読む（{len(st.session_state.read_later)}）"
    ):
        for title in st.session_state.read_later:
            st.write(f"・{title}")

if st.session_state.not_interested:
    with st.sidebar.expander(
        f"🙅 興味なし（{len(st.session_state.not_interested)}）"
    ):
        for title in st.session_state.not_interested:
            st.write(f"・{title}")


# ==================================================
# ホーム
# ==================================================


if page == "🏠 ホーム":
    render_page_header(
        "📚 マンガレコメンド",
        "好きなマンガから、次に読む作品をサクッと探せます。",
    )

    st.markdown(
        f'<div class="home-stats">回答データ：{len(responses)}人 ／ '
        f'登録作品：{len(all_titles)}冊</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-feature-grid">
            <div class="home-feature-card">
                <p class="home-feature-title">📖 1冊から探す</p>
                <p class="home-feature-text">
                    好きな作品を1冊選んで、相性のよいマンガをチェック。
                </p>
            </div>
            <div class="home-feature-card">
                <p class="home-feature-title">📚 2〜5冊から探す</p>
                <p class="home-feature-text">
                    複数作品から、好み全体に合うマンガを提案。
                </p>
            </div>
            <div class="home-feature-card">
                <p class="home-feature-title">🪐 マンガ嗜好診断</p>
                <p class="home-feature-text">
                    嗜好タイプや異端度を気軽にチェック。
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# 1冊から探す
# ==================================================


elif page == "📖 1冊から探す":
    render_page_header(
        "📖 1冊から探す",
        "好きな作品を1冊選んで、相性のよいマンガを探します。",
    )

    selected_title = st.selectbox(
        "好きなマンガを選択",
        all_titles,
    )

    result_count = int(
        st.number_input(
            "おすすめ件数",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
        )
    )

    selected_user_count = sum(
        selected_title in titles
        for titles in responses
    )

    st.caption(
        f"「{selected_title}」の回答数："
        f"{selected_user_count}件"
    )

    if selected_user_count == 1:
        st.warning(
            "回答数が1件のため、おすすめは参考として見てください。"
        )

    tab1, tab2 = st.tabs(
        [
            "👥 一緒に選ばれた作品",
            "🧭 好みの近さ",
        ]
    )

    with tab1:
        group_results = create_group_recommendations(
            responses,
            selected_title,
            excluded_titles,
        )

        if group_results.empty:
            st.warning("一緒に選ばれた作品はありません。")
        else:
            display_df = group_results.head(result_count).copy()
            display_df.insert(
                0,
                "順位",
                range(1, len(display_df) + 1),
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "相性スコア": st.column_config.ProgressColumn(
                        "相性スコア",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                },
            )

            render_recommendation_actions(
                display_df["マンガ"].tolist(),
                "single_group",
            )

    with tab2:
        preference_results = create_preference_recommendations(
            preference_model,
            responses,
            selected_title,
            result_count,
            excluded_titles,
        )

        if preference_results.empty:
            st.warning("おすすめを作れませんでした。")
        else:
            preference_results.insert(
                0,
                "順位",
                range(1, len(preference_results) + 1),
            )

            st.dataframe(
                preference_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

            render_recommendation_actions(
                preference_results["マンガ"].tolist(),
                "single_vector",
            )


# ==================================================
# 2〜5冊から探す
# ==================================================


elif page == "📚 2〜5冊から探す":
    render_page_header(
        "📚 2〜5冊から探す",
        "好きなマンガを2〜5冊選んで、好み全体に合う作品を探します。",
    )

    selected_titles = st.multiselect(
        "好きなマンガを2〜5冊選択",
        all_titles,
        max_selections=5,
        placeholder="作品名を検索",
    )

    result_count = int(
        st.number_input(
            "おすすめ件数",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="multi_result_count",
        )
    )

    if len(selected_titles) < 2:
        st.info(
            f"あと{2 - len(selected_titles)}冊選ぶと診断できます。"
        )
    else:
        results = create_multi_recommendations(
            preference_model,
            responses,
            selected_titles,
            result_count,
            excluded_titles,
        )

        if results.empty:
            st.warning("おすすめを作れませんでした。")
        else:
            results.insert(
                0,
                "順位",
                range(1, len(results) + 1),
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "相性スコア": st.column_config.ProgressColumn(
                        "相性スコア",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

            st.caption(
                "好みの近さ80％と、同じ回答データで選ばれた実績20％から相性スコアを計算しています。"
            )

            render_recommendation_actions(
                results["マンガ"].tolist(),
                "multi",
            )


# ==================================================
# 作品検索
# ==================================================


elif page == "🔎 作品検索":
    render_page_header(
        "🔎 作品検索",
        "作品名から、回答数や好みの近いマンガを探せます。",
    )

    search_text = st.text_input(
        "作品名で検索",
        placeholder="タイトルの一部を入力",
    )

    if search_text:
        matching_titles = [
            title
            for title in all_titles
            if title_key(search_text) in title_key(title)
        ]
    else:
        matching_titles = all_titles

    if not matching_titles:
        st.warning("作品が見つかりませんでした。")
    else:
        detail_title = st.selectbox(
            "作品を選択",
            matching_titles,
        )

        title_count = popularity[detail_title]
        title_rank = popularity_rank[detail_title]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("回答数", f"{title_count}件")

        with col2:
            st.metric("人気順位", f"{title_rank}位")

        with col3:
            status = (
                "あとで読む"
                if detail_title in st.session_state.read_later
                else (
                    "興味なし"
                    if detail_title in st.session_state.not_interested
                    else "未登録"
                )
            )
            st.metric("登録状態", status)

        related = create_preference_recommendations(
            preference_model,
            responses,
            detail_title,
            5,
            excluded_titles={detail_title},
        )

        render_section_title("この作品と好みが近いマンガ")

        if related.empty:
            st.info("近いマンガを見つけられませんでした。")
        else:
            st.dataframe(
                related,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

        render_title_actions(detail_title, "detail")

        st.caption(
            "作者・ジャンル・あらすじは元データに含まれていません。"
        )


# ==================================================
# 作品相性チェック
# ==================================================


elif page == "⚖️ 作品相性チェック":
    render_page_header(
        "⚖️ 作品相性チェック",
        "2冊の相性を、回答データの選ばれ方からチェックします。",
    )

    col1, col2 = st.columns(2)

    with col1:
        first_title = st.selectbox(
            "作品A",
            all_titles,
            key="compare_first",
        )

    with col2:
        second_candidates = [
            title
            for title in all_titles
            if title != first_title
        ]
        second_title = st.selectbox(
            "作品B",
            second_candidates,
            key="compare_second",
        )

    similarity = item_similarity(
        preference_model,
        first_title,
        second_title,
    )

    together_count = sum(
        first_title in titles and second_title in titles
        for titles in responses
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("相性", f"{similarity * 100:.1f}%")

    with col2:
        st.metric("両方を選んだ回答", f"{together_count}件")

    with col3:
        st.metric(
            "各作品の回答数",
            f"{popularity[first_title]}件 / {popularity[second_title]}件",
        )

    st.progress(similarity)

    if similarity >= 0.70:
        st.success("かなり相性のよい組み合わせです。")
    elif similarity >= 0.40:
        st.info("少し共通点のある組み合わせです。")
    else:
        st.warning("好みの方向が離れた組み合わせです。")

    st.caption(
        "相性は、回答データ内で一緒に選ばれた傾向から計算しています。"
    )


# ==================================================
# マンガ嗜好診断
# ==================================================


elif page == "🪐 マンガ嗜好診断":
    render_page_header(
        "🪐 マンガ嗜好診断",
        "好きなマンガから、嗜好タイプと異端度をチェックします。",
    )

    selected_favorites = st.multiselect(
        "好きなマンガを2〜5冊選択",
        all_titles,
        max_selections=5,
        placeholder="作品名を検索",
    )

    if len(selected_favorites) < 2:
        st.info(
            f"あと{2 - len(selected_favorites)}冊選ぶと診断できます。"
        )
    else:
        diagnosis = diagnose_taste(
            selected_favorites,
            popularity,
            preference_model,
        )

        rarity_result = analyze_rarity(
            responses,
            selected_favorites,
            preference_model,
        )

        render_section_title("嗜好タイプ")
        st.success(
            f"### {diagnosis['label']}\n\n"
            f"{diagnosis['message']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "人気度",
                f"{diagnosis['popularity_level'] * 100:.0f}%",
            )

        with col2:
            st.metric(
                "好みのまとまり",
                f"{diagnosis['cohesion'] * 100:.0f}%",
            )

        render_section_title("異端度")

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "異端度",
                f"{rarity_result['rarity_score']} / 100",
            )

        with metric2:
            st.metric(
                "いちばん近い人",
                f"{rarity_result['best_similarity'] * 100:.0f}%",
            )

        with metric3:
            st.metric(
                "近い回答",
                f"{rarity_result['close_match_count']}件",
            )

        with metric4:
            st.metric(
                "全作品一致",
                f"{rarity_result['complete_match_count']}件",
            )

        st.progress(rarity_result["rarity_score"])
        st.write(rarity_message(rarity_result["rarity_score"]))

        st.caption(
            "異端度は、作品名の直接一致を60％、"
            "item2vecによる作品構成の近さを40％として"
            "回答者ごとの近さを計算し、"
            "近い上位3人の平均から算出しています。"
        )

        with st.expander("近い回答者ランキング"):
            closest_df = pd.DataFrame(
                rarity_result["closest_results"]
            )
            closest_df.insert(
                0,
                "順位",
                range(1, len(closest_df) + 1),
            )

            st.dataframe(
                closest_df[
                    [
                        "順位",
                        "一致数",
                        "直接一致率",
                        "作品構成の近さ",
                        "総合的な近さ",
                        "一致した作品",
                        "回答作品",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "直接一致率": st.column_config.ProgressColumn(
                        "直接一致率",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "作品構成の近さ": st.column_config.ProgressColumn(
                        "作品構成の近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "総合的な近さ": st.column_config.ProgressColumn(
                        "総合的な近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                },
            )


# ==================================================
# マンガガチャ
# ==================================================


elif page == "🎲 マンガガチャ":
    render_page_header(
        "🎲 マンガガチャ",
        "ランダムな1冊や、まだ知られていない作品に出会えます。",
    )

    available_titles = [
        title
        for title in all_titles
        if title not in excluded_titles
    ]

    minor_titles = [
        title
        for title in available_titles
        if popularity[title] <= 2
    ]

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🎲 全作品ガチャ",
            use_container_width=True,
        ):
            st.session_state.gacha_title = random.choice(
                available_titles
            )

    with col2:
        if st.button(
            "💎 レア作品ガチャ",
            use_container_width=True,
        ):
            candidate_pool = minor_titles or available_titles
            st.session_state.gacha_title = random.choice(
                candidate_pool
            )

    if st.session_state.gacha_title:
        gacha_title = st.session_state.gacha_title

        st.success(
            f"## {gacha_title}\n\n"
            f"回答数：{popularity[gacha_title]}件\n\n"
            f"人気順位：{popularity_rank[gacha_title]}位"
        )

        related = create_preference_recommendations(
            preference_model,
            responses,
            gacha_title,
            3,
            excluded_titles={gacha_title},
        )

        if not related.empty:
            st.write("**この作品と好みが近いマンガ**")
            st.dataframe(
                related[["マンガ", "好みの近さ"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

        render_title_actions(gacha_title, "gacha")

    render_section_title("レア作品発掘")

    minor_df = pd.DataFrame(
        sorted(
            [
                {
                    "マンガ": title,
                    "回答数": popularity[title],
                    "人気順位": popularity_rank[title],
                }
                for title in minor_titles
            ],
            key=lambda item: (
                item["回答数"],
                item["マンガ"],
            ),
        )
    )

    if minor_df.empty:
        st.info("回答数2件以下の作品はありません。")
    else:
        st.dataframe(
            minor_df,
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# ランキング
# ==================================================


elif page == "🏆 ランキング":
    render_page_header(
        "🏆 ランキング",
        "人気作とレア作品を、回答数順にチェックできます。",
    )

    ranking_size = int(
        st.number_input(
            "表示件数",
            min_value=5,
            max_value=min(50, len(all_titles)),
            value=min(20, len(all_titles)),
            step=1,
        )
    )

    popular_df = pd.DataFrame(
        popularity.most_common(ranking_size),
        columns=["マンガ", "回答数"],
    )
    popular_df.insert(
        0,
        "順位",
        range(1, len(popular_df) + 1),
    )

    render_section_title("人気ランキング")
    st.dataframe(
        popular_df,
        use_container_width=True,
        hide_index=True,
    )

    render_section_title("レア作品ランキング")
    st.write(
        "回答数が少ない順に並べています。"
    )

    discovery_df = pd.DataFrame(
        sorted(
            [
                {
                    "マンガ": title,
                    "回答数": popularity[title],
                    "人気順位": popularity_rank[title],
                }
                for title in all_titles
            ],
            key=lambda item: (
                item["回答数"],
                item["マンガ"],
            ),
        )[:ranking_size]
    )

    st.dataframe(
        discovery_df,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# あとで読む
# ==================================================


elif page == "🔖 あとで読む":
    render_page_header(
        "🔖 あとで読む",
        "気になるマンガと、興味なしにした作品を整理できます。",
    )
    st.caption(
        "保存内容は、このブラウザを開いている間だけ残ります。"
    )

    if not st.session_state.read_later:
        st.info("まだ登録はありません。")
    else:
        list_df = pd.DataFrame(
            [
                {
                    "マンガ": title,
                    "回答数": popularity[title],
                    "人気順位": popularity_rank[title],
                }
                for title in st.session_state.read_later
            ]
        )

        st.dataframe(
            list_df,
            use_container_width=True,
            hide_index=True,
        )

        remove_title = st.selectbox(
            "リストから削除する作品",
            st.session_state.read_later,
        )

        if st.button("選択した作品を削除"):
            remove_from_state("read_later", remove_title)
            st.rerun()

    render_section_title("興味なしリスト")

    if not st.session_state.not_interested:
        st.info("興味なしの登録はありません。")
    else:
        st.write(
            " / ".join(st.session_state.not_interested)
        )

        restore_title = st.selectbox(
            "興味なしを解除する作品",
            st.session_state.not_interested,
        )

        if st.button("興味なしを解除"):
            remove_from_state("not_interested", restore_title)
            st.rerun()
