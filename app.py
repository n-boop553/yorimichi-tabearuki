from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="買い食い散歩ルート推薦",
    page_icon="🥐",
    layout="wide",
)


# ==========================================
# ファイルパス
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ==========================================
# データ読み込み
# ==========================================

@st.cache_data
def load_data():
    places_path = DATA_DIR / "places.csv"
    roads_path = DATA_DIR / "roads.csv"

    if not places_path.exists():
        raise FileNotFoundError(
            "data/places.csvが見つかりません。"
        )

    if not roads_path.exists():
        raise FileNotFoundError(
            "data/roads.csvが見つかりません。"
        )

    places = pd.read_csv(places_path)
    roads = pd.read_csv(roads_path)

    return places, roads


try:
    places, roads = load_data()
except (FileNotFoundError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()


# ==========================================
# データ準備
# ==========================================

start_places = places.loc[
    places["role"] == "start",
    "name",
].tolist()

goal_places = places.loc[
    places["role"] == "goal",
    "name",
].tolist()

food_places = places.loc[
    places["role"] == "stop"
].copy()

food_types = sorted(
    food_places["food_type"]
    .dropna()
    .unique()
    .tolist()
)


# ==========================================
# 画面
# ==========================================

st.title("🥐 買い食い散歩ルート推薦")

st.write(
    "目的地まで歩きながら、その日の気分に合う"
    "買い食いスポットを探す推薦システムです。"
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("登録地点", f"{len(places)}か所")

with metric2:
    st.metric("買い食いスポット", f"{len(food_places)}店")

with metric3:
    st.metric("登録ルート", f"{len(roads)}本")


st.subheader("散歩の条件")

column1, column2 = st.columns(2)

with column1:
    start_name = st.selectbox(
        "出発地",
        start_places,
    )

    walk_minutes = st.slider(
        "歩ける時間",
        min_value=20,
        max_value=90,
        value=45,
        step=5,
        format="%d分",
    )

    budget = st.slider(
        "予算",
        min_value=300,
        max_value=2000,
        value=1000,
        step=100,
        format="%d円",
    )

with column2:
    goal_name = st.selectbox(
        "目的地",
        goal_places,
    )

    selected_foods = st.multiselect(
        "食べたいもの",
        food_types,
        placeholder="複数選択できます",
    )

    hunger_level = st.slider(
        "空腹度",
        min_value=1,
        max_value=5,
        value=3,
        help="1は少しだけ、5はかなり空腹です。",
    )


if st.button(
    "おすすめ候補を見る",
    type="primary",
    use_container_width=True,
):
    candidates = food_places.copy()

    if selected_foods:
        candidates["food_match"] = (
            candidates["food_type"]
            .isin(selected_foods)
            .astype(float)
        )
    else:
        candidates["food_match"] = 0.5

    candidates["budget_fit"] = (
        candidates["price"] <= budget
    ).astype(float)

    candidates["hunger_fit"] = (
        1
        - (
            candidates["fullness"] - hunger_level
        ).abs() / 5
    ).clip(lower=0)

    candidates["score"] = (
        0.50 * candidates["food_match"]
        + 0.25 * candidates["budget_fit"]
        + 0.25 * candidates["hunger_fit"]
    )

    recommendations = (
        candidates
        .sort_values(
            by="score",
            ascending=False,
        )
        .head(3)
        .copy()
    )

    recommendations["相性スコア"] = (
        recommendations["score"] * 100
    ).round(1)

    st.subheader("現在のおすすめ候補")

    st.write(
        f"**{start_name}**から**{goal_name}**へ向かう途中の"
        "候補です。"
    )

    st.dataframe(
        recommendations[
            [
                "name",
                "category",
                "food_type",
                "price",
                "texture",
                "相性スコア",
            ]
        ].rename(
            columns={
                "name": "店名",
                "category": "店の種類",
                "food_type": "食べ物",
                "price": "価格",
                "texture": "食感",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "現在は店舗の条件一致を確認する初期版です。"
        "次の実装で、道のつながりと所要時間を使って"
        "実際の寄り道ルートを作成します。"
    )
