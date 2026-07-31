from itertools import permutations
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="買い食い散歩ルート推薦",
    page_icon="🥐",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MOOD_WEIGHTS = {
    "バランス重視": {
        "food": 0.30,
        "time": 0.20,
        "budget": 0.15,
        "hunger": 0.20,
        "variety": 0.10,
        "novelty": 0.05,
    },
    "食べたいもの優先": {
        "food": 0.45,
        "time": 0.15,
        "budget": 0.10,
        "hunger": 0.20,
        "variety": 0.05,
        "novelty": 0.05,
    },
    "安く楽しみたい": {
        "food": 0.25,
        "time": 0.15,
        "budget": 0.35,
        "hunger": 0.15,
        "variety": 0.05,
        "novelty": 0.05,
    },
    "たくさん歩きたい": {
        "food": 0.25,
        "time": 0.35,
        "budget": 0.10,
        "hunger": 0.15,
        "variety": 0.10,
        "novelty": 0.05,
    },
    "新しい店を試したい": {
        "food": 0.25,
        "time": 0.15,
        "budget": 0.10,
        "hunger": 0.10,
        "variety": 0.05,
        "novelty": 0.35,
    },
}

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                rgba(255, 210, 225, 0.18),
                transparent 26rem
            ),
            radial-gradient(
                circle at 90% 8%,
                rgba(220, 205, 255, 0.16),
                transparent 28rem
            );
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero,
    .route-card {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(235, 162, 190, 0.28);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
    }

    .hero {
        margin-bottom: 1.3rem;
    }

    .hero h1,
    .route-card h3 {
        margin: 0;
    }

    .hero p {
        margin: 0.6rem 0 0;
        opacity: 0.82;
    }

    .route-card {
        margin: 0.4rem 0 1rem;
    }

    div[data-testid="stMetric"] {
        padding: 0.8rem 1rem;
        border: 1px solid rgba(180, 180, 200, 0.18);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.04);
    }

    .stButton > button {
        min-height: 3rem;
        border-radius: 16px;
        font-weight: 750;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        min-height: 3rem;
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

    places_df = pd.read_csv(places_path)
    roads_df = pd.read_csv(roads_path)

    place_columns = {
        "id",
        "name",
        "role",
        "category",
        "x",
        "y",
        "food_type",
        "price",
        "fullness",
        "texture",
        "novelty",
    }

    road_columns = {
        "from_id",
        "to_id",
        "minutes",
        "quietness",
        "scenery",
    }

    missing_places = (
        place_columns - set(places_df.columns)
    )
    missing_roads = (
        road_columns - set(roads_df.columns)
    )

    if missing_places:
        raise ValueError(
            "places.csvに不足している列があります："
            + ", ".join(sorted(missing_places))
        )

    if missing_roads:
        raise ValueError(
            "roads.csvに不足している列があります："
            + ", ".join(sorted(missing_roads))
        )

    for column in [
        "x",
        "y",
        "price",
        "fullness",
        "novelty",
    ]:
        places_df[column] = pd.to_numeric(
            places_df[column],
            errors="coerce",
        )

    for column in [
        "minutes",
        "quietness",
        "scenery",
    ]:
        roads_df[column] = pd.to_numeric(
            roads_df[column],
            errors="coerce",
        )

    if places_df[
        [
            "x",
            "y",
            "price",
            "fullness",
            "novelty",
        ]
    ].isna().any().any():
        raise ValueError(
            "places.csvの数値列に読み取れない値があります。"
        )

    if roads_df[
        [
            "minutes",
            "quietness",
            "scenery",
        ]
    ].isna().any().any():
        raise ValueError(
            "roads.csvの数値列に読み取れない値があります。"
        )

    return places_df, roads_df


@st.cache_resource
def build_graph(
    place_records,
    road_records,
):
    graph = nx.Graph()

    for place in place_records:
        graph.add_node(
            place["id"],
            **place,
        )

    for road in road_records:
        graph.add_edge(
            road["from_id"],
            road["to_id"],
            minutes=float(
                road["minutes"]
            ),
            quietness=float(
                road["quietness"]
            ),
            scenery=float(
                road["scenery"]
            ),
        )

    return graph


try:
    places, roads = load_data()

except (
    FileNotFoundError,
    ValueError,
    pd.errors.ParserError,
) as error:
    st.error(str(error))
    st.stop()


route_graph = build_graph(
    tuple(
        places.to_dict("records")
    ),
    tuple(
        roads.to_dict("records")
    ),
)

place_by_id = (
    places
    .set_index("id")
    .to_dict("index")
)

name_to_id = dict(
    zip(
        places["name"],
        places["id"],
    )
)

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
    .astype(str)
    .unique()
    .tolist()
)


def clamp(value):
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def connect_waypoints(
    graph,
    waypoint_ids,
):
    full_path = []

    for first_id, second_id in zip(
        waypoint_ids,
        waypoint_ids[1:],
    ):
        segment = nx.shortest_path(
            graph,
            first_id,
            second_id,
            weight="minutes",
        )

        if not full_path:
            full_path.extend(segment)
        else:
            full_path.extend(
                segment[1:]
            )

    minutes = sum(
        float(
            graph[first][second][
                "minutes"
            ]
        )
        for first, second in zip(
            full_path,
            full_path[1:],
        )
    )

    return full_path, minutes


def score_route(
    stops,
    total_minutes,
    direct_minutes,
    selected_foods,
    walk_minutes,
    budget,
    hunger_level,
    mood,
):
    total_price = sum(
        int(stop["price"])
        for stop in stops
    )

    total_fullness = sum(
        float(stop["fullness"])
        for stop in stops
    )

    selected_food_set = set(
        selected_foods
    )

    if selected_food_set:
        food_score = (
            sum(
                stop["food_type"]
                in selected_food_set
                for stop in stops
            )
            / len(stops)
        )
    else:
        food_score = 0.65

    if total_minutes <= walk_minutes:
        time_score = (
            0.70
            + 0.30
            * total_minutes
            / max(
                walk_minutes,
                1,
            )
        )
    else:
        time_score = (
            1.0
            - (
                total_minutes
                - walk_minutes
            )
            / max(
                walk_minutes,
                1,
            )
        )

    if total_price <= budget:
        budget_score = 1.0
    else:
        budget_score = (
            1.0
            - (
                total_price
                - budget
            )
            / max(
                budget,
                1,
            )
        )

    target_fullness = (
        hunger_level + 0.5
    )

    hunger_score = (
        1.0
        - abs(
            total_fullness
            - target_fullness
        )
        / max(
            target_fullness,
            1,
        )
    )

    textures = {
        str(stop["texture"])
        for stop in stops
        if pd.notna(
            stop["texture"]
        )
    }

    variety_score = (
        len(textures)
        / len(stops)
    )

    novelty_score = (
        sum(
            float(stop["novelty"])
            for stop in stops
        )
        / len(stops)
    )

    efficiency_score = (
        direct_minutes
        / max(
            total_minutes,
            1,
        )
    )

    components = {
        "food": clamp(
            food_score
        ),
        "time": clamp(
            time_score
        ),
        "budget": clamp(
            budget_score
        ),
        "hunger": clamp(
            hunger_score
        ),
        "variety": clamp(
            variety_score
        ),
        "novelty": clamp(
            novelty_score
        ),
    }

    score = sum(
        MOOD_WEIGHTS[mood][key]
        * components[key]
        for key
        in MOOD_WEIGHTS[mood]
    )

    score += (
        0.05
        * clamp(
            efficiency_score
        )
    )

    time_over = max(
        0.0,
        total_minutes
        - walk_minutes,
    )

    budget_over = max(
        0.0,
        total_price
        - budget,
    )

    score -= min(
        0.35,
        time_over
        / max(
            walk_minutes,
            1,
        ),
    )

    score -= min(
        0.35,
        budget_over
        / max(
            budget,
            1,
        ),
    )

    return {
        "score": clamp(
            score
        ),
        "total_price": total_price,
        "total_fullness": total_fullness,
        "time_over": time_over,
        "budget_over": budget_over,
        "components": components,
    }


def generate_candidates(
    start_id,
    goal_id,
    selected_foods,
    walk_minutes,
    budget,
    hunger_level,
    mood,
):
    try:
        _, direct_minutes = (
            connect_waypoints(
                route_graph,
                [
                    start_id,
                    goal_id,
                ],
            )
        )
    except nx.NetworkXNoPath:
        return []

    candidates = []
    seen = set()

    food_ids = (
        food_places["id"]
        .tolist()
    )

    for stop_count in (1, 2):
        for stop_order in permutations(
            food_ids,
            stop_count,
        ):
            try:
                path, minutes = (
                    connect_waypoints(
                        route_graph,
                        [
                            start_id,
                            *stop_order,
                            goal_id,
                        ],
                    )
                )

            except nx.NetworkXNoPath:
                continue

            candidate_key = (
                tuple(stop_order),
                tuple(path),
            )

            if candidate_key in seen:
                continue

            seen.add(
                candidate_key
            )

            stops = [
                place_by_id[
                    stop_id
                ]
                for stop_id
                in stop_order
            ]

            score_data = score_route(
                stops,
                minutes,
                direct_minutes,
                selected_foods,
                walk_minutes,
                budget,
                hunger_level,
                mood,
            )

            candidates.append(
                {
                    "path": path,
                    "stop_ids": list(
                        stop_order
                    ),
                    "stops": stops,
                    "minutes": round(
                        minutes
                    ),
                    **score_data,
                }
            )

    candidates.sort(
        key=lambda route: (
            route["time_over"] == 0
            and route[
                "budget_over"
            ] == 0,
            route["score"],
        ),
        reverse=True,
    )

    return candidates


def select_top_routes(
    candidates,
    count=3,
):
    selected = []
    used_stop_sets = set()

    for route in candidates:
        stop_set = frozenset(
            route["stop_ids"]
        )

        if stop_set in used_stop_sets:
            continue

        selected.append(
            route
        )

        used_stop_sets.add(
            stop_set
        )

        if len(selected) == count:
            break

    return selected


def create_reason(
    selected_foods,
    route,
    walk_minutes,
    budget,
    mood,
):
    reasons = []

    matched = [
        stop["food_type"]
        for stop in route["stops"]
        if stop["food_type"]
        in selected_foods
    ]

    if matched:
        reasons.append(
            "食べたいもの（"
            + "・".join(
                dict.fromkeys(
                    matched
                )
            )
            + "）を楽しめる"
        )

    if (
        route["minutes"]
        <= walk_minutes
    ):
        reasons.append(
            "希望時間内で歩ける"
        )

    if (
        route["total_price"]
        <= budget
    ):
        reasons.append(
            "予算内に収まる"
        )

    textures = {
        str(stop["texture"])
        for stop
        in route["stops"]
        if pd.notna(
            stop["texture"]
        )
    }

    if len(textures) >= 2:
        reasons.append(
            "違う食感を楽しめる"
        )

    if (
        mood
        == "新しい店を試したい"
    ):
        reasons.append(
            "珍しさの高い店を含む"
        )

    if not reasons:
        reasons.append(
            "入力した条件全体との相性が高い"
        )

    return (
        "、".join(
            reasons[:3]
        )
        + "ルートです。"
    )


def create_route_figure(
    route,
):
    figure = go.Figure()

    for _, road in roads.iterrows():
        first = place_by_id[
            road["from_id"]
        ]

        second = place_by_id[
            road["to_id"]
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    first["x"],
                    second["x"],
                ],
                y=[
                    first["y"],
                    second["y"],
                ],
                mode="lines",
                line={
                    "width": 2,
                    "color": (
                        "rgba("
                        "150,150,170,.28"
                        ")"
                    ),
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    route_x = [
        place_by_id[
            place_id
        ]["x"]
        for place_id
        in route["path"]
    ]

    route_y = [
        place_by_id[
            place_id
        ]["y"]
        for place_id
        in route["path"]
    ]

    figure.add_trace(
        go.Scatter(
            x=route_x,
            y=route_y,
            mode="lines",
            line={
                "width": 7,
                "color": "#ff5f87",
            },
            name="おすすめルート",
            hoverinfo="skip",
        )
    )

    symbols = {
        "start": "circle",
        "goal": "star",
        "stop": "diamond",
    }

    labels = {
        "start": "出発地",
        "goal": "目的地",
        "stop": "買い食いスポット",
    }

    for role in [
        "start",
        "goal",
        "stop",
    ]:
        role_places = places[
            places["role"] == role
        ].copy()

        role_places[
            "food_type"
        ] = role_places[
            "food_type"
        ].fillna("")

        figure.add_trace(
            go.Scatter(
                x=role_places["x"],
                y=role_places["y"],
                mode="markers+text",
                text=role_places["name"],
                textposition=(
                    "top center"
                ),
                marker={
                    "size": 16,
                    "symbol": symbols[
                        role
                    ],
                    "line": {
                        "width": 1,
                        "color": "white",
                    },
                },
                customdata=role_places[
                    [
                        "category",
                        "food_type",
                        "price",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "種類："
                    "%{customdata[0]}<br>"
                    "食べ物："
                    "%{customdata[1]}<br>"
                    "価格："
                    "%{customdata[2]}円"
                    "<extra></extra>"
                ),
                name=labels[role],
            )
        )

    planned = [
        place_by_id[
            place_id
        ]
        for place_id
        in route["stop_ids"]
    ]

    figure.add_trace(
        go.Scatter(
            x=[
                place["x"]
                for place in planned
            ],
            y=[
                place["y"]
                for place in planned
            ],
            mode="markers",
            marker={
                "size": 27,
                "symbol": (
                    "circle-open"
                ),
                "line": {
                    "width": 4,
                    "color": "#ffd166",
                },
            },
            name="立ち寄る店",
            hoverinfo="skip",
        )
    )

    figure.update_layout(
        height=520,
        margin={
            "l": 10,
            "r": 10,
            "t": 25,
            "b": 10,
        },
        xaxis={
            "visible": False,
            "range": [
                places["x"].min()
                - 1,
                places["x"].max()
                + 1,
            ],
        },
        yaxis={
            "visible": False,
            "range": [
                places["y"].min()
                - 1,
                places["y"].max()
                + 1,
            ],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        hovermode="closest",
    )

    return figure


st.markdown(
    """
    <div class="hero">
        <h1>
            🥐 買い食い散歩ルート推薦
        </h1>
        <p>
            目的地まで歩きながら、
            その日の気分に合う
            買い食いスポットと
            寄り道ルートを提案します。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = (
    st.columns(3)
)

with metric1:
    st.metric(
        "登録地点",
        f"{len(places)}か所",
    )

with metric2:
    st.metric(
        "買い食いスポット",
        f"{len(food_places)}店",
    )

with metric3:
    st.metric(
        "登録ルート",
        f"{len(roads)}本",
    )

st.subheader(
    "散歩の条件"
)

left, right = st.columns(2)

with left:
    start_name = st.selectbox(
        "出発地",
        start_places,
    )

    walk_minutes = st.slider(
        "歩ける時間",
        20,
        90,
        45,
        5,
        format="%d分",
    )

    budget = st.slider(
        "予算",
        300,
        2000,
        1000,
        100,
        format="%d円",
    )

with right:
    goal_name = st.selectbox(
        "目的地",
        goal_places,
    )

    default_foods = [
        food
        for food in [
            "グミ",
            "お菓子",
        ]
        if food in food_types
    ]

    selected_foods = st.multiselect(
        "食べたいもの",
        food_types,
        default=default_foods,
        placeholder=(
            "複数選択できます"
        ),
    )

    hunger_level = st.slider(
        "空腹度",
        1,
        5,
        3,
        help=(
            "1は少しだけ、"
            "5はかなり空腹です。"
        ),
    )

mood = st.selectbox(
    "今日の気分",
    list(
        MOOD_WEIGHTS.keys()
    ),
)

if st.button(
    "おすすめルートを見る",
    type="primary",
    width="stretch",
):
    candidates = (
        generate_candidates(
            name_to_id[
                start_name
            ],
            name_to_id[
                goal_name
            ],
            selected_foods,
            walk_minutes,
            budget,
            hunger_level,
            mood,
        )
    )

    recommended_routes = (
        select_top_routes(
            candidates
        )
    )

    if not recommended_routes:
        st.warning(
            "条件に合う寄り道ルートを"
            "作れませんでした。"
        )

    else:
        st.subheader(
            "おすすめルート"
        )

        tabs = st.tabs(
            [
                (
                    f"{index}位　"
                    f"{route['score'] * 100:.0f}点"
                )
                for index, route
                in enumerate(
                    recommended_routes,
                    1,
                )
            ]
        )

        component_labels = {
            "food": "食べたいもの",
            "time": "歩行時間",
            "budget": "予算",
            "hunger": "空腹度",
            "variety": "食感の変化",
            "novelty": "店の珍しさ",
        }

        for index, (
            tab,
            route,
        ) in enumerate(
            zip(
                tabs,
                recommended_routes,
            ),
            1,
        ):
            with tab:
                stop_names = [
                    stop["name"]
                    for stop
                    in route["stops"]
                ]

                reason = create_reason(
                    selected_foods,
                    route,
                    walk_minutes,
                    budget,
                    mood,
                )

                st.markdown(
                    (
                        '<div class="route-card">'
                        f"<h3>{index}位　"
                        f'{" → ".join(stop_names)}'
                        "</h3>"
                        f"<p>{reason}</p>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4 = (
                    st.columns(4)
                )

                with m1:
                    st.metric(
                        "相性スコア",
                        (
                            f"{route['score'] * 100:.0f}点"
                        ),
                    )

                with m2:
                    st.metric(
                        "所要時間",
                        (
                            f"{route['minutes']}分"
                        ),
                    )

                with m3:
                    st.metric(
                        "予算目安",
                        (
                            f"{route['total_price']}円"
                        ),
                    )

                with m4:
                    st.metric(
                        "寄り道",
                        (
                            f"{len(route['stops'])}軒"
                        ),
                    )

                route_names = [
                    place_by_id[
                        place_id
                    ]["name"]
                    for place_id
                    in route["path"]
                ]

                st.write(
                    "**通る順番：** "
                    + " → ".join(
                        route_names
                    )
                )

                stop_df = pd.DataFrame(
                    [
                        {
                            "店名": stop[
                                "name"
                            ],
                            "種類": stop[
                                "category"
                            ],
                            "食べ物": stop[
                                "food_type"
                            ],
                            "価格": int(
                                stop["price"]
                            ),
                            "満腹度": int(
                                stop["fullness"]
                            ),
                            "食感": stop[
                                "texture"
                            ],
                        }
                        for stop
                        in route["stops"]
                    ]
                )

                component_df = (
                    pd.DataFrame(
                        [
                            {
                                "評価項目": (
                                    component_labels[
                                        key
                                    ]
                                ),
                                "一致度": round(
                                    value
                                    * 100
                                ),
                            }
                            for key, value
                            in route[
                                "components"
                            ].items()
                        ]
                    )
                )

                col1, col2 = (
                    st.columns(
                        [
                            1.05,
                            1.65,
                        ]
                    )
                )

                with col1:
                    st.markdown(
                        "#### 立ち寄る店"
                    )

                    st.dataframe(
                        stop_df,
                        width="stretch",
                        hide_index=True,
                    )

                    st.markdown(
                        "#### 条件との一致"
                    )

                    st.dataframe(
                        component_df,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "一致度": (
                                st.column_config
                                .ProgressColumn(
                                    "一致度",
                                    min_value=0,
                                    max_value=100,
                                    format="%d%%",
                                )
                            )
                        },
                    )

                with col2:
                    st.markdown(
                        "#### ダミー地図"
                    )

                    st.plotly_chart(
                        create_route_figure(
                            route
                        ),
                        width="stretch",
                        config={
                            "displayModeBar": False
                        },
                    )

                if (
                    route["time_over"] > 0
                    or route[
                        "budget_over"
                    ] > 0
                ):
                    st.warning(
                        "この案は希望条件を"
                        "一部超えています。"
                        "条件内の候補が"
                        "少ないため、"
                        "近い案として"
                        "表示しています。"
                    )

with st.expander(
    "この推薦システムの仕組み"
):
    st.write(
        """
        このアプリは、入力条件を使う
        **知識ベース型推薦**と、
        店舗の特徴を比べる
        **コンテンツベース型推薦**を
        組み合わせています。

        NetworkXで出発地・寄り道先・
        目的地を結ぶ経路を作り、
        食べたいもの、時間、予算、
        空腹度、食感の変化、
        店の珍しさを重み付きで評価します。

        その後、推薦スコアが高いルートを
        Top-N形式で3件表示します。
        """
    )

st.caption(
    "このデモでは架空の街・"
    "店舗・道路データを使用しています。"
)
