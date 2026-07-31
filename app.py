from itertools import combinations
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

MAX_ROUTE_OVERLAP = 0.60
ACCEPTABLE_SCORE_GAP = 0.12
MAX_STOPS = 4

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


# ==================================================
# デザイン
# ==================================================

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

    .hero p,
    .route-card p {
        margin: 0.6rem 0 0;
        opacity: 0.84;
        line-height: 1.7;
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

    .hunger-scale-labels {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.2rem 0 0.1rem;
        font-size: 0.88rem;
        font-weight: 650;
        opacity: 0.88;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        justify-content: space-between;
        width: 100%;
        gap: 0.25rem;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1;
        justify-content: center;
        margin: 0;
    }

    @media (max-width: 700px) {
        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            font-size: 0.78rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# データ読み込み
# ==================================================

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

    places_df = pd.read_csv(
        places_path
    )

    roads_df = pd.read_csv(
        roads_path
    )

    required_place_columns = {
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

    required_road_columns = {
        "from_id",
        "to_id",
        "minutes",
        "quietness",
        "scenery",
    }

    missing_places = (
        required_place_columns
        - set(
            places_df.columns
        )
    )

    missing_roads = (
        required_road_columns
        - set(
            roads_df.columns
        )
    )

    if missing_places:
        raise ValueError(
            "places.csvに不足している列があります："
            + ", ".join(
                sorted(
                    missing_places
                )
            )
        )

    if missing_roads:
        raise ValueError(
            "roads.csvに不足している列があります："
            + ", ".join(
                sorted(
                    missing_roads
                )
            )
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

    if (
        places_df[
            [
                "x",
                "y",
                "price",
                "fullness",
                "novelty",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "places.csvの数値列に"
            "読み取れない値があります。"
        )

    if (
        roads_df[
            [
                "minutes",
                "quietness",
                "scenery",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "roads.csvの数値列に"
            "読み取れない値があります。"
        )

    return (
        places_df,
        roads_df,
    )


def build_graph(
    places_df,
    roads_df,
):
    graph = nx.Graph()

    for place in places_df.to_dict(
        "records"
    ):
        graph.add_node(
            place["id"],
            **place,
        )

    for road in roads_df.to_dict(
        "records"
    ):
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
    st.error(
        str(
            error
        )
    )
    st.stop()


route_graph = build_graph(
    places,
    roads,
)

place_by_id = (
    places
    .set_index(
        "id"
    )
    .to_dict(
        "index"
    )
)

name_to_id = dict(
    zip(
        places["name"],
        places["id"],
    )
)

endpoint_places = places.loc[
    places["role"].isin(
        [
            "start",
            "goal",
        ]
    ),
    "name",
].tolist()

food_places = places.loc[
    places["role"] == "stop"
].copy()

food_ids = set(
    food_places[
        "id"
    ].tolist()
)

food_types = sorted(
    food_places[
        "food_type"
    ]
    .dropna()
    .astype(
        str
    )
    .unique()
    .tolist()
)


# ==================================================
# 経路計算
# ==================================================

def clamp(
    value
):
    return max(
        0.0,
        min(
            1.0,
            float(
                value
            ),
        ),
    )


def get_path_minutes(
    path
):
    return sum(
        float(
            route_graph[
                first
            ][
                second
            ][
                "minutes"
            ]
        )
        for first, second in zip(
            path,
            path[1:],
        )
    )


def route_edge_set(
    path
):
    return {
        frozenset(
            (
                first,
                second,
            )
        )
        for first, second in zip(
            path,
            path[1:],
        )
    }


def has_repeated_road(
    path
):
    edges = [
        frozenset(
            (
                first,
                second,
            )
        )
        for first, second in zip(
            path,
            path[1:],
        )
    ]

    return (
        len(
            edges
        )
        != len(
            set(
                edges
            )
        )
    )


def is_simple_route(
    path,
    start_id,
    goal_id,
):
    if not path:
        return False

    if (
        path[0] != start_id
        or path[-1] != goal_id
    ):
        return False

    if has_repeated_road(
        path
    ):
        return False

    if start_id == goal_id:
        if len(
            path
        ) < 4:
            return False

        internal_nodes = path[
            1:-1
        ]

        if start_id in internal_nodes:
            return False

        return (
            len(
                internal_nodes
            )
            == len(
                set(
                    internal_nodes
                )
            )
        )

    return (
        len(
            path
        )
        == len(
            set(
                path
            )
        )
    )


def canonical_cycle_key(
    path
):
    forward = tuple(
        path
    )

    backward = tuple(
        [
            path[0],
        ]
        + list(
            reversed(
                path[1:-1]
            )
        )
        + [
            path[0],
        ]
    )

    return min(
        forward,
        backward,
    )


def route_overlap(
    first_route,
    second_route,
):
    first_edges = route_edge_set(
        first_route[
            "path"
        ]
    )

    second_edges = route_edge_set(
        second_route[
            "path"
        ]
    )

    if (
        not first_edges
        or not second_edges
    ):
        return 1.0

    shared_edges = (
        first_edges
        & second_edges
    )

    smaller_size = min(
        len(
            first_edges
        ),
        len(
            second_edges
        ),
    )

    return (
        len(
            shared_edges
        )
        / smaller_size
    )


def is_feasible(
    route
):
    return (
        route[
            "time_over"
        ] == 0
        and route[
            "budget_over"
        ] == 0
    )


def generate_simple_paths(
    start_id,
    goal_id,
):
    cutoff = (
        len(
            route_graph.nodes
        )
        - 1
    )

    return list(
        nx.all_simple_paths(
            route_graph,
            source=start_id,
            target=goal_id,
            cutoff=cutoff,
        )
    )


def generate_simple_cycles(
    start_id
):
    neighbors = list(
        route_graph.neighbors(
            start_id
        )
    )

    graph_without_start = (
        route_graph.copy()
    )

    graph_without_start.remove_node(
        start_id
    )

    cycles = []
    seen_cycles = set()

    cutoff = (
        len(
            route_graph.nodes
        )
        - 2
    )

    for (
        first_neighbor,
        second_neighbor,
    ) in combinations(
        neighbors,
        2,
    ):
        inner_paths = (
            nx.all_simple_paths(
                graph_without_start,
                source=first_neighbor,
                target=second_neighbor,
                cutoff=cutoff,
            )
        )

        for inner_path in inner_paths:
            cycle = [
                start_id,
                *inner_path,
                start_id,
            ]

            if not is_simple_route(
                cycle,
                start_id,
                start_id,
            ):
                continue

            cycle_key = canonical_cycle_key(
                cycle
            )

            if cycle_key in seen_cycles:
                continue

            seen_cycles.add(
                cycle_key
            )

            cycles.append(
                cycle
            )

    return cycles


# ==================================================
# 推薦スコア
# ==================================================

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
        int(
            stop[
                "price"
            ]
        )
        for stop in stops
    )

    total_fullness = sum(
        float(
            stop[
                "fullness"
            ]
        )
        for stop in stops
    )

    selected_food_set = set(
        selected_foods
    )

    if selected_food_set:
        food_score = (
            sum(
                stop[
                    "food_type"
                ]
                in selected_food_set
                for stop in stops
            )
            / len(
                stops
            )
        )

    else:
        food_score = 0.65

    if (
        total_minutes
        <= walk_minutes
    ):
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
        hunger_level
        + 0.5
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
        str(
            stop[
                "texture"
            ]
        )
        for stop in stops
        if pd.notna(
            stop[
                "texture"
            ]
        )
    }

    variety_score = (
        len(
            textures
        )
        / len(
            stops
        )
    )

    novelty_score = (
        sum(
            float(
                stop[
                    "novelty"
                ]
            )
            for stop in stops
        )
        / len(
            stops
        )
    )

    if direct_minutes > 0:
        efficiency_score = (
            direct_minutes
            / max(
                total_minutes,
                1,
            )
        )

    else:
        efficiency_score = 1.0

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
        MOOD_WEIGHTS[
            mood
        ][
            key
        ]
        * components[
            key
        ]
        for key in MOOD_WEIGHTS[
            mood
        ]
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


# ==================================================
# 候補生成
# ==================================================

def generate_candidates(
    start_id,
    goal_id,
    selected_foods,
    walk_minutes,
    budget,
    hunger_level,
    mood,
):
    if start_id == goal_id:
        paths = generate_simple_cycles(
            start_id
        )

        if not paths:
            return []

        direct_minutes = min(
            get_path_minutes(
                path
            )
            for path in paths
        )

    else:
        paths = generate_simple_paths(
            start_id,
            goal_id,
        )

        try:
            direct_minutes = (
                nx.shortest_path_length(
                    route_graph,
                    source=start_id,
                    target=goal_id,
                    weight="minutes",
                )
            )

        except nx.NetworkXNoPath:
            return []

    candidates = []
    seen_paths = set()

    for path in paths:
        if not is_simple_route(
            path,
            start_id,
            goal_id,
        ):
            continue

        stop_ids = [
            place_id
            for place_id in path[
                1:-1
            ]
            if place_id in food_ids
        ]

        if not (
            1
            <= len(
                stop_ids
            )
            <= MAX_STOPS
        ):
            continue

        if start_id == goal_id:
            path_key = canonical_cycle_key(
                path
            )

        else:
            path_key = tuple(
                path
            )

        if path_key in seen_paths:
            continue

        seen_paths.add(
            path_key
        )

        stops = [
            place_by_id[
                stop_id
            ]
            for stop_id in stop_ids
        ]

        total_minutes = get_path_minutes(
            path
        )

        score_data = score_route(
            stops=stops,
            total_minutes=total_minutes,
            direct_minutes=direct_minutes,
            selected_foods=selected_foods,
            walk_minutes=walk_minutes,
            budget=budget,
            hunger_level=hunger_level,
            mood=mood,
        )

        candidates.append(
            {
                "path": path,
                "stop_ids": stop_ids,
                "stops": stops,
                "minutes": round(
                    total_minutes
                ),
                **score_data,
            }
        )

    candidates.sort(
        key=lambda route: (
            is_feasible(
                route
            ),
            route[
                "score"
            ],
        ),
        reverse=True,
    )

    return candidates


def select_top_routes(
    candidates,
    count=3,
    max_overlap=MAX_ROUTE_OVERLAP,
    acceptable_score_gap=ACCEPTABLE_SCORE_GAP,
):
    if not candidates:
        return []

    best_route = candidates[0]
    best_score = best_route[
        "score"
    ]
    best_is_feasible = is_feasible(
        best_route
    )

    selected = [
        best_route
    ]

    selected_ids = {
        id(
            best_route
        )
    }

    def is_acceptable(
        route
    ):
        score_ok = (
            route[
                "score"
            ]
            >= best_score
            - acceptable_score_gap
        )

        if best_is_feasible:
            feasibility_ok = is_feasible(
                route
            )

        else:
            feasibility_ok = True

        return (
            score_ok
            and feasibility_ok
        )

    acceptable_routes = [
        route
        for route in candidates[
            1:
        ]
        if is_acceptable(
            route
        )
    ]

    while (
        len(
            selected
        ) < count
        and acceptable_routes
    ):
        ranked_choices = []

        for route in acceptable_routes:
            maximum_overlap = max(
                route_overlap(
                    route,
                    selected_route,
                )
                for selected_route in selected
            )

            ranked_choices.append(
                (
                    maximum_overlap,
                    -route[
                        "score"
                    ],
                    route,
                )
            )

        ranked_choices.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        (
            overlap,
            _,
            route,
        ) = ranked_choices[0]

        if overlap > max_overlap:
            break

        selected.append(
            route
        )

        selected_ids.add(
            id(
                route
            )
        )

        acceptable_routes = [
            candidate
            for candidate in acceptable_routes
            if id(
                candidate
            )
            != id(
                route
            )
        ]

    if len(
        selected
    ) < count:
        remaining_acceptable = [
            route
            for route in candidates[
                1:
            ]
            if (
                id(
                    route
                )
                not in selected_ids
                and is_acceptable(
                    route
                )
            )
        ]

        remaining_acceptable.sort(
            key=lambda route: (
                max(
                    route_overlap(
                        route,
                        selected_route,
                    )
                    for selected_route in selected
                ),
                -route[
                    "score"
                ],
            )
        )

        for route in remaining_acceptable:
            selected.append(
                route
            )

            selected_ids.add(
                id(
                    route
                )
            )

            if len(
                selected
            ) >= count:
                break

    if len(
        selected
    ) < count:
        for route in candidates[
            1:
        ]:
            if id(
                route
            ) in selected_ids:
                continue

            selected.append(
                route
            )

            selected_ids.add(
                id(
                    route
                )
            )

            if len(
                selected
            ) >= count:
                break

    annotated_routes = []

    for index, route in enumerate(
        selected
    ):
        annotated_route = dict(
            route
        )

        if index == 0:
            overlap_with_higher = 0.0

        else:
            overlap_with_higher = max(
                route_overlap(
                    route,
                    higher_route,
                )
                for higher_route in selected[
                    :index
                ]
            )

        annotated_route[
            "overlap_with_higher"
        ] = overlap_with_higher

        annotated_route[
            "diversity_fallback"
        ] = (
            index > 0
            and overlap_with_higher
            > max_overlap
        )

        annotated_routes.append(
            annotated_route
        )

    return annotated_routes


# ==================================================
# 表示補助
# ==================================================

def create_reason(
    selected_foods,
    route,
    walk_minutes,
    budget,
    mood,
    same_endpoint,
):
    reasons = []

    matched_foods = [
        stop[
            "food_type"
        ]
        for stop in route[
            "stops"
        ]
        if stop[
            "food_type"
        ] in selected_foods
    ]

    if matched_foods:
        reasons.append(
            "食べたいもの（"
            + "・".join(
                dict.fromkeys(
                    matched_foods
                )
            )
            + "）を楽しめる"
        )

    if (
        route[
            "minutes"
        ]
        <= walk_minutes
    ):
        reasons.append(
            "希望時間内で歩ける"
        )

    if (
        route[
            "total_price"
        ]
        <= budget
    ):
        reasons.append(
            "予算内に収まる"
        )

    textures = {
        str(
            stop[
                "texture"
            ]
        )
        for stop in route[
            "stops"
        ]
        if pd.notna(
            stop[
                "texture"
            ]
        )
    }

    if len(
        textures
    ) >= 2:
        reasons.append(
            "違う食感を楽しめる"
        )

    if mood == "新しい店を試したい":
        reasons.append(
            "珍しさの高い店を含む"
        )

    if same_endpoint:
        reasons.append(
            "同じ道を往復せず"
            "出発地点に戻れる"
        )

    if not reasons:
        reasons.append(
            "入力した条件全体との"
            "相性が高い"
        )

    return (
        "、".join(
            reasons[
                :3
            ]
        )
        + "ルートです。"
    )


def create_route_figure(
    route,
    start_id,
    goal_id,
):
    figure = go.Figure()

    for _, road in roads.iterrows():
        first = place_by_id[
            road[
                "from_id"
            ]
        ]

        second = place_by_id[
            road[
                "to_id"
            ]
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    first[
                        "x"
                    ],
                    second[
                        "x"
                    ],
                ],
                y=[
                    first[
                        "y"
                    ],
                    second[
                        "y"
                    ],
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
        ][
            "x"
        ]
        for place_id in route[
            "path"
        ]
    ]

    route_y = [
        place_by_id[
            place_id
        ][
            "y"
        ]
        for place_id in route[
            "path"
        ]
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

    endpoint_ids = places.loc[
        places["role"].isin(
            [
                "start",
                "goal",
            ]
        ),
        "id",
    ].tolist()

    unselected_endpoint_ids = [
        endpoint_id
        for endpoint_id in endpoint_ids
        if endpoint_id
        not in {
            start_id,
            goal_id,
        }
    ]

    if unselected_endpoint_ids:
        unselected_endpoints = [
            place_by_id[
                endpoint_id
            ]
            for endpoint_id in unselected_endpoint_ids
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    place[
                        "x"
                    ]
                    for place in unselected_endpoints
                ],
                y=[
                    place[
                        "y"
                    ]
                    for place in unselected_endpoints
                ],
                mode="markers+text",
                text=[
                    place[
                        "name"
                    ]
                    for place in unselected_endpoints
                ],
                textposition="top center",
                marker={
                    "size": 14,
                    "symbol": "circle",
                    "color": "#9ca3af",
                    "line": {
                        "width": 1,
                        "color": "white",
                    },
                },
                name="選択可能な地点",
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<extra></extra>"
                ),
            )
        )

    all_food_places = [
        place_by_id[
            place_id
        ]
        for place_id in food_places[
            "id"
        ].tolist()
    ]

    figure.add_trace(
        go.Scatter(
            x=[
                place[
                    "x"
                ]
                for place in all_food_places
            ],
            y=[
                place[
                    "y"
                ]
                for place in all_food_places
            ],
            mode="markers+text",
            text=[
                place[
                    "name"
                ]
                for place in all_food_places
            ],
            textposition="top center",
            marker={
                "size": 15,
                "symbol": "diamond",
                "color": "#1687d9",
                "line": {
                    "width": 1,
                    "color": "white",
                },
            },
            customdata=[
                [
                    place[
                        "category"
                    ],
                    place[
                        "food_type"
                    ],
                    int(
                        place[
                            "price"
                        ]
                    ),
                ]
                for place in all_food_places
            ],
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
            name="買い食いスポット",
        )
    )

    if start_id == goal_id:
        same_place = place_by_id[
            start_id
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    same_place[
                        "x"
                    ]
                ],
                y=[
                    same_place[
                        "y"
                    ]
                ],
                mode="markers+text",
                text=[
                    same_place[
                        "name"
                    ]
                ],
                textposition="bottom center",
                marker={
                    "size": 23,
                    "symbol": "star",
                    "color": "#f6c453",
                    "line": {
                        "width": 2,
                        "color": "white",
                    },
                },
                name="出発地・目的地",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "出発地・目的地"
                    "<extra></extra>"
                ),
            )
        )

    else:
        start_place = place_by_id[
            start_id
        ]

        goal_place = place_by_id[
            goal_id
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    start_place[
                        "x"
                    ]
                ],
                y=[
                    start_place[
                        "y"
                    ]
                ],
                mode="markers+text",
                text=[
                    start_place[
                        "name"
                    ]
                ],
                textposition="bottom center",
                marker={
                    "size": 19,
                    "symbol": "circle",
                    "color": "#d9e2ef",
                    "line": {
                        "width": 2,
                        "color": "white",
                    },
                },
                name="出発地",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "出発地"
                    "<extra></extra>"
                ),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=[
                    goal_place[
                        "x"
                    ]
                ],
                y=[
                    goal_place[
                        "y"
                    ]
                ],
                mode="markers+text",
                text=[
                    goal_place[
                        "name"
                    ]
                ],
                textposition="bottom center",
                marker={
                    "size": 22,
                    "symbol": "star",
                    "color": "#54a8e8",
                    "line": {
                        "width": 2,
                        "color": "white",
                    },
                },
                name="目的地",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "目的地"
                    "<extra></extra>"
                ),
            )
        )

    planned_places = [
        place_by_id[
            place_id
        ]
        for place_id in route[
            "stop_ids"
        ]
    ]

    figure.add_trace(
        go.Scatter(
            x=[
                place[
                    "x"
                ]
                for place in planned_places
            ],
            y=[
                place[
                    "y"
                ]
                for place in planned_places
            ],
            mode="markers",
            marker={
                "size": 27,
                "symbol": "circle-open",
                "line": {
                    "width": 4,
                    "color": "#f5a0aa",
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
                places[
                    "x"
                ].min()
                - 1,
                places[
                    "x"
                ].max()
                + 1,
            ],
        },
        yaxis={
            "visible": False,
            "range": [
                places[
                    "y"
                ].min()
                - 1,
                places[
                    "y"
                ].max()
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


# ==================================================
# 画面
# ==================================================

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

metric1, metric2, metric3 = st.columns(
    3
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

left, right = st.columns(
    2
)

with left:
    start_name = st.selectbox(
        "出発地",
        endpoint_places,
        index=0,
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

with right:
    default_goal_index = (
        endpoint_places.index(
            "市立図書館"
        )
        if "市立図書館"
        in endpoint_places
        else 0
    )

    goal_name = st.selectbox(
        "目的地",
        endpoint_places,
        index=default_goal_index,
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
        placeholder="複数選択できます",
    )

    st.markdown(
        """
        <div class="hunger-scale-labels">
            <span>お腹が空いている</span>
            <span>空いていない</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hunger_level = st.radio(
        "空腹度",
        options=[
            5,
            4,
            3,
            2,
            1,
        ],
        index=2,
        horizontal=True,
        label_visibility="collapsed",
        format_func=lambda value: {
            5: "かなり",
            4: "やや",
            3: "ふつう",
            2: "少し",
            1: "なし",
        }[
            value
        ],
    )

st.caption(
    "出発地と目的地は"
    "同じ地点でも選べます。"
    "同じ地点を選ぶと、"
    "同じ道を往復しない"
    "周遊ルートを提案します。"
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
    start_id = name_to_id[
        start_name
    ]

    goal_id = name_to_id[
        goal_name
    ]

    candidates = generate_candidates(
        start_id=start_id,
        goal_id=goal_id,
        selected_foods=selected_foods,
        walk_minutes=walk_minutes,
        budget=budget,
        hunger_level=hunger_level,
        mood=mood,
    )

    recommended_routes = select_top_routes(
        candidates
    )

    if not recommended_routes:
        st.warning(
            "同じ道路を繰り返さずに通れる"
            "寄り道ルートが見つかりませんでした。"
            "歩ける時間を長くするか、"
            "出発地・目的地を変更してください。"
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
                for index, route in enumerate(
                    recommended_routes,
                    start=1,
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
            start=1,
        ):
            with tab:
                stop_names = [
                    stop[
                        "name"
                    ]
                    for stop in route[
                        "stops"
                    ]
                ]

                reason = create_reason(
                    selected_foods=selected_foods,
                    route=route,
                    walk_minutes=walk_minutes,
                    budget=budget,
                    mood=mood,
                    same_endpoint=(
                        start_id
                        == goal_id
                    ),
                )

                st.markdown(
                    f"""
                    <div class="route-card">
                        <h3>
                            {index}位　
                            {" → ".join(stop_names)}
                        </h3>
                        <p>{reason}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4 = st.columns(
                    4
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
                    ][
                        "name"
                    ]
                    for place_id in route[
                        "path"
                    ]
                ]

                st.write(
                    "**通る順番：** "
                    + " → ".join(
                        route_names
                    )
                )

                if route[
                    "diversity_fallback"
                ]:
                    st.info(
                        "別の道を通る候補では"
                        "相性スコアが大きく下がるため、"
                        "上位ルートと一部同じ道を通る案を"
                        "表示しています。"
                    )

                elif index > 1:
                    route_difference = round(
                        (
                            1
                            - route[
                                "overlap_with_higher"
                            ]
                        )
                        * 100
                    )

                    st.caption(
                        "上位ルートとの経路差："
                        f"{route_difference}%"
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
                                stop[
                                    "price"
                                ]
                            ),
                            "満腹度": int(
                                stop[
                                    "fullness"
                                ]
                            ),
                            "食感": stop[
                                "texture"
                            ],
                        }
                        for stop in route[
                            "stops"
                        ]
                    ]
                )

                component_df = pd.DataFrame(
                    [
                        {
                            "評価項目": component_labels[
                                key
                            ],
                            "一致度": round(
                                value
                                * 100
                            ),
                        }
                        for key, value in route[
                            "components"
                        ].items()
                    ]
                )

                col1, col2 = st.columns(
                    [
                        1.05,
                        1.65,
                    ]
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
                            "一致度": st.column_config.ProgressColumn(
                                "一致度",
                                min_value=0,
                                max_value=100,
                                format="%d%%",
                            )
                        },
                    )

                with col2:
                    st.markdown(
                        "#### ダミー地図"
                    )

                    st.plotly_chart(
                        create_route_figure(
                            route=route,
                            start_id=start_id,
                            goal_id=goal_id,
                        ),
                        width="stretch",
                        config={
                            "displayModeBar": False,
                        },
                    )

                if (
                    route[
                        "time_over"
                    ] > 0
                    or route[
                        "budget_over"
                    ] > 0
                ):
                    st.warning(
                        "この案は希望条件を"
                        "一部超えています。"
                        "条件内の候補が少ないため、"
                        "近い案として表示しています。"
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

        NetworkXで地点と道路をグラフとして表し、
        出発地から目的地までの
        **単純経路**を列挙します。

        出発地と目的地が同じ場合は、
        同じ道路や地点を途中で繰り返さない
        **単純な周遊経路**だけを候補にします。

        各候補は、食べたいもの、時間、予算、
        空腹度、食感の変化、店の珍しさを
        重み付きで評価します。

        上位3件を選ぶ際は、
        希望条件との相性を大きく落とさない範囲で、
        上位案と異なる道路を通るルートを
        優先します。
        """
    )

st.caption(
    "このデモでは架空の街・"
    "店舗・道路データを使用しています。"
)
