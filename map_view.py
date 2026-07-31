import plotly.graph_objects as go


def create_route_figure(
    route,
    start_id,
    goal_id,
    places,
    roads,
    place_by_id,
    food_places,
):
    figure = go.Figure()

    min_x = float(
        places["x"].min()
    ) - 1

    max_x = float(
        places["x"].max()
    ) + 1

    min_y = float(
        places["y"].min()
    ) - 1

    max_y = float(
        places["y"].max()
    ) + 1

    map_width = max_x - min_x
    map_height = max_y - min_y

    background_shapes = [
        {
            "type": "rect",
            "x0": min_x,
            "x1": min_x + map_width * 0.31,
            "y0": max_y - map_height * 0.30,
            "y1": max_y,
            "fillcolor": (
                "rgba(199, 232, 201, 0.72)"
            ),
            "line": {
                "width": 0,
            },
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": max_x - map_width * 0.27,
            "x1": max_x,
            "y0": min_y,
            "y1": min_y + map_height * 0.26,
            "fillcolor": (
                "rgba(207, 235, 214, 0.68)"
            ),
            "line": {
                "width": 0,
            },
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": min_x + map_width * 0.27,
            "x1": min_x + map_width * 0.49,
            "y0": min_y + map_height * 0.08,
            "y1": min_y + map_height * 0.31,
            "fillcolor": (
                "rgba(255, 221, 205, 0.60)"
            ),
            "line": {
                "color": (
                    "rgba(218, 181, 164, 0.35)"
                ),
                "width": 1,
            },
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": min_x + map_width * 0.57,
            "x1": min_x + map_width * 0.82,
            "y0": min_y + map_height * 0.56,
            "y1": min_y + map_height * 0.82,
            "fillcolor": (
                "rgba(240, 224, 245, 0.62)"
            ),
            "line": {
                "color": (
                    "rgba(195, 169, 204, 0.30)"
                ),
                "width": 1,
            },
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": min_x + map_width * 0.44,
            "x1": min_x + map_width * 0.50,
            "y0": min_y,
            "y1": max_y,
            "fillcolor": (
                "rgba(198, 229, 241, 0.52)"
            ),
            "line": {
                "width": 0,
            },
            "layer": "below",
        },
    ]

    for _, road in roads.iterrows():
        first = place_by_id[
            road["from_id"]
        ]

        second = place_by_id[
            road["to_id"]
        ]

        road_x = [
            first["x"],
            second["x"],
        ]

        road_y = [
            first["y"],
            second["y"],
        ]

        figure.add_trace(
            go.Scatter(
                x=road_x,
                y=road_y,
                mode="lines",
                line={
                    "width": 10,
                    "color": (
                        "rgba(255, 255, 255, 0.95)"
                    ),
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=road_x,
                y=road_y,
                mode="lines",
                line={
                    "width": 3,
                    "color": (
                        "rgba(160, 154, 165, 0.72)"
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
                "width": 14,
                "color": (
                    "rgba(255, 255, 255, 0.96)"
                ),
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=route_x,
            y=route_y,
            mode="lines",
            line={
                "width": 8,
                "color": "#ff6f91",
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
        for endpoint_id
        in endpoint_ids
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
            for endpoint_id
            in unselected_endpoint_ids
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    place["x"]
                    for place
                    in unselected_endpoints
                ],
                y=[
                    place["y"]
                    for place
                    in unselected_endpoints
                ],
                mode="markers+text",
                text=[
                    place["name"]
                    for place
                    in unselected_endpoints
                ],
                textposition="top center",
                textfont={
                    "size": 13,
                    "color": "#665d67",
                },
                marker={
                    "size": 16,
                    "symbol": "circle",
                    "color": "#d7d3dc",
                    "line": {
                        "width": 2,
                        "color": "#ffffff",
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
        for place_id
        in food_places["id"].tolist()
    ]

    figure.add_trace(
        go.Scatter(
            x=[
                place["x"]
                for place
                in all_food_places
            ],
            y=[
                place["y"]
                for place
                in all_food_places
            ],
            mode="markers+text",
            text=[
                place["name"]
                for place
                in all_food_places
            ],
            textposition="top center",
            textfont={
                "size": 13,
                "color": "#504650",
            },
            marker={
                "size": 17,
                "symbol": "diamond",
                "color": "#f3a25e",
                "line": {
                    "width": 2,
                    "color": "#ffffff",
                },
            },
            customdata=[
                [
                    place["category"],
                    place["food_type"],
                    int(
                        place["price"]
                    ),
                ]
                for place
                in all_food_places
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "種類：%{customdata[0]}<br>"
                "食べ物：%{customdata[1]}<br>"
                "価格：%{customdata[2]}円"
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
                    same_place["x"]
                ],
                y=[
                    same_place["y"]
                ],
                mode="markers+text",
                text=[
                    same_place["name"]
                ],
                textposition="bottom center",
                textfont={
                    "size": 14,
                    "color": "#4b414d",
                },
                marker={
                    "size": 26,
                    "symbol": "star",
                    "color": "#9f8ce8",
                    "line": {
                        "width": 3,
                        "color": "#ffffff",
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
                    start_place["x"]
                ],
                y=[
                    start_place["y"]
                ],
                mode="markers+text",
                text=[
                    start_place["name"]
                ],
                textposition="bottom center",
                textfont={
                    "size": 14,
                    "color": "#4b414d",
                },
                marker={
                    "size": 21,
                    "symbol": "circle",
                    "color": "#79cdb1",
                    "line": {
                        "width": 3,
                        "color": "#ffffff",
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
                    goal_place["x"]
                ],
                y=[
                    goal_place["y"]
                ],
                mode="markers+text",
                text=[
                    goal_place["name"]
                ],
                textposition="bottom center",
                textfont={
                    "size": 14,
                    "color": "#4b414d",
                },
                marker={
                    "size": 25,
                    "symbol": "star",
                    "color": "#9f8ce8",
                    "line": {
                        "width": 3,
                        "color": "#ffffff",
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
        for place_id
        in route["stop_ids"]
    ]

    figure.add_trace(
        go.Scatter(
            x=[
                place["x"]
                for place
                in planned_places
            ],
            y=[
                place["y"]
                for place
                in planned_places
            ],
            mode="markers",
            marker={
                "size": 29,
                "symbol": "circle-open",
                "line": {
                    "width": 5,
                    "color": "#ff6f91",
                },
            },
            name="立ち寄る店",
            hoverinfo="skip",
        )
    )

    figure.update_layout(
        height=540,
        margin={
            "l": 14,
            "r": 14,
            "t": 72,
            "b": 14,
        },
        paper_bgcolor=(
            "rgba(0, 0, 0, 0)"
        ),
        plot_bgcolor="#f8f4e9",
        shapes=background_shapes,
        font={
            "family": (
                "Hiragino Sans, "
                "Hiragino Kaku Gothic ProN, "
                "Yu Gothic UI, "
                "Yu Gothic, "
                "Noto Sans JP, "
                "sans-serif"
            ),
            "color": "#4b414d",
        },
        xaxis={
            "visible": False,
            "range": [
                min_x,
                max_x,
            ],
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "visible": False,
            "range": [
                min_y,
                max_y,
            ],
            "scaleanchor": "x",
            "scaleratio": 1,
            "showgrid": False,
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0,
            "bgcolor": (
                "rgba(255, 255, 255, 0.86)"
            ),
            "bordercolor": (
                "rgba(195, 181, 198, 0.75)"
            ),
            "borderwidth": 1,
            "font": {
                "size": 12,
            },
        },
        annotations=[
            {
                "x": max_x - 0.15,
                "y": min_y + 0.15,
                "xref": "x",
                "yref": "y",
                "text": (
                    "架空のよりみちタウン"
                ),
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "bottom",
                "font": {
                    "size": 11,
                    "color": "#8c7f89",
                },
                "bgcolor": (
                    "rgba(255, 255, 255, 0.70)"
                ),
                "borderpad": 4,
            }
        ],
        hovermode="closest",
    )

    return figure
