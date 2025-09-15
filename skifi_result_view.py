import argparse
import base64
import json
import typing as t
import io
import ipaddress

import dash
import dash.html as dhtml
import dash.dcc as dcc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dcc.Loading(children=[
    dbc.Container([
        dhtml.Table(children=[
            dhtml.Tr(children=[
                dhtml.Td(children=[dhtml.H3("Upload Files")], colSpan=2)
            ], style={"height": "5%"}),
            dhtml.Tr(children=[
                dhtml.Td(children=[
                    dhtml.Div([
                        dhtml.Div("Select Time Data"),
                        dcc.Upload(id='upload-timeseries', children=dhtml.Button("Upload")),
                        dcc.Checklist(id='signal-selection', options=[])
                    ])
                ], style={"width": "15%", "vertical-align": "top"}),
                dhtml.Td(children=[
                    dcc.Graph(id='graph', style={"height": "90vh", "width": "80%"}),
                ], style={"width": "85%"})
            ], style={"height": "95%"}),
        ], style={"width": "100%", "height": "95%"}),
        dcc.Store(id="raw_signals"),
        dcc.Store(id="preprocessed"),
        dcc.Store(id="time_limits"),
        dcc.Store(id="ski_events"),
        dcc.Store(id="estimated_class")
    ], fluid=True),
])

def _decode_content(content: str) -> str:
    """Helper to decode uploaded file contents into a DataFrame."""
    content_type, content_string = content.split(',')
    return base64.b64decode(content_string).decode('utf-8')


def parse_skifi_debug_data(content: str, filename: str) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decoded = _decode_content(content)
    content_parsed = json.loads(decoded)

    windowed_data = content_parsed["WindowedData"]
    ski_data = content_parsed["SkiData"]
    time_segment = content_parsed["TimeSegment"]

    ski_data_df: pd.DataFrame = pd.DataFrame(ski_data)
    time_segment_df: pd.DataFrame = pd.DataFrame(time_segment)

    preprocessed_list = []
    raw_data_list = {}
    raw_estimated_class = []

    for idx, wins in enumerate(windowed_data):
        raw_estimated_class.append({
            "estimatedClass": wins["estimatedClass"],
            "startTimeSecs": idx * 500 /100,
            "endTimeSecs": (idx + 1) * 500 /100 - 1
        })
        preprocessed_list.append(wins["preprocessed"])
        for name, signal in wins["data"].items():
            if name not in raw_data_list:
                raw_data_list[name] = []
            raw_data_list[name].extend(signal)

    preprocessed: pd.DataFrame = pd.DataFrame(preprocessed_list)
    raw_signals: pd.DataFrame = pd.DataFrame(raw_data_list)
    raw_estimated_class_df: pd.DataFrame = pd.DataFrame(raw_estimated_class)

    return raw_signals, preprocessed, ski_data_df, time_segment_df, raw_estimated_class_df


@app.callback(
    dash.Output('signal-selection', 'options'),
    dash.Output('raw_signals', 'data'),
    dash.Output('preprocessed', 'data'),
    dash.Output('time_limits', 'data'),
    dash.Output('ski_events', 'data'),
    dash.Output('estimated_class', 'data'),
    dash.Input('upload-timeseries', 'contents'),
    dash.State('upload-timeseries', 'filename'),
    prevent_initial_call=True
)
def update_graph(ts_content, ts_filename):
    if (ts_content is None) or (ts_filename is None):
        return ["Signal 1"], "", "", "", ""

    raw_signals, preprocessed, ski_events, time_limits, estimated_class_df = parse_skifi_debug_data(content=ts_content, filename=ts_filename)

    raw_signals_json: str = raw_signals.to_json()
    preprocessed_json: str = preprocessed.to_json()
    time_limits_json: str = time_limits.to_json()
    ski_events_json: str = ski_events.to_json()
    estimated_class_json: str = estimated_class_df.to_json()

    signames = list(raw_signals.columns)
    signames.sort()

    return signames, raw_signals_json, preprocessed_json, time_limits_json, ski_events_json, estimated_class_json

@app.callback(
    dash.Output('graph', 'figure'),
    dash.Input('signal-selection', 'value'),
    dash.State('raw_signals', 'data'),
    dash.State('preprocessed', 'data'),
    dash.State('time_limits', 'data'),
    dash.State('ski_events', 'data'),
    dash.State('estimated_class', 'data'),
    prevent_initial_call=True
)
def redraw_graph(selected_signals, raw_signals_json, preprocessed_json, time_limits_json, ski_events_json, estimated_class_json):
    fig = go.Figure()
    raw_signals: pd.DataFrame = pd.read_json(io.StringIO(raw_signals_json))
    time_axis_values = raw_signals.index.to_numpy() / 100.0
    preprocessed: pd.DataFrame = pd.read_json(io.StringIO(preprocessed_json))
    time_limits: pd.DataFrame = pd.read_json(io.StringIO(time_limits_json))
    ski_events: pd.DataFrame = pd.read_json(io.StringIO(ski_events_json))
    estimated_class: pd.DataFrame = pd.read_json(io.StringIO(estimated_class_json))

    top_row_up = raw_signals[selected_signals].max().max()
    top_row_up = top_row_up if np.isfinite(top_row_up) else 1000
    top_row_down = max(top_row_up - 500, 0.0)
    bottom_row_down = raw_signals[selected_signals].min().min()
    bottom_row_down = bottom_row_down if np.isfinite(bottom_row_down) else -1000

    action_segments_x: np.ndarray = time_limits["segmentStartInSecs"].to_numpy(dtype=float)
    action_segments_x = np.append(action_segments_x, time_limits["segmentEndInSecs"].iloc[-1])
    action_segments_y = np.array([top_row_down, top_row_up])
    action_segments_z = time_limits["categoryId"].map(lambda x: 1.0 if x == "SkiSegments.uphill" else 0.0).to_numpy()
    action_segments_z = np.reshape(action_segments_z, shape=[1, action_segments_z.shape[0]])
    action_segments_texts = time_limits["categoryId"].map(lambda x: "UPHILL" if x == "SkiSegments.uphill" else "DOWNHILL").tolist()
    action_segments_texts = [action_segments_texts]

    fig.add_trace(go.Heatmap(x=action_segments_x, y=action_segments_y, z=action_segments_z, colorscale=[[0.0, 'rgb(255,128,0)'], [1.0, 'rgb(128,255,0)']], showscale=False, text=action_segments_texts, hoverinfo='text', showlegend=False))

    est_cls_x: np.ndarray = estimated_class["startTimeSecs"].to_numpy(dtype=float)
    est_cls_x = np.append(est_cls_x, estimated_class["endTimeSecs"].iloc[-1])
    est_cls_y = np.array([top_row_up, top_row_up+250])

    z_conv_map = {cls: idx for idx, cls in enumerate(estimated_class["estimatedClass"].unique())}
    est_cls_z = np.array([z_conv_map[c] for c in estimated_class["estimatedClass"]])
    est_cls_z = np.reshape(est_cls_z, shape=[1, est_cls_z.shape[0]])
    est_cls_texts = [[f"{x}" for x in estimated_class["estimatedClass"]]]

    fig.add_trace(go.Heatmap(x=est_cls_x, y=est_cls_y, z=est_cls_z, showscale=False, text=est_cls_texts, hoverinfo='text', showlegend=False))

    ski_events_list = []
    for row_num, row in ski_events.iterrows():
        txt = ""

        event_time = row["timeSinceStartSecs"]
        time_win = time_limits.loc[(time_limits["segmentStartInSecs"] < event_time) & (time_limits["segmentEndInSecs"] >= event_time)]

        event_type = "NOT KNOWN"
        if time_win.shape[0] == 1:
            event_type = time_win["categoryId"].iloc[0]

        if event_type == "SkiSegments.uphill":
            txt = f"""
<b>Step Length</b>: {row['stepLength']:0.2f}<br>
<b>Step Stroke</b>: {row['stepStroke']:0.2f}<br>
<b>Step Slide</b>: {row['stepSlide']:0.2f}<br>
"""
        elif event_type == "SkiSegments.downhill":
            txt = f"""
<b>Arc</b>: {row['arc']:0.2f}<br>
<b>Gradient</b>: {row['grad']:0.2f}<br>
<b>Load in N</b>: {row['loadInN']:0.2f}<br>
<b>Load in KG</b>: {row['loadInKgf']:0.2f}<br>
"""
        else:
            txt = f"""
<b>Step Length</b>: {row['stepLength']:0.2f}<br>
<b>Step Stroke</b>: {row['stepStroke']:0.2f}<br>
<b>Step Slide</b>: {row['stepSlide']:0.2f}<br>
<b>Arc</b>: {row['arc']:0.2f}<br>
<b>Gradient</b>: {row['grad']:0.2f}<br>
<b>Load in N</b>: {row['loadInN']:0.2f}<br>
<b>Load in KG</b>: {row['loadInKgf']:0.2f}<br>
"""

        ski_events_list.append({
            "x": row["timeSinceStartSecs"],
            "y": bottom_row_down,
            "text": txt
        })
    ski_events_makers = pd.DataFrame(ski_events_list)
    fig.add_trace(go.Scatter(x=ski_events_makers["x"], y=ski_events_makers["y"], mode="markers", text=ski_events_makers["text"], hoverinfo="text", showlegend=False))

    for sel_sig in selected_signals:
        graph_line = go.Scatter(x=time_axis_values, y=raw_signals[sel_sig], showlegend=True, name=sel_sig)
        fig.add_trace(graph_line)

    for pos in action_segments_x[1:-1]:
        fig.add_shape(type="line", x0=pos, x1=pos, y0=bottom_row_down, y1=top_row_up, line=dict(color="rgb(255,0,0)", dash="dot", width=1), showlegend=False)

    fig.update_yaxes(range=[bottom_row_down-100, top_row_up+350], ticksuffix="", showticklabels=True)
    fig.update_xaxes(ticksuffix="s", showticklabels=True)
    fig.update_layout(autosize=True, title={"text": "Recorded signals and classes"}, xaxis={"title": "Time (s)"})

    return fig



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simple server configuration parser.",
        usage="%(prog)s --host_ip <IP> --port <PORT> [--debug]"
    )

    parser.add_argument("--host_ip", type=lambda ip: ipaddress.ip_address(ip), default=ipaddress.ip_address("127.0.0.1"), help="IP address to listen to (e.g., 127.0.0.1 for local connections only, 192.168.1.1 specific interface or 0.0.0.0 for all).")
    parser.add_argument("--port", type=int, default=8050, help="Port number where to listen (integer, e.g., 8080).")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (default: False).")
    args = parser.parse_args()

    ipaddress.ip_address("127.0.0.1")

    host_ip = str(args.host_ip)
    port = args.port
    debug = args.debug

    app.run(debug=debug, port=port, host=host_ip, dev_tools_hot_reload=False, dev_tools_serve_dev_bundles=debug)