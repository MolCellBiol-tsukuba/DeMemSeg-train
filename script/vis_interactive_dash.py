
from dash import Dash, dcc, html, Input, Output, no_update
from typing import List, Optional
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import base64
import sys
import numpy as np
import os

# ===== CSV読み込み関数 =====
def load_and_concat_csvs(path_list):
    df_list = []
    for path in path_list:
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return pd.concat(df_list, ignore_index=True)

# ===== 引数からファイル読み込み =====
args = sys.argv[1:]
if len(args) == 0:
    raise ValueError("CSVファイルのパスを1つ以上指定してください。")

# リストとして受け取り
data_paths = args
df = load_and_concat_csvs(data_paths)

# ===== WTを別ファイルから追加（任意） =====
wt_path = "../data/mmdet_results_PSM_exps_CorrectAnnotation_100percent_mask-rcnn_r50_fpn_2x_coco_epoch_24.csv"
if os.path.exists(wt_path):
    df_wt = pd.read_csv(wt_path, low_memory=False)
    df_wt.columns = df_wt.columns.str.strip()
    df = pd.concat([df, df_wt], ignore_index=True)

# ===== Roundness補正 =====
df["Roundness"] = np.where(df["Roundness"] > 1, 2 - df["Roundness"], df["Roundness"])

# ===== 可視化対象のStrain =====
strain_wt = "YSIY874"
strains = sorted(df["StrainName"].dropna().unique())
strains = [s for s in strains if s != strain_wt]

# ===== 図の作成関数 =====
def create_scatter_plot(sub_df, strain, color):
    return go.Scatter(
        x=sub_df["Perimeter"],
        y=sub_df["Roundness"],
        mode='markers',
        marker=dict(color=color, size=4, opacity=0.7),
        name=strain,
        hoverinfo="skip"
    )

cols = 2
rows = int(np.ceil(len(strains) / cols))
fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=[f"{s} vs {strain_wt}" for s in strains])

for idx, strain in enumerate(strains):
    row = idx // cols + 1
    col = idx % cols + 1

    df_wt = df[df["StrainName"] == strain_wt]
    df_target = df[df["StrainName"] == strain]
    max_points = 1000
    if len(df_wt) > max_points:
        df_wt = df_wt.sample(n=min(max_points, len(df_wt)), random_state=42) if len(df_wt) > 10000 else df_wt
        df_target = df_target.sample(n=min(max_points, len(df_target)), random_state=42) if len(df_target) > 10000 else df_target

    fig.add_trace(create_scatter_plot(df_wt, strain_wt, "black"), row=row, col=col)
    fig.add_trace(create_scatter_plot(df_target, strain, f"rgba({50+idx*40}, {100+idx*10}, 150, 0.9)"), row=row, col=col)

    fig.update_xaxes(title_text="Perimeter", range=[0, 450], row=row, col=col)
    fig.update_yaxes(title_text="Roundness", range=[0, 1], row=row, col=col)

fig.update_layout(height=300*rows, width=900, showlegend=False, title="Interactive WT vs Strain Comparison")

# ===== Dash App構築 =====
app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="scatter-plot", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="hover-tooltip")
])

@app.callback(
    Output("hover-tooltip", "show"),
    Output("hover-tooltip", "bbox"),
    Output("hover-tooltip", "children"),
    Input("scatter-plot", "clickData"),
)
def display_click(clickData):
    if clickData is None or not clickData["points"]:
        return False, no_update, no_update

    pt = clickData["points"][0]
    bbox = pt["bbox"]
    pt_x, pt_y = pt["x"], pt["y"]

    df_row = df[(df["Perimeter"] == pt_x) & (df["Roundness"] == pt_y)]
    if df_row.empty:
        return False, no_update, no_update

    row = df_row.iloc[0]
    encoded_image = encode_image(row["ImagePath"])
    encoded_mask = encode_image(row["ImageInfPath"])

    children = html.Div([
        html.Div([
            html.Img(src=f"data:image/png;base64,{encoded_image}", style={"width": "45%", "margin-right": "10px"}),
            html.Img(src=f"data:image/png;base64,{encoded_mask}", style={"width": "45%"})
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        html.H6(
            f"strain: {row['StrainName']}\nTimepoint: {row['TimeMin']}\nfile_name: {row['ImageName']}",
            style={"color": "darkblue", "overflow-wrap": "break-word", "white-space": "pre-wrap"}
        )
    ], style={'width': '300px', 'white-space': 'normal'})

    return True, bbox, children

def encode_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode()
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

if __name__ == "__main__":
    app.run_server(debug=True, port=8053)
