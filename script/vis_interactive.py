from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import base64
import sys

args = sys.argv
data_path = args[1]
df = pd.read_csv(data_path)

def create_scatter_plot(strain, color, hoverinfo):
    df_strain = df[df['StrainName'] == strain]
    return go.Scatter(
        x=df_strain["Perimeter"],
        y=df_strain["Roundness"],
        mode='markers',
        marker=dict(color=color),
        name=strain,
        hoverinfo=hoverinfo
    )

fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("wt", "spr28", "spr3", "spr28spr3"),
                       vertical_spacing=0.1, horizontal_spacing=0.1)

fig.add_trace(create_scatter_plot('YSIY1682', "#57A6A1", "none"), row=1, col=1)
fig.add_trace(create_scatter_plot('YSIY1682', "gray", "skip"), row=1, col=2)
fig.add_trace(create_scatter_plot('YSIY1683', "#577B8D", "none"), row=1, col=2)
fig.add_trace(create_scatter_plot('YSIY1682', "gray", "skip"), row=2, col=1)
fig.add_trace(create_scatter_plot('YSIY1684', "#344C64", "none"), row=2, col=1)
fig.add_trace(create_scatter_plot('YSIY1682', "gray", "skip"), row=2, col=2)
fig.add_trace(create_scatter_plot('YSIY1685', "#240750", "none"), row=2, col=2)

x_range = [0, 300]
y_range = [0, 1.30]
x_title = "Perimeter (μm)"
y_title = "Roundness"
fig.update_layout(
    height=1200,
    width=1500,
    xaxis=dict(title=x_title, range=x_range),
    yaxis=dict(title=y_title, range=y_range),
    xaxis2=dict(title=x_title, range=x_range),
    yaxis2=dict(title=y_title, range=y_range),
    xaxis3=dict(title=x_title, range=x_range),
    yaxis3=dict(title=y_title, range=y_range),
    xaxis4=dict(title=x_title, range=x_range),
    yaxis4=dict(title=y_title, range=y_range)
)

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])

@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "clickData"),
)
def display_click(clickData):
    if clickData is None or not clickData["points"]:
        return False, no_update, no_update

    pt = clickData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    pt_x = pt["x"]
    pt_y = pt["y"]

    df_row = df[(df["Perimeter"] == pt_x) & (df["Roundness"] == pt_y)].iloc[0]
    img_path = df_row['ImagePath']
    mask_path = df_row['ImageInfPath']
    timepoint = df_row['TimeMin']
    strain = df_row['StrainName']
    file_name = df_row['ImageName']

    encoded_image = encode_image(img_path)
    encoded_mask = encode_image(mask_path)

    children = [
        html.Div([
            html.Div([
                html.Img(src=f'data:image/png;base64,{encoded_image}', style={"width": "45%", "margin-right": "10px"}),
                html.Img(src=f'data:image/png;base64,{encoded_mask}', style={"width": "45%"}),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.H6(
                f"strain: {strain}\nTimepoint: {timepoint}\nfile_name: {file_name}",
                style={
                    "color": "darkblue",
                    "overflow-wrap": "break-word",
                    "fontFamily": "Arial",
                    "white-space": "pre-wrap"
                }
            ),
        ], style={'width': '300px', 'white-space': 'normal'})
    ]

    return True, bbox, children

def encode_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode()
        return encoded_image
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8053)