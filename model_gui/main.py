from dash.dependencies import Input, Output, State
from statsmodels.tsa.arima.model import ARIMA
from dtools.dataprocess import DataProcessor
from dash.exceptions import PreventUpdate
from dtools.functions import *
from dtools.walks import *
from dash import dcc
from dash import html
import plotly.graph_objects as go
import plotly.express as px
import plotly.express as px
import pandas as pd
import numpy as np
import colorsys
import base64
import dash
import io


full_rw_model = load_project_variable(os.path.join(os.getcwd(), "data\model"), project_type="full", variable_type="rw_model")
full_arima_model = load_project_variable(os.path.join(os.getcwd(), "data\model"), project_type="full", variable_type="arima_model")

# Define the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Model Comparison Dashboard"),
    html.H2("Short-term Models"),
    dcc.Dropdown(
        id='short-term-model',
        options=[
            {'label': 'ARIMA', 'value': 'arima'},
            {'label': 'Random Walk', 'value': 'random_walk'}
        ],
        value='arima'
    ),
    dcc.Slider(
        id='num-change-points',
        min=0,
        max=10,
        step=1,
        value=4,
        marks={
            0: {'label': '0'},
            5: {'label': '5'},
            10: {'label': '10'}
        },
        tooltip = { 'always_visible': True }
    ),
    html.H2("Long-term Models"),
    dcc.Dropdown(
        id='long-term-model',
        options=[
            {'label': 'ARIMA', 'value': 'arima'},
            {'label': 'Random Walk', 'value': 'random_walk'}
        ],
        value='arima'
    ),
    dcc.Slider(
        id='num-change-points-long',
        min=0,
        max=10,
        step=1,
        value=4,
        marks={
            0: {'label': '0'},
            5: {'label': '5'},
            10: {'label': '10'}
        },
        tooltip = { 'always_visible': True }
    ),
    html.Hr(),
    # html.H3("Upload Test Data"),
    # dcc.Upload(
    #     id='upload-data',
    #     children=html.Div([
    #         'Drag and Drop or ',
    #         html.A('Select Files')
    #     ]),
    #     style={
    #         'width': '50%',
    #         'height': '60px',
    #         'lineHeight': '60px',
    #         'borderWidth': '1px',
    #         'borderStyle': 'dashed',
    #         'borderRadius': '5px',
    #         'textAlign': 'center',
    #         'margin': '10px'
    #     },
    #     multiple=False
    # ),
    # html.Div(id='output-data-upload'),
    # html.Button('Calculate MAPE', id='calc-mape', n_clicks=0),
    html.Hr(),
    html.Div([
        dcc.Graph(id='short-term-plot', style={'display': 'inline-block', 'width': '49%'}),
        dcc.Graph(id='long-term-plot', style={'display': 'inline-block', 'width': '49%'})
    ])
])

# # Define the callback to calculate the MAPE
# @app.callback(Output('calc-mape', 'children'),
#               Input('upload-data', 'contents'),
#               State('upload-data', 'filename'),
#               State('short-term-model', 'value'),
#               State('num-change-points', 'value'))
# def calculate_mape(contents, filename, short_model_type, num_change_points):
#     if contents is not None:
#         # df = pd.read_json(contents, orient='split')
#         # Get the predictions
#         if short_model_type == 'arima':
#             model = ARIMA(df, order=(1, 1, 0))
#             predictions = model.fit().predict()
#         else:
#             predictions = np.cumsum(np.random.randn(len(df)))
#         # Calculate the MAPE
#         test_data = df[num_change_points:]
#         pred_data = predictions[num_change_points:]
#         mape = np.mean(np.abs((test_data - pred_data) / test_data)) * 100
#         return html.Div([
#         html.H4("Mean Absolute Percentage Error: {:.2f}%".format(mape))
#         ])
#     else:
#         return html.Div('Please upload a file.')
    
# import json
# @app.callback(
#     Output("output-data-upload", "children"),
#     Input("upload-data", "contents"),
#     State("upload-data", "filename"),
# )
# def on_data_upload(contents, filename):
#     if contents is None:
#         raise PreventUpdate

#     if not filename.endswith(".json"):
#         return "Please upload a file with the .json extension"
#     # content_type, content_string = contents[0].split(',')
#     jsonData = json.loads(contents)
#     print(jsonData)
#     decoded = base64.b64decode(contents)
#     df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#     # content_type, content_string = contents.split(",")
#     # decoded = base64.b64decode(content_string)
    
#     # # this is a dict!
#     # content_dict = json.loads(decoded)
    
#     # df = pd.read_json(contents)
#     lpa_df = lines_per_author(df)

#     return dcc.Graph(figure=lpa_df)


@app.callback(Output('short-term-plot', 'figure'),
                Input('short-term-model', 'value'),
                Input('num-change-points', 'value'))
                # State('upload-data', 'contents'))
def update_short_term_plot(short_model_type, num_change_points):

    if short_model_type == 'arima':
        data = full_arima_model["short_model"]
        data["model"] = data["model"][1:]
        data["time"] = data["time"][1:]
    else:
        data = full_rw_model["short_model"]
    # Create the plot
    HSV_tuples = [(x*1.0/num_change_points, 0.5, 0.5) for x in range(num_change_points)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    df = pd.DataFrame(dict(time=data["time"], model=data["model"]))

    chng_dfs = get_change_dfs(df, data["model"], num_change_points)

    fig = go.Figure()
    for i in range(num_change_points):
        col = "rgba(" + ", ".join([str(l) for l in RGB_tuples[i]]) + ", 0.8)"
        fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, fillcolor=col, fill='tozeroy',mode='none', name=f"Phase {i+1}")) # override default markers+lines))
        fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, mode='lines',opacity=1,line=dict(color=col, width=1.5), name=""))


    fig.update_layout(
        title='Short-term Model: {}'.format(short_model_type),
        xaxis_title='Weeks',
        yaxis_title='Lines Per Author',
    )
    for trace in fig["data"]:
        if trace["name"].split(" ")[0] != "Phase": trace["showlegend"] = False
    return fig
    

@app.callback(Output('long-term-plot', 'figure'),
                Input('long-term-model', 'value'),
                Input('num-change-points-long', 'value'))
                # State('upload-data', 'contents'))
def update_long_term_plot(long_model_type, num_change_points_long):
    if long_model_type == 'arima':
        data = full_arima_model["long_model"]
        data["model"] = data["model"][1:]
        data["time"] = data["time"][1:]
    else:
        data = full_rw_model["long_model"]
    # Create the plot
    HSV_tuples = [(x*1.0/num_change_points_long, 0.5, 0.5) for x in range(num_change_points_long)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    df = pd.DataFrame(dict(time=data["time"], model=data["model"]))

    chng_dfs = get_change_dfs(df, data["model"], num_change_points_long)

    fig = go.Figure()
    for i in range(num_change_points_long):
        col = "rgba(" + ", ".join([str(l) for l in RGB_tuples[i]]) + ", 0.8)"
        fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, fillcolor=col, fill='tozeroy',mode='none', name=f"Phase {i+1}")) # override default markers+lines))
        fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, mode='lines',opacity=1,line=dict(color=col, width=1.5), name=""))

    fig.update_layout(
        title='Long-term Model: {}'.format(long_model_type),
        xaxis_title='Weeks',
        yaxis_title='Lines Per Author',
    )

    for trace in fig["data"]:
        if trace["name"].split(" ")[0] != "Phase": trace["showlegend"] = False
    return fig
    

def get_change_dfs(df, model, n):
    change_pts = detect_change_pts(model, n)
    change_pts.insert(0, 0)
    xs = [[j for j in range(change_pts[i], change_pts[i+1])] for i in range(len(change_pts)-1)]
    dfs = [df[(xs[i][0] <= df["time"]) & (df["time"] <= xs[i][-1])] for i in range(len(xs))]
    return dfs
if __name__ == '__main__':
    app.run_server(debug=True)