from dash.dependencies import Input, Output, State
from statsmodels.tsa.arima.model import ARIMA
from dtools.dataprocess import DataProcessor
from dash.exceptions import PreventUpdate
from dtools.functions import *
from dtools.walks import *
from dash import dcc, html, dash_table
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
        value=3,
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
        value=3,
        marks={
            0: {'label': '0'},
            5: {'label': '5'},
            10: {'label': '10'}
        },
        tooltip = { 'always_visible': True }
    ),
    html.Hr(),
    html.H3("Upload Test Data"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    html.Div(id='processed-data-upload'),
    html.Hr(),
    html.Div([
        dcc.Graph(id='short-term-plot', style={'display': 'inline-block', 'width': '49%'}),
        dcc.Graph(id='long-term-plot', style={'display': 'inline-block', 'width': '49%'})
    ])
])

              
@app.callback(Output('short-term-plot', 'figure'),
                Input('short-term-model', 'value'),
                Input('num-change-points', 'value'),
                Input('upload-data', 'contents'),
                State('upload-data', 'filename'))
                # State('upload-data', 'contents'))
def update_short_term_plot(short_model_type, num_change_points, list_of_contents, list_of_names):
    if list_of_contents is not None:
                # Parse the contents of the file
        children = [parse_contents(c, n) for c,n in zip(list_of_contents, list_of_names)]

        processed_data = process_data(children[-1])

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

        chng_dfs, chng_xs = get_change_dfs(df, data["model"], num_change_points)

        processed_dfs = [processed_data[(chng_xs[i][0] <= processed_data["week"]) & (processed_data["week"] <= chng_xs[i][-1])] for i in range(len(chng_xs))]
        processed_dfs_avgs = [df.week_linesperauthor.mean() for df in processed_dfs]

        chng_dfs_avgs = [df.model.mean() for df in chng_dfs]

        fig = go.Figure()
        for i in range(num_change_points):
            col = "rgba(" + ", ".join([str(l) for l in RGB_tuples[i]]) + ", 0.2)"
            fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, fillcolor=col, fill='tozeroy',mode='none', name=f"")) # override default markers+lines))
            fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, mode='lines',opacity=1,line=dict(color=col, width=1.5), name=""))
            fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=[chng_dfs_avgs[i] for _ in range(len(chng_dfs[i].time))], name=f"Phase {i+1} Model Avg"))
            fig.add_trace(go.Scatter(x=processed_dfs[i].week, y=[processed_dfs_avgs[i] for _ in range(len(processed_dfs[i].week))], name=f"Phase {i+1} Test Data Avg"))
            
            fig.add_trace(go.Scatter(x=[chng_dfs[i].time.median() for _ in range(len(chng_dfs[i].time))], y=[processed_dfs_avgs[i] for _ in range(int(len(processed_dfs[i].week)/2))] + [chng_dfs_avgs[i] for _ in range(int(len(chng_dfs[i].time)/2))], fill="toself", name=f""))
            
            xloc = chng_dfs[i].time.median()
            smaller = min([processed_dfs_avgs[i], chng_dfs_avgs[i]])
            bigger = max([processed_dfs_avgs[i], chng_dfs_avgs[i]])
            yloc = smaller + (bigger - smaller)/2
            plus_minus = "+" if (chng_dfs_avgs[i] == smaller) else "-"

            increase = (bigger-smaller)/smaller
            decrease = (bigger-smaller)/bigger
            diff = increase if (chng_dfs_avgs[i] == smaller) else decrease
            text = "PCT Change" + plus_minus + str(round(diff*100, 2)) + "%"
            fig.add_annotation(
                x=xloc, y=yloc,
                text=text,
                showarrow=False
            )
        fig.update_layout(
            title='Short-term Model: {}'.format(short_model_type),
            xaxis_title='Weeks',
            yaxis_title='Lines Per Author',
        )
        for trace in fig["data"]:
            if trace["name"].split(" ")[0] != "Phase": trace["showlegend"] = False
        return fig
    else:
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

        chng_dfs, chng_xs = get_change_dfs(df, data["model"], num_change_points)

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
                Input('num-change-points-long', 'value'),
                Input('upload-data', 'contents'),
                State('upload-data', 'filename'))
def update_long_term_plot(long_model_type, num_change_points_long, list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [parse_contents(c, n) for c,n in zip(list_of_contents, list_of_names)]

        # # Process the data
        processed_data = process_data(children[-1])
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

        chng_dfs, chng_xs = get_change_dfs(df, data["model"], num_change_points_long)

        processed_dfs = [processed_data[(chng_xs[i][0] <= processed_data["week"]) & (processed_data["week"] <= chng_xs[i][-1])] for i in range(len(chng_xs))]
        processed_dfs_avgs = [df.week_linesperauthor.mean() for df in processed_dfs]

        chng_dfs_avgs = [df.model.mean() for df in chng_dfs]

        fig = go.Figure()
        for i in range(num_change_points_long):
            col = "rgba(" + ", ".join([str(l) for l in RGB_tuples[i]]) + ", 0.2)"
            fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, fillcolor=col, fill='tozeroy',mode='none', name=f"")) # override default markers+lines))
            fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=chng_dfs[i].model, mode='lines',opacity=1,line=dict(color=col, width=1.5), name=""))
            fig.add_trace(go.Scatter(x=chng_dfs[i].time, y=[chng_dfs_avgs[i] for _ in range(len(chng_dfs[i].time))], name=f"Phase {i+1} Model Avg"))
            fig.add_trace(go.Scatter(x=processed_dfs[i].week, y=[processed_dfs_avgs[i] for _ in range(len(processed_dfs[i].week))], name=f"Phase {i+1} Test Data Avg"))
            
            fig.add_trace(go.Scatter(x=[chng_dfs[i].time.median() for _ in range(len(chng_dfs[i].time))], y=[processed_dfs_avgs[i] for _ in range(int(len(processed_dfs[i].week)/2))] + [chng_dfs_avgs[i] for _ in range(int(len(chng_dfs[i].time)/2))], fill="toself", name=f""))
            
            xloc = chng_dfs[i].time.median()
            smaller = min([processed_dfs_avgs[i], chng_dfs_avgs[i]])
            bigger = max([processed_dfs_avgs[i], chng_dfs_avgs[i]])
            yloc = smaller + (bigger - smaller)/2
            plus_minus = "+" if (chng_dfs_avgs[i] == smaller) else "-"

            increase = (bigger-smaller)/smaller
            decrease = (bigger-smaller)/bigger
            diff = increase if (chng_dfs_avgs[i] == smaller) else decrease
            text = "PCT Change" + plus_minus + str(round(diff*100, 2)) + "%"
            fig.add_annotation(
                x=xloc, y=yloc,
                text=text,
                showarrow=False
            )
        fig.update_layout(
            title='Long-term Model: {}'.format(long_model_type),
            xaxis_title='Weeks',
            yaxis_title='Lines Per Author',
        )

        for trace in fig["data"]:
            if trace["name"].split(" ")[0] != "Phase": trace["showlegend"] = False
        return fig
    else:
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

        chng_dfs, chng_xs = get_change_dfs(df, data["model"], num_change_points_long)

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


def parse_contents(contents, filename):

    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    
    try:
        decoded = io.StringIO(decoded.decode('utf-8'))
        df = pd.read_json(decoded)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Add a column with the filename
    df['filename'] = filename

    return df

def process_data(df):
    clean_data = clean_dataframe(df)
    processed_data = filter_dataframe(clean_data, "java")
    lpa_data = lines_per_author(processed_data, timeframe="week")

    return lpa_data

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    
    if list_of_contents is not None:
        # Parse the contents of the file
        children = [parse_contents(c, n) for c,n in zip(list_of_contents, list_of_names)]

        # # Process the data
        processed_data = process_data(children[-1])

        return f"Loaded {list_of_names[-1]} successfully!"


    return html.Div(['No data has been uploaded yet.'])
    

def get_change_dfs(df, model, n):
    change_pts = detect_change_pts(model, n)
    change_pts.insert(0, 0)
    xs = [[j for j in range(change_pts[i], change_pts[i+1])] for i in range(len(change_pts)-1)]
    dfs = [df[(xs[i][0] <= df["time"]) & (df["time"] <= xs[i][-1])] for i in range(len(xs))]
    return dfs, xs


if __name__ == '__main__':
    app.run_server(debug=True)