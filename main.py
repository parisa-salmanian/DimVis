import dash
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output
import umap
from sklearn.tree import DecisionTreeClassifier, plot_tree
import io
import base64
import numpy as np
import shap
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret.data import ClassHistogram
import json
import pickle
import os
import flask
from flask_session import Session
import uuid
import time
import json


server = flask.Flask(__name__)
server.secret_key = os.urandom(24)
server.config['SESSION_TYPE'] = 'filesystem'

Session(server)

app = dash.Dash(__name__, server=server, external_stylesheets=[
                dbc.themes.BOOTSTRAP, 'assets/css/style.css'], suppress_callback_exceptions=True)

# Load datasets

breast_cancer_df = pd.read_csv(
    r'C:/Users/Parisa/Desktop/EuroVis Paper Code/Datasets/breast_cancer.csv', header=None)
breast_cancer_df.columns = ['id', 'class', 'clumpthic', 'sizeun', 'shapeun',
                            'marg_adh', 'epithsize', 'barenuc', 'blandchr', 'nornuc', 'mitoses']


indian_diabetes_df = pd.read_csv(
    r'C:/Users/Parisa/Desktop/EuroVis Paper Code/Datasets/indian_diabetes.csv', header=None)
indian_diabetes_df.columns = ['id', 'class', 'Pregnan', 'Glucose',
                              'BloodPr', 'SkinThick', 'Insulin', 'BMI', 'DPF', 'Age']


# Define layout
app.layout = html.Div([
    html.Header(className='navbar navbar-secondary sticky-top nav-bg flex-md-nowrap p-2 shadow',
                children=[
                    html.A(
                        className='navbar-brand ms-4',
                        href='#',
                        children=[
                            html.Img(src="assets/images/logos/lnu_logo.png",
                                     width="50", className="rounded")
                        ]
                    ),
                    html.Ul(
                        
                        className="navbar-nav",
                        style={'width': 'max-content'}, 
                        children=[
                            html.Li(
                                className="nav-item ms-4 pe-4 me-4 d-inline-block",
                                children=[
                                    dcc.Dropdown(
                                        id='dataset-dropdown',
                                        options=[
                                            {'label': 'Breast Cancer',
                                                'value': 'breast_cancer'},
                                            {'label': 'Indian Diabetes',
                                                'value': 'indian_diabetes'},
                                        ],
                                        value='breast_cancer',
                                        placeholder='Select a dataset',
                                        className='nav-link text-dark',
                                        style={'display': 'inline-block',
                                               'width': '250px'}
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Ul(
                       
                        className="navbar-nav",
                        style={'min-width': '400px'}, 
                        children=[
                            html.Li(
                                className="nav-item pe-4 me-4 d-inline-block",
                                children=[
                                    html.Label(
                                        "Neighbors:", id='n_neighbors-label', className='pr-2'),
                                    dcc.Slider(
                                        id='n_neighbors-slider',
                                        min=3,
                                        max=100,
                                        step=1,
                                        value=15
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Ul(
                        
                        className="navbar-nav",
                        style={'min-width': '400px'},  
                        children=[
                            html.Li(
                                className="nav-item pe-4 me-4 d-inline-block",
                                children=[
                                    html.Label(
                                        "Min Dist:", id='min_dist-label', className='pr-2'),
                                    dcc.Slider(
                                        id='min_dist-slider',
                                        min=0,
                                        max=1,
                                        step=0.1,
                                        value=0.1
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Button(
                        className='navbar-toggler position-absolute d-md-none collapsed',
                        type='button',
                        **{
                            'data-bs-toggle': 'collapse',
                            'data-bs-target': '#sidebarMenu',
                            'aria-controls': 'sidebarMenu',
                            'aria-expanded': 'false',
                            'aria-label': 'Toggle navigation',
                            'children': html.Span(className='navbar-toggler-icon')
                        }
                    ),
                    dbc.Button(id="precision-result", color="info",
                               className="text-light me-4", children="Precision: Not calculated yet"),
                    dbc.Button(id="recall-result", color="warning",
                               className="text-light", children="Recall: Not calculated yet"),
                ]
                ),

    html.Div(className='container-fluid', children=[
        html.Div(className='row', children=[

            html.Main(className='col-12', children=[
                html.Div(className="row w-100", children=[
                    html.Div(className="show", id="loading-spinner", children=[
                        dbc.Spinner(color="#ffe000", type="grow",
                                    fullscreen=True, show_initially=True),
                    ]
                    ),
                    html.Div(className="col-12 col-md-6 mb-0 height-amount", children=[
                        dcc.Graph(id='scatter-plot', className="collapse"),
                    ]),
                    html.Div(className="col-12 col-md-6 mb-0 height-amount", children=[
                        dcc.Graph(id='ebm-plot', className="collapse"),
                        dcc.Store(id='feature-names'),
                        dcc.Store(id='feature-importance'),
                        dcc.Store(id='ebm-model-explainer'),
                    ]),
                    html.Div(className="col-12 mx-auto mt-0", children=[
                        dcc.Graph(id='ebm-feature-score-plot',
                                  className="collapse")
                    ]),
                    dcc.Store(id='prev-dataset'),
                    dcc.Store(id='last-clicked-feature', data=None)
                ])
            ])
        ])
    ]),
])


@app.callback(
    Output(component_id='loading-spinner', component_property='className'),
    [
        Input(component_id='dataset-dropdown', component_property='value'),
        Input(component_id='n_neighbors-slider', component_property='value'),
        Input(component_id='min_dist-slider', component_property='value'),
        Input('scatter-plot', 'selectedData')
    ],
    prevent_initial_call='initial_duplicate'

)
def show_spinner(input1, input2, input3, input4):
    return "show"


@app.callback(
    Output('n_neighbors-label', 'children'),
    [Input('n_neighbors-slider', 'value')]
)
def update_neighbors_label(value):
    return f"Neighbors: {value}"


@app.callback(
    Output('min_dist-label', 'children'),
    [Input('min_dist-slider', 'value')]
)
def update_mindist_label(value):
    return f"Min Dist: {value}"


# define the callback function for visualizing the datasets


@app.callback(
    [
        Output(component_id='scatter-plot', component_property='figure'),
        Output(component_id='scatter-plot', component_property='className'),
        Output(component_id='ebm-plot', component_property='className'),
        Output(component_id='loading-spinner',
               component_property='className', allow_duplicate=True),
    ],
    [
        Input(component_id='dataset-dropdown', component_property='value'),
        Input(component_id='n_neighbors-slider', component_property='value'),
        Input(component_id='min_dist-slider', component_property='value'),
    ],
    prevent_initial_call='initial_duplicate'
)
def update_scatter_plot(dataset, n_neighbors, min_dist):
    if dataset is None:
        raise PreventUpdate

    umap_status = False  # Initialize umap_status as False
    fig_title = ''

    if dataset == 'breast_cancer':
        data_df = breast_cancer_df.copy()
        fig_title = 'Breast Cancer Dataset'
    else:
        data_df = indian_diabetes_df.copy()
        fig_title = 'Indian Diabetes Dataset'

    # Prepare the hyper_parameters dictionary using the slider values
    hyper_parameters = {'n_neighbors': n_neighbors, 'min_dist': min_dist, 'random_state':42}

    # Check if the same UMAP has been previously applied to the same dataset
    if umap_status and umap_status['dataset'] == dataset and umap_status['hyperparameters'] == hyper_parameters:
        # Use the stored UMAP data
        data_df = pd.read_json(umap_status['umap_data'])
    else:
        # Apply UMAP and store the result
        reducer = umap.UMAP(**hyper_parameters)
        embeddings = reducer.fit_transform(
            data_df.drop(['class', 'id'], axis=1))
        data_df['umap_1'] = embeddings[:, 0]
        data_df['umap_2'] = embeddings[:, 1]
        umap_status = {
            'dataset': dataset,
            'hyperparameters': hyper_parameters,
            'umap_data': data_df.to_json()
        }

        if dataset == 'breast_cancer':
            color_sequence = ["gray"]
        else:
            color_sequence = ["red", "blue"]


        fig = px.scatter(
            data_df,
            x='umap_1',
            y='umap_2',
            color='class',
            color_discrete_sequence=color_sequence,
            title='UMAP Peojection - ' + fig_title,
            hover_data=['id'],  # Add 'id' column to hover data
        )
        fig.update_xaxes(title_text='', scaleanchor='y', scaleratio=1, showticklabels=False)  
        fig.update_yaxes(title_text='', scaleanchor='x', scaleratio=1, showticklabels=False) 
        fig.update_traces(text=data_df['id'], textposition='top center')
        fig.update_layout(title={
            'text': 'UMAP - ' + fig_title,
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'right',
            'yanchor': 'top',
            'font': dict(
                size=16,  
                family="Courier New, monospace",  
                color="RebeccaPurple"  
            )
        },
            legend=dict(x=0, y=-0.05, xanchor='left',
                        yanchor='top', orientation='h'),
            autosize=False,
            width=650,
            height=650,
            margin=dict(
            autoexpand=False,
            l=120,
            r=0,
            t=30,
            b=150,
        ),)


    return fig, "show" if umap_status else "", "show", "collapse"


@app.callback(
    [
        Output(component_id='ebm-plot', component_property='figure'),
        Output(component_id='precision-result',
               component_property='className'),
        Output(component_id='precision-result', component_property='children'),
        Output(component_id='recall-result', component_property='className'),
        Output(component_id='recall-result', component_property='children'),
        Output(component_id='feature-names', component_property='data'),
        Output(component_id='feature-importance', component_property='data'),
        Output(component_id='ebm-model-explainer', component_property='data'),
        Output(component_id='loading-spinner',
               component_property='className', allow_duplicate=True),
        Output(component_id='ebm-feature-score-plot', component_property='style'),


    ],
    [
        Input(component_id='scatter-plot', component_property='selectedData'),
        Input(component_id='dataset-dropdown', component_property='value'),

    ],
    [State(component_id='scatter-plot', component_property='selectedData')],
    prevent_initial_call='initial_duplicate'
)
def update_ebm_plot(selectedData, dataset, state_selectedData):

    if dataset == 'breast_cancer':
        data_df = breast_cancer_df.copy()
    else:
        data_df = indian_diabetes_df.copy()

    # Here, we assume that the second column is the class column
    class_column = data_df.columns[1]

    # Check if there's any selected data. If not, use the entire dataset.
    if not state_selectedData or 'points' not in selectedData:
        y = data_df[class_column]
    else:
        selected_points = selectedData['points']
        id_vals = [point['customdata'] for point in selected_points]
        id_vals = [point[0] for point in id_vals]

        # If all data points are selected, use the original classes
        if len(id_vals) == len(data_df):
            y = data_df[class_column]
        else:
            data_df['class'] = data_df['id'].apply(
                lambda x: 'selected' if x in id_vals else 'unselected')
            y = data_df['class']

    X = data_df.drop(['class', 'id', class_column], axis=1)

    model = ExplainableBoostingClassifier(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    if selectedData is None or len(id_vals) == len(data_df):
        precision = "{:.2f}".format(precision_score(
            y, y_pred, average='weighted'))
        recall = "{:.2f}".format(recall_score(
            y, y_pred, average='weighted'))
    else:
        precision = "{:.2f}".format(precision_score(
            y, y_pred, pos_label="selected"))
        recall = "{:.2f}".format(recall_score(
            y, y_pred, pos_label="selected"))

    explainer = model.explain_global()

    session_id = str(uuid.uuid4())
    flask.session['session_id'] = session_id

    if not os.path.exists('explainer_instances'):
        os.makedirs('explainer_instances')

    explainer_filename = os.path.join(
        "explainer_instances", f"explainer_{session_id}.p")
    pickle.dump(explainer, open(explainer_filename, "wb"))

    # Extract feature importance values and names
    feature_importance = explainer.data(0)['scores']
    feature_names = explainer.data(0)['names']

    if isinstance(feature_importance[0], np.ndarray):
        # Flatten the arrays to scalars
        feature_importance_avg = [
            np.mean(fi).item() for fi in feature_importance]
    else:
        feature_importance_avg = feature_importance

    # Sort feature importance and names in descending order
    feature_importance, feature_names = zip(
        *sorted(zip(feature_importance_avg, feature_names), reverse=True))

    # Get the original explainer figure
    fig = explainer.visualize()

    fig.update_layout(title={
        'y': 0.99,
        'x': 0.5,
        'xanchor': 'right',
        'yanchor': 'top',
        'font': dict(
            size=16,  
            family="Courier New, monospace",  
            color="RebeccaPurple"  
        )
    },
        legend=dict(x=0, y=-0.1, xanchor='left',
                    yanchor='top', orientation='h'),
        autosize=False,
        width=650,
        height=650,
        margin=dict(
        autoexpand=False,
        l=120,
        r=0,
        t=30,
        b=150,
    ),)

    return fig.to_dict(), "show text-light me-4", "Precision: " + str(precision), "show text-light me-4", "Recall: " + str(recall), explainer.feature_names, feature_importance, explainer_filename, "collapse", {'display': 'none'}

@app.callback(
    [
        Output(component_id='ebm-feature-score-plot',
               component_property='figure'),
        Output(component_id='ebm-feature-score-plot',
               component_property='className'),
        Output(component_id='ebm-feature-score-plot', component_property='style', allow_duplicate=True),
        Output(component_id='last-clicked-feature', component_property='data')  

    ],
    [
        Input(component_id='ebm-plot', component_property='clickData')
    ],
    [
        State(component_id='feature-names', component_property='data'),
        State(component_id='ebm-model-explainer', component_property='data'),
        State(component_id='ebm-feature-score-plot', component_property='style'),
        State(component_id='last-clicked-feature', component_property='data')
    ],
    prevent_initial_call='initial_duplicate'
)
def plot_ebm_feature_score(clickData, feature_names, explainer_filename, plot_style, last_clicked_feature):
    if clickData is not None and feature_names is not None:
        feature_name = clickData['points'][0]['y']

        if feature_name == last_clicked_feature:
            # Toggle visibility if the same feature is clicked
            new_style = {'display': 'block'} if plot_style.get('display', '') == 'none' else {'display': 'none'}
            return dash.no_update, "show", new_style, feature_name

        explainer = pickle.load(open(explainer_filename, "rb"))
        feature_index = feature_names.index(feature_name)
        fig = explainer.visualize(feature_index)

        # Always show the plot but update its content
        return fig, "show", {'display': 'block'}, feature_name

    else:
        return dash.no_update, "collapse", dash.no_update, last_clicked_feature




# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
