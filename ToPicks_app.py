import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import base64
from gensim.models import Word2Vec
from joblib import dump, load

import json

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#######################################################################
# DEFINE ALL FUNCTIONS
#######################################################################

def user2topic(uinput):
    """
    uInput: User Input -  string; ngram=(1, 2)' 
    Function: 
        - transform user input with trained NMF model to match against the topic.
        - If topic not found in first pass in case the model has never seen the user input before,
          find synonyms with the trained word2vec model and then pass the synonyms through NMF.
    Output: Matched topic number. Default: 0 (That is, misc. category)
    """
    try:
        user_nmf = nmf.transform(vectorizer.transform([uinput]))[0]
        matched_topic = user_nmf.argmax()
        if matched_topic == 0: # try synonyms to user-input from word2vec 
            user2similar = w2v.wv.most_similar(positive=uinput,topn=10)
            related_topics = [x for x in list(sum(user2similar, ())) if str(x).isalpha()]
            user_nmf = nmf.transform(vectorizer.transform([" ".join(related_topics)]))[0]
            matched_topic = user_nmf.argmax()
        return matched_topic
    except KeyError:
        return 0


def process_related_topics(input_array):

    #######################################################################
    # QUERY AGAINST INTERNAL TOPICS WITH TRAINED NMF MODEL
    #######################################################################

    all_matched_topics=[]
    for uinput in input_array:
        try:
            matched_topic = user2topic(uinput)
            if matched_topic == 0:
                matched_topic = user2topic(uinput[:-1]) # try removing the last char ('s)
                if matched_topic == 0:
                    matched_topic = user2topic(uinput[:-2]) # try removing the last 2 chars ('es)
                    
            # sanity check for debugging:
            print(matched_topic, all_topics[str(matched_topic)])
            
        except KeyError:
            user2similar = ''
            pass
        
        all_matched_topics.append(matched_topic)

    #######################################################################
    # EXTRACT TOPIC DATA FOR FORECASTING
    #######################################################################

    utopics = {}
    for topic in all_matched_topics:
        utopics[topic] = pilot[pilot['predicted_topic'] == topic].set_index('date')\
                                                            ['likes_count'].resample('M').median()

    return utopics, all_matched_topics

#-----------------------------------------END OF FUNCTIONS--------------------------------------------#

#######################################################################
# LOAD PREVIOUSLY TRAINED MODELS AND DATA
#######################################################################

# word2vec model for synonyms:
w2v = Word2Vec.load("model/word2vec/w2v_bigram.model")
# NMF topic models for matching user input to topics:
nmf = load('model/nmf/latest_nmf_tfidf_ntop-200_nftr_50000_ngrams_(1, 3)_.joblib') 
vectorizer = load('model/nmf/latest_tfidf_tfidf_ntop-200_nftr_50000_ngrams_(1, 3)_.joblib')

# import data containing topic assignments from Step2:
tp_name = 'data/parquet_data/step2_NMF_topics=200_assigned.parquet'
pilot = pd.read_parquet(tp_name, engine='pyarrow')
all_topics_fname = 'data/json_data/all_topics_nmf_ntop-200_nftr_50000_ngrams_(1, 3).json'
with open(all_topics_fname, 'r') as fp:
    all_topics = json.load(fp)

# File with time-series analytics for that topic:
ts_name = 'data/ts_predictions/all_preds_ts_gb_hptuning=False_nmf_ntopics=198.parquet'
predictions = pd.read_parquet(ts_name, engine='pyarrow')


#######################################################################
# DASH APP
#######################################################################

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
main_img = base64.b64encode(open('./img/blocks.jpg', 'rb').read())


app.layout = html.Div([
#title:
    html.Div(html.H1('TO·P·icks', style = {'textAlign': 'center', 'padding': '2px', 'height': '20px', 'margin-top': '10px', 'fontSize':70, 'font-weight':'bold','color': '#D81111'})),
# subtitle:    
    html.Div(html.H3('Forecast Consumer-Interest in Topics', style = {'textAlign': 'center', 'height': '10px','color':'#07329C', 'fontSize':35, 'font-weight':'bold', 'margin-top': '60px'})),
# imagetitle:    
    html.Div(html.Img(id='head-image', src='data:image/jpeg;base64,{}'.format(main_img.decode('ascii')),
                      style = {'width':'100%', 'height': '500px', 'padding':'0','margin':'0','margin-top': '30px','box-sizing':'border-box'})),

# User Inputs:
    html.H4('Select topics of interest: '),    
    html.Div(title='select inputs', id='selections',children=[

# Text input:
    html.P('(Suggestions: Law, Tax, Climate change, Ebola, guns, Russia, Mexico, Technology, India etc.)'),
        dcc.Input(id='text_input1', type='text', placeholder='Enter first topic'),
        dcc.Input(id='text_input2', type='text', placeholder='Enter second topic'),
        dcc.Input(id='text_input3', type='text', placeholder='Enter third topic'),

# Hidden (clickable) dropdown menu with pre-configured options

    html.Br(),
    html.Br(),
    html.Details([
    html.Summary('Click to select topics from the predefined list:',style = {'fontSize':18, 'font-weight':'bold'}),
    html.Div([
        # dropdown:
    html.Div([
        dcc.Dropdown(
        id='dropdown_input1',
        options=[{'label': v, 'value': v} for v in all_topics.values()],
        placeholder='Select Topic# 1'),
        ],
        style={'width': '15%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
        id='dropdown_input2',
        options=[{'label': v, 'value': v} for v in all_topics.values()],
        placeholder='Select Topic# 2'),
        ],
        style={'width': '15%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
        id='dropdown_input3',
        options=[{'label': v, 'value': v} for v in all_topics.values()],
        placeholder='Select Topic# 3'),
        ],
        style={'width': '15%', 'display': 'inline-block'}),
        ])
    ]),
    html.Br(),
    html.Div(html.Button(id='submit-input', n_clicks=0,children='Submit', style={'fontSize':20, 'color': '#D81111'}),
                         style = {'textAlign': 'center'}),
    html.Div(html.Button(id='reset-input', n_clicks=0,children='reset', style={'color': '#07329C'}),
                         style = {'textAlign': 'center'}),

    # results:
    #html.Div(html.H3("RESULTS:", style = {'textAlign': 'center', 'height': '10px', 'fontSize':40,'color':'#1B698E'})),
    html.Br(),
    html.Br(),
    html.Div(html.H3("Comparative Historical Trends for the Selected Topics", 
            style = {'textAlign': 'center', 'height': '10px', 'fontSize':22, 'color': '#07329C', 'font-weight':'bold'})),
    dcc.Graph(id='historical-graph',),
    html.Br(),
    html.Div(html.H3("Comparative Consumer-Interest Forecasts", 
            style = {'textAlign': 'center', 'height': '10px', 'fontSize':22, 'color': '#07329C', 'font-weight':'bold'})),
    dcc.Graph(id='predictive-graph',),
    html.Br(),
    html.Br(),
    html.P('* Note that the predictive model needs to be updated weekly with new data for accurate forecasts going forward', 
            style = {'textAlign': 'center','color':'#07329C', 'fontSize':14, 'font-style': 'italic'}),
    ])])

#####################################################################

@app.callback(Output('historical-graph', 'figure'),
    [Input('submit-input', 'n_clicks')],
    [State('text_input1','value'),
    State('text_input2','value'),
    State('text_input3','value'),
    State('dropdown_input1','value'),
    State('dropdown_input2','value'),
    State('dropdown_input3','value'),])
def print_inp(n_clicks, tinput1, tinput2, tinput3, dinput1, dinput2, dinput3):
    if (n_clicks != 0):
        # extract only the inputs entered by user: This allows for more than 3 inputs through both text and dropdown channels
        uinputs = [tinput1, tinput2, tinput3, dinput1, dinput2, dinput3]
        uinputs = [i.lower() for i in uinputs if i is not None]

        try:
            utopics, all_matched_topics = process_related_topics(uinputs)
            print("Inputs are:", uinputs)

            ## Plot Historical Analytics
            allts = []
            for key, ts in utopics.items():
                label = [each[0] for each in zip(uinputs, all_matched_topics) if key in each][0]
                x = ts.index
                y = ts.values
                allts.append(go.Scatter(
                    x=x, y=y, name=label))

            # Layout goes separately
            layout = go.Layout(
                #title='If I wanted another subtitle',
                xaxis=dict(
                    title='Time',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f')),
                
                yaxis=dict(
                    title='Median Monthly Popularity',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f')))

            fig = go.Figure(data=allts, layout=layout)
            return fig

        except AttributeError:
            return html.Div(html.H4("Something went wrong. uinputs entered", uinputs
            ),style = {'textAlign': 'center', 'height': '10px', 'fontSize':26})

########################################################
# Second callback for predictive analytics:
# This is inefficient. I need to implement chained callbacks
### Still need to operationalize the reset button.
########################################################

@app.callback(Output('predictive-graph', 'figure'),
    [Input('submit-input', 'n_clicks')],
    [State('text_input1','value'),
    State('text_input2','value'),
    State('text_input3','value'),
    State('dropdown_input1','value'),
    State('dropdown_input2','value'),
    State('dropdown_input3','value'),])
def print_inp(n_clicks, tinput1, tinput2, tinput3, dinput1, dinput2, dinput3):
    if (n_clicks != 0):
        # extract only the inputs entered by user: This allows for more than 3 inputs through both text and dropdown channels
        uinputs = [tinput1, tinput2, tinput3, dinput1, dinput2, dinput3]
        uinputs = [i.lower() for i in uinputs if i is not None]

        try:
            utopics, all_matched_topics = process_related_topics(uinputs)
            print("Inputs are:", uinputs)

            ## Plot Historical Analytics
            # Weekly Forecast: Median Popularity
            allpredts = []
            for key in utopics.keys():
                label = [each[0] for each in zip(uinputs, all_matched_topics) if key in each][0]
                predts = predictions[predictions['predicted_topic']==key][-8:].reset_index()\
                        ['preds_likes_gb']
                x = predts.index
                y = predts.values
                allpredts.append(go.Scatter(
                    x=x, y=y, name=label))
            # Layout goes separately
            layout = go.Layout(
                title='Comparative Predictive Analytics',
                xaxis=dict(
                    title='Future Weeks',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f')),
                
                yaxis=dict(
                    title='Median Popularity\nWeekly Forecast',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f')))

            fig = go.Figure(data=allpredts, layout=layout)
    
            return fig

        except AttributeError:
            return html.Div(html.H4("Something went wrong. uinputs entered", uinputs
            ),style = {'textAlign': 'center', 'height': '10px', 'fontSize':26})





if __name__ == '__main__':
    app.run_server(debug=True)