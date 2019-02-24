# # -*- coding: utf-8 -*-
# Import all libraries 

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import base64
from gensim.models import Word2Vec
from joblib import dump, load
import json

import plotly
import plotly.plotly as py
import plotly.graph_objs as go


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


#################### DEBUGGING MEMORY ############################
# from pympler import tracker
# tr = tracker.SummaryTracker() 
# tr.print_diff() 
######################################################################

#######################################################################
# DEFINE ALL FUNCTIONS
#####################################process###########################

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

#-----------------------------------------------------------------------------------------#

def process_related_topics(input_array):

    """
    QUERY AGAINST INTERNAL TOPICS WITH TRAINED NMF MODEL
    """

    all_matched_topics=[]
    for uinput in input_array:
        try:
            matched_topic = user2topic(uinput)
            # RECURSION BELOW MIGHT BE CAUSING MEMORY LEAK:
            if matched_topic == 0:
                matched_topic = user2topic(uinput[:-1]) # try removing the last char ('s)
                if matched_topic == 0:
                    matched_topic = user2topic(uinput[:-2]) # try removing the last 2 chars ('es)        
            
        except KeyError:
            user2similar = ''
            pass
        
        all_matched_topics.append(matched_topic)
    return all_matched_topics

#-----------------------------------------------------------------------------------------#

def topic_data(list_matched_topics):

    """
    EXTRACT HISTORICAL TOPIC DATA FOR FORECASTING.
    RESAMPLED BY MEDIAN
    """
    utopics = {}
    for topic in list_matched_topics:
        utopics[topic] = pilot[pilot['predicted_topic'] == topic].set_index('date')\
                                                            ['likes_count'].resample('M').median()

    return utopics

#-----------------------------------------END OF FUNCTIONS----------------------------------#


#######################################################################
# LOAD PREVIOUSLY TRAINED MODELS AND DATA
# original Slug's greater than heroku's soft limit.
# Used Lz4 compression on nmf and vect; based on this comparison: https://goo.gl/tHzmJU 
# Cut down ~100 MB on Slug size.
#######################################################################

# word2vec model for stopic_dataynonyms:
w2v_loc = "deploy_data/w2v_bigram.model"
w2v = Word2Vec.load(w2v_loc)

# NMF topic models for matching user input to topics:
nmf_loc = 'deploy_data/latest_nmf_tfidf_ntop-200_nftr_50000_ngrams_(1, 3)_.joblib'
nmf = load(nmf_loc+'.lz4') 

vect_loc = 'deploy_data/latest_tfidf_tfidf_ntop-200_nftr_50000_ngrams_(1, 3)_.joblib'
vectorizer = load(vect_loc+'.lz4')

# import data containing topic assignments from Step2:
pilot_loc = 'deploy_data/step2_NMF_topics=200_assigned.parquet'
pilot = pd.read_parquet(pilot_loc, engine='pyarrow')

all_topics_loc = 'deploy_data/all_topics_nmf_ntop-200_nftr_50000_ngrams_(1, 3).json'
with open(all_topics_loc, 'r') as fp:
    all_topics = json.load(fp)

# File with time-series analytics for that topic:
ts_loc = 'deploy_data/all_preds_ts_gb_hptuning=False_nmf_ntopics=198.parquet'
predictions = pd.read_parquet(ts_loc, engine='pyarrow')



#######################################################################
# DASH APP
#######################################################################

main_img = base64.b64encode(open('./img/blocks.jpg', 'rb').read())


app.layout = html.Div([
# storage for utopics: lost on page reload
    dcc.Storage(id='mem_matchedtop'),
#title:
    html.Div(html.H1('TO·P·icks', style = {'textAlign': 'center', 'padding': '2px', 'height': '20px', 'margin-top': '10px', 'fontSize':70, 'font-weight':'bold','color': '#D81111'})),
# subtitle:    
    html.Div(html.H3('Forecast Consumer-Interest in Topics', style = {'textAlign': 'center', 'height': '10px','color':'#07329C', 'fontSize':35, 'font-weight':'bold', 'margin-top': '60px'})),
# imagetitle:    
    html.Div(html.Img(id='head-image', src='data:image/jpeg;base64,{}'.format(main_img.decode('ascii')),
                      style = {'width':'100%', 'height': '500px', 'padding':'0','margin':'0','margin-top': '30px','box-sizing':'border-box'})),
    #html.Br(),
# Basic info about Product:
     html.Div(html.P('TO·P·icks uses machine-learning to predict relative consumer-interest in topics. Enter any three topics below and TO·P·icks will help you forecast\
        how popular these topics be in the next few weeks or months relative to each other:', 
        style = {'textAlign': 'center', 'margin':'30', 'color':'#07329C', 'fontSize':18})),

# User Inputs:
    html.H4('Select topics of interest: '),    
    html.Div(title='select inputs', id='selections',children=[

# Text input:
    html.P('(Suggestions: Law, Tax, Climate change, Guns, Russia, Mexico, Technology, Syria, Terrorism, Health care etc.)'),
        dcc.Input(id='text_input1', type='text', placeholder='Enter first topic'),
        dcc.Input(id='text_input2', type='text', placeholder='Enter second topic'),
        dcc.Input(id='text_input3', type='text', placeholder='Enter third topic'),


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
    html.Div(html.Button(id='submit-input', n_clicks=1,children='Submit', style={'fontSize':20, 'color': '#D81111'}),
                         style = {'textAlign': 'center'}),
    html.Div(html.P("(Our Monkeys are a little slow at plotting graphs. Please wait after submitting.)",
            style = {'textAlign': 'center','color':'#07329C', 'fontSize':16, 'font-style': 'italic'})),
    html.Br(),
    html.Div(html.Button(id='reset-input', n_clicks=0,children='reset', style={'color': '#07329C'}),
                         style = {'textAlign': 'center'}),

    # results:
    html.Br(),
    html.Div(html.H3("Comparative Historical Trends for the Selected Topics", 
            style = {'textAlign': 'center', 'height': '10px', 'fontSize':22, 'color': '#07329C', 'font-weight':'bold'})),

    # details about historical trends in a collapsable format:
    html.Details([
    html.Summary('Click to see details about this graph',style = {'fontSize':18, 'font-weight':'bold'}),
    html.Div([
        html.P("This graph represents historical data on how topics ebb and flow in their popularity over the last few years. \
            Popularity is defined in terms of the number of Facebook likes received by a story or an article related to that specific topic.\
        As the number of user comments and re-shares correlate strongly with the popularity of a topic, popularity was used as a\
         composite indicator of consumer-reception and engagement.", 
         style = {'textAlign': 'left','color':'#07329C', 'fontSize':18}),

         html.P("The axis labels are not being displayed because of a known incompatibility with DCC Storage.",)],
            style = {'textAlign': 'center','color': '#D81111', 'fontSize':16, 'font-style': 'italic'}),], 
    style = {'textAlign': 'center'}),

    # Plot historical trends:
    dcc.Graph(id='historical-graph',),
    html.Br(),
    html.Div(html.H3("Comparative Consumer-Interest Forecasts", 
            style = {'textAlign': 'center', 'height': '10px', 'fontSize':22, 'color': '#07329C', 'font-weight':'bold'})),

    # details about the graph in a collapsable format:
    html.Details([
    html.Summary('Click to see details about this graph',style = {'fontSize':18, 'font-weight':'bold'}),
    html.Div([
        html.P("This graph represents predicted median popularity per week for the next 8 weeks for your selected set of topics.",
            style = {'textAlign': 'center','color':'#07329C', 'fontSize':18}),
        html.P("- If there are missing forecasts, that means TOPicks doesn't have enough data to make predictions on that specific topic at the moment.",)],
            style = {'textAlign': 'center','color':'#07329C', 'fontSize':16, 'font-style': 'italic'}),], 
    style = {'textAlign': 'center'}),

    # Generate predictive graph
    dcc.Graph(id='predictive-graph',),
    html.Br(),
    html.Br(),


    # Author details:
    html.P('© Tarun Gupta, Ph.D. (Data Science Fellow) Insight, Toronto, ON', 
            style = {'textAlign': 'center','color':'#07329C', 'fontSize':18, 'font-weight':'bold'}),
    html.P('https://linkedin.com/in/taruneuro/', 
            style = {'textAlign': 'center','color':'#07329C', 'fontSize':16, 'font-style': 'italic'}),
    ])])

########################################################
# Process user-input; send NMF cluster and time-series predictions to hidden Div
## pd.series is not json serializable in Dash; utopics changes to list when passed to dcc.storage
########################################################

@app.callback(Output('mem_matchedtop', 'data'),
    [Input('submit-input', 'n_clicks')],
    [State('text_input1','value'),
    State('text_input2','value'),
    State('text_input3','value'),
    State('dropdown_input1','value'),
    State('dropdown_input2','value'),
    State('dropdown_input3','value'),])
def print_inp(n_clicks, tinput1, tinput2, tinput3, dinput1, dinput2, dinput3):
    if (n_clicks != 0):
        # extract only the inputs entered by user: 
        # Allow for more than 3 inputs through both text and dropdown channels
        uinputs = [tinput1, tinput2, tinput3, dinput1, dinput2, dinput3]
        uinputs = [y.lower() for y in [x.strip() for x in uinputs if x is not None] if len(y)>0] # get rid of empty input
        all_matched_topics = process_related_topics(uinputs)
        return {'all_matched_topics': all_matched_topics, 'uinputs': uinputs}


@app.callback(Output('historical-graph', 'figure'),
    [Input('mem_matchedtop', 'data')],)
def process_utop(mem_data):
    if  mem_data: 
        
        all_matched_topics = mem_data['all_matched_topics']
        uinputs = mem_data['uinputs']
        utopics = topic_data(all_matched_topics)

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
            xaxis=dict(
                title='Time',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#07329C')),
            
            yaxis=dict(
                title='Monthly Popularity / Topic (Median)',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#07329C')))

        fig = go.Figure(data=allts, layout=layout)
        return fig


########################################################
# Second callback for predictive analytics:
########################################################

@app.callback(Output('predictive-graph', 'figure'),
    [Input('mem_matchedtop', 'data')],)
def process_utop(mem_data):
    if  mem_data: 
        
        all_matched_topics = mem_data['all_matched_topics']
        uinputs = mem_data['uinputs']
        utopics = topic_data(all_matched_topics)


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
            xaxis=dict(
                title='Future Weeks',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='rgb(107, 107, 107)')),
            
            yaxis=dict(
                title='Weekly Popularity Forecast (Median)',
                titlefont=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#07329C')))

        fig = go.Figure(data=allpredts, layout=layout)

        # release References for gc collection
        all_matched_topics = None
        uinputs = None
        utopics = None
        return fig


#############################################
# RESET ALL FIELDS IF RESET IS CLICKED
#############################################

@app.callback(Output('text_input1','value'),
    [Input('reset-input', 'n_clicks')],)
def reset(n_clicks):
    if n_clicks>0:
        return ''

@app.callback(Output('text_input2','value'),
    [Input('reset-input', 'n_clicks')],)
def reset(n_clicks):
    if n_clicks>0:
        return ''

@app.callback(Output('text_input3','value'),
    [Input('reset-input', 'n_clicks')],)
def reset(n_clicks):
    if n_clicks>0:
        return ''

@app.callback(Output('dropdown_input1','value'),
    [Input('reset-input', 'n_clicks')],)
def reset(n_clicks):
    if n_clicks>0:
        return ''

@app.callback(Output('dropdown_input2','value'),
    [Input('reset-input', 'n_clicks')],)
def reset(n_clicks):
    if n_clicks>0:
        return ''

@app.callback(Output('dropdown_input3','value'),
    [Input('reset-input', 'n_clicks')],)
def reset(n_clicks):
    if n_clicks>0:
        return ''

#############################################



if __name__ == '__main__':
    app.run_server(debug=True)