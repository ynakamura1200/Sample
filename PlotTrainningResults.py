import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from sqlalchemy import desc

from app import app
from assets.database import db_session
from assets.models import Conditions
from assets.models import Loss
from assets.models import Accuracy

def plot():

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    dashapp = dash.Dash(__name__, server=app, url_base_pathname='/plot/')
    
    condition_accuracys = db_session.query(Conditions.id, Conditions, Accuracy).\
        outerjoin(Accuracy, Conditions.id==Accuracy.condition_id).\
        order_by(desc(Accuracy.created_at)).\
        all()

    condition_ids = []
    conditions = []
    accuracys = []
    for ca in condition_accuracys:
        condition_ids.append(ca[0])
        conditions.append(ca[1])
        accuracys.append(ca[2])

    losses = db_session.query(Loss).\
        filter(Loss.condition_id.in_(condition_ids)).all()

    cond_loss_acc = {}
    for condition_id in condition_ids:
        condition = filter(lambda con: con.condition_id == condition_id, conditions)
        loss_list = list(filter(lambda loss: loss.condition_id == condition_id, losses))
        train_losses =[]
        test_losses = []
        epochs = []
        for loss in loss_list:
            train_losses.append(loss.train_loss)
            test_losses.append(loss.test_loss)
            epochs.append(loss.epoch)

        acc = filter(lambda acc: acc.condition_id == condition_id, accuracys)
        cond_loss_acc[condition_id] = [condition, epochs, train_losses, test_losses, acc]

    dashapp.layout = html.Div(children=[
        html.H2(children='学習結果'),
        html.Div(children=[
            dcc.Graph(
                id='review_graph1',
                figure={
                    'data':[
                        go.Scatter(
                            x=cond_loss_acc[condition_ids[0]][1],
                            y=cond_loss_acc[condition_ids[0]][2],
                            mode='lines+markers',
                            name='教師データ',
                            opacity=0.7,
                            yaxis='y1'
                        ),
                        go.Scatter(
                            x=cond_loss_acc[condition_ids[0]][1],
                            y=cond_loss_acc[condition_ids[0]][3],
                            mode='lines+markers',
                            name='テストデータ',
                            opacity=0.7,
                            yaxis='y1'
                        )
                    ],
                    'layout': go.Layout(
                        title='損失関数',
                        xaxis=dict(title='epoch数'),
                        yaxis=dict(title='loss',side='left', showgrid=False,
                            range=[0, max(train_losses)+1]),
                        margin=dict(l=150, r=150, b=50, t=50)
                    )
                }
            )
        ]),
        html.H2(children=''),
        html.Div(children=[
            dcc.Graph(
                id='review_graph2',
                figure={
                    'data':[
                        go.Scatter(
                            x=cond_loss_acc[condition_ids[1]][1],
                            y=cond_loss_acc[condition_ids[1]][2],
                            mode='lines+markers',
                            name='教師データ',
                            opacity=0.7,
                            yaxis='y1'
                        ),
                        go.Scatter(
                            x=cond_loss_acc[condition_ids[1]][1],
                            y=cond_loss_acc[condition_ids[1]][3],
                            mode='lines+markers',
                            name='テストデータ',
                            opacity=0.7,
                            yaxis='y1'
                        )
                    ],
                    'layout': go.Layout(
                        title='損失関数',
                        xaxis=dict(title='epoch数'),
                        yaxis=dict(title='loss',side='left', showgrid=False,
                            range=[0, max(train_losses)+1]),
                        margin=dict(l=150, r=150, b=50, t=50)
                    )
                }
            )    
        ])
    ],style={
        'textAlign': 'center',
        'width': '1200px',
        'margin': '0 auto'
    })
    return dashapp.index()
