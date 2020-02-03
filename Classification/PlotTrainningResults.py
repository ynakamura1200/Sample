from decimal import Decimal, ROUND_DOWN
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from sqlalchemy import desc

from assets.database import db_session
from assets.models import Conditions
from assets.models import Loss
from assets.models import Accuracy

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
plot_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = plot_app.server

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
    condition = list(filter(lambda con: con.id == condition_id, conditions))
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

plot_app.layout = html.Div(children=[
    html.H2(children='学習結果'),
    html.Div(children="【条件１】",
        id='condition1',
        style={
            'font-size': '16px',
            'margin-left':'-700px'
        }),
    html.Div(children=[
        html.Div(children="バッチサイズ=" + str(cond_loss_acc[condition_ids[0]][0][0].batch_size)),
        html.Div(children="学習係数=" + str(cond_loss_acc[condition_ids[0]][0][0].eta.quantize(Decimal('.01'), rounding=ROUND_DOWN))),
        html.Div(children="中間層のニューロン数=" + str(cond_loss_acc[condition_ids[0]][0][0].n_mid))
    ],
    style={
        'font-size': '12px',
        'margin-left':'-700px'
    }),
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
                        range=[0, max(max(cond_loss_acc[condition_ids[0]][2]), max(cond_loss_acc[condition_ids[0]][3]))+1]),
                    margin=dict(l=150, r=150, b=50, t=50)
                )
            }
        )
    ]),
    html.H2(children='　'),
    html.Div(children="【条件２】",
        id='condition2',
        style={
            'font-size': '16px',
            'margin-left':'-700px'
    }),
    html.Div(children=[
        html.Div(children="バッチサイズ=" + str(cond_loss_acc[condition_ids[1]][0][0].batch_size)),
        html.Div(children="学習係数=" + str(cond_loss_acc[condition_ids[1]][0][0].eta.quantize(Decimal('.01'), rounding=ROUND_DOWN))),
        html.Div(children="中間層のニューロン数=" + str(cond_loss_acc[condition_ids[1]][0][0].n_mid))
    ],
    style={
        'font-size': '12px',
        'margin-left':'-700px'
    }),
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
                        range=[0, max(max(cond_loss_acc[condition_ids[1]][2]), max(cond_loss_acc[condition_ids[1]][3]))+1]),
                    margin=dict(l=150, r=150, b=50, t=50)
                )
            }
        )    
    ])
],style={
    'textAlign': 'center',
    'width': '1000px',
    'margin': '0 auto'
})

if __name__ == '__main__':
    plot_app.run_server(debug=True)
