from math import *
import numpy as np
import matplotlib.pylab as plt
# import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp


import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff

# plotly.offline.init_notebook_mode()

from ipywidgets import widgets

import dash_defer_js_import as dji


x0 = np.linspace(0, 1, 6)
tspan = t_eval=np.linspace(0,500,3000)

iColor = 'gold' #[.9,.8,0]
eColor = 'darkgoldenrod' #[.45,.4,0]
dColor_w = 'maroon' #[.8, 0, 0]
dColor_s = 'crimson' #[1, .1, .3]
uColor = 'black' #[0,0,0]
tol = pow(10,-3)


def h(x1,x2,ell):
    return ell*exp(-ell*(x1-x2))/pow((1+exp(-ell*(x1-x2))),2) #need "return" if you want it to output whatever is on this line
                                           #(!) Nothing after return will run. Python breaks out of the function @ return. 
def D1C(I,xm,aye,bee,G):
    return 2*aye*I+bee*max(G-xm,0)

def D1Ci(I,xm,aye,bee,G):
    return (I-bee*max(G-xm,0))/(2*aye)


def classFinder(x,y):
    if x>=1-tol and y >=1-tol:
        col = iColor
    elif x<=tol and y<=tol:
        col = eColor
    elif min(x,y)<tol and max(x,y)>=tol:
        if max(x,y)>=1-tol:
            col = dColor_s
        else:
            col = dColor_w
    else:
        col = uColor
    return col


def xd1(x1,x2,aye,bee,G,ell,dee): 
    if h(x1,x2,ell)<D1C(0,x1,aye,bee,G) and x1>0:
        v = -dee
    elif h(x1,x2,ell)<D1C(0,x1,aye,bee,G) and x1<=0:
        v = 0
    elif h(x1,x2,ell)>D1C(dee,x1,aye,bee,G) and x1>=1:
        v = 0
    else:
        v = D1Ci(h(x1,x2,ell),x1,aye,bee,G)-dee
        
    if v>0 and x1>1:
        v=0
    elif v<0 and x1<0:
        v = 0
        
    return v


def xd2(x1,x2,aye,bee,G,ell,dee): 
    if h(x2,x1,ell)<D1C(0,x2,aye,bee,G) and x2>0:
        v = -dee
    elif h(x2,x1,ell)<D1C(0,x2,aye,bee,G) and x2==0:
        v = 0
    elif h(x2,x1,ell)>D1C(dee,x2,aye,bee,G) and x2==1:
        v = 0
    else:
        v = D1Ci(h(x2,x1,ell),x2,aye,bee,G)-dee
        
    if v>0 and x2>1:
        v=0
    elif v<0 and x2<0:
        v = 0
        
    return v



traj1 = []
traj2 = []
ssColor = []
# plt.axis('square')
# plt.axis([0,1,0,1])
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')

lgd_lw = 5


# fig, ax = plt.subplots()   
# plt.plot(0,0,'-',color=iColor,label='Escalated Inclusive', linewidth =lgd_lw)
# plt.plot(0,0,'-',color=eColor,label='De-escalated Inclusive',linewidth =lgd_lw)
# plt.plot(0,0,'-',color=dColor_w,label='Weak Dictatorial',linewidth =lgd_lw)
# plt.plot(0,0,'-',color=dColor_s,label='Strong Dictatorial',linewidth =lgd_lw)
# ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, shadow=True, ncol=2) 



# if B.value == 1 and A.value!=1:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+str(A.value)+"I_{it}^2+"+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"
# elif A.value==1 and B.value!=1:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+"I_{it}^2+"+str(B.value)+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"
# elif A.value==1 and B.value==1:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+"I_{it}^2+"+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"    
# else:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+str(A.value)+"I_{it}^2+"+str(B.value)+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"

##################################################################
#ALTERNATIVE setup where you feed fig into figmaker
##################################################################
# fig = go.FigureWidget()
# fig.update_layout(
# #     autosize=False,
# #     width=600,
# #     height=600,
#     margin=dict(
#         l=50,
#         r=50,
#         b=50,
#         t=100,
#         pad=4
#     ),
#     title={
# #         'text': ('Phase Portrait\n' + parameter_info),
#         'text': ('Phase Portrait'),
#         'y':0.9,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'},
#     showlegend=False,
#     paper_bgcolor="LightSteelBlue",
# )
# fig.update_xaxes(range=[0,1],title_text='$x_1$')
# fig.update_yaxes(range=[0,1],title_text='$x_2$')
# def figMaker(fig,aye,bee,G,ell,dee):
    
#     def f(t,r):
#         x1,x2 = r
#         f1 = xd1(x1,x2,aye,bee,G,ell,dee)
#         f2 = xd2(x1,x2,aye,bee,G,ell,dee)
#         return f1, f2

#     for x10 in x0:
#         for x20 in x0:
#             sol=scipy.integrate.solve_ivp(f,(tspan[0],tspan[-1]),(x10,x20),t_eval=tspan)
#             xx,yy = sol.y
#             xx[np.where((xx>1)|(yy>1))]=1
#             yy[np.where((xx>1)|(yy>1))]=1
#             c = classFinder(xx[-1],yy[-1])
        
# #         traj1.extend(x)
# #         traj2.extend(y)

# #         plt.plot(x,y,color = c)
# #         plt.plot(x[0],y[0],'.',color=c)
# #         plt.plot(x[-1],y[-1],'o',color='green')
        
#             fig.add_trace(go.Scatter(x=xx,y=yy,
#                                      line=dict(color=c, width=4)))
# #         fig.add_trace(go.Scatter(x=xx[-2:-1],y=yy[-2:-1],
# #                                  mode='markers',
# #                                  marker_color='rgba(255,255,255, .9)',
# #                                  marker_line_width=2, marker_size=15))
        
#################
def figMaker(aye,bee,G,ell,dee):
#     parameter_info='$\lambda = '+str(ell)+"$; "+'$\delta = '+str(dee)+"$; " #+"$C(I_{it},x_{t-\Delta})="+str(aye)+"I_{it}^2+"+str(bee)+"\max\{"+str(G)+"- x_{i,t-\Delta},0 \}I_{it}$"
    parameter_info='$\lambda = '+str(ell)+";\ "+'\delta = '+str(dee)+";\ "+"C(I_{it},x_{t-\Delta})="+str(aye)+"I_{it}^2+"+str(bee)+"\max\{"+str(G)+"- x_{i,t-\Delta},0 \}I_{it}$"
#     print(parameter_info)
    theFig = go.FigureWidget()

    theFig.update_layout(
#         autosize=False,
#         width=600,
#         height=600,
#         margin=dict(
#             l=50,
#             r=50,
#             b=50,
#             t=100,
#             pad=4
#         ),
        autosize=True,
        title={
            'text': ((parameter_info)),
#             'text': ('Phase Portrait'),
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        showlegend=False,
        paper_bgcolor="LightSteelBlue",
    )

    theFig.update_xaxes(range=[0,1],title_text='$x_1$',constrain="domain")
    theFig.update_yaxes(range=[0,1],title_text='$x_2$')
    theFig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    
    
# if B.value == 1 and A.value!=1:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+str(A.value)+"I_{it}^2+"+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"
# elif A.value==1 and B.value!=1:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+"I_{it}^2+"+str(B.value)+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"
# elif A.value==1 and B.value==1:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+"I_{it}^2+"+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"    
# else:
#     parameter_info='$\lambda = '+str(L.value)+"$; "+'$\delta = '+str(delta.value)+"$; "+"$C(I_{it},x_{t-\Delta})="+str(A.value)+"I_{it}^2+"+str(B.value)+"\max\{"+str(g.value)+"- x_{i,t-\Delta},0 \}I_{it}$"
    

    def f(t,r):
        x1,x2 = r
        f1 = xd1(x1,x2,aye,bee,G,ell,dee)
        f2 = xd2(x1,x2,aye,bee,G,ell,dee)
        return f1, f2

    for x10 in x0:
        for x20 in x0:
            sol=scipy.integrate.solve_ivp(f,(tspan[0],tspan[-1]),(x10,x20),t_eval=tspan)
            xx,yy = sol.y
            xx[np.where((xx>1)|(yy>1))]=1
            yy[np.where((xx>1)|(yy>1))]=1
            c = classFinder(xx[-1],yy[-1])
        
#         traj1.extend(x)
#         traj2.extend(y)

#         plt.plot(x,y,color = c)
#         plt.plot(x[0],y[0],'.',color=c)
#         plt.plot(x[-1],y[-1],'o',color='green')
        
            theFig.add_trace(go.Scatter(x=xx,y=yy,
                                     line=dict(color=c, width=4)))
#         fig.add_trace(go.Scatter(x=xx[-2:-1],y=yy[-2:-1],
#                                  mode='markers',
#                                  marker_color='rgba(255,255,255, .9)',
#                                  marker_line_width=2, marker_size=15))
    return theFig
        
        

# container = widgets.HBox(children=[L, delta])        
# widgets.VBox([container,
#               container2,
#               fig])

# plt.show()


fig = figMaker(3.2,5,.35,5,.1)


fig.show()


# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly.express as px
# import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
# import dash_design_kit as ddk
# import dash_daq as daq

import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#for displaying LaTeX
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
"""

#Original app.layout B4 trying to do mathjax...
# app.layout = html.Div(children=[
#     html.H1(children='Phase Portrait'),

# #     html.Div(children='''
# #         Dash: A web application framework for Python.
# #     '''),
#     dcc.Slider(
#         id='L--slider',
#         min=0.1,
#         max=10,
#         value=5,
#         # marks={0.1: '0.1', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1', 1.25: '1.25', 1.5: '1.5', 1.75: '1.75', 2: '2', 2.25: '2.25', 2.5: '2.5', 2.75: '2.75', 3: '3', 3.25: '3.25', 3.5: '3.5', 3.75: '3.75', 4: '4', 4.25: '4.25', 4.5: '4.5', 4.75: '4.75', 5: '5', 5.25: '5.25', 5.5: '5.5', 5.75: '5.75', 6: '6', 6.25: '6.25', 6.5: '6.5', 6.75: '6.75', 7: '7', 7.25: '7.25', 7.5: '7.5', 7.75: '7.75', 8: '8', 8.25: '8.25', 8.5: '8.5', 8.75: '8.75', 9: '9', 9.25: '9.25', 9.5: '9.5', 9.75: '9.75', 10: '10'},
#         marks = {0.1: '0.1', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'},
#         step=.1,
#         tooltip={
#             'placement': 'top'
#         }
#     ),
#     dcc.Graph(
#         id='phase-portrait',
#         figure=fig
#     )        
# ])


app.layout = html.Div([
    dbc.Container(children=[
        html.H1(children='Phase Portrait'),
#      html.Div(children='''
#          Dash: A web application framework for Python.
#      '''),
        html.Label('$\lambda$'),
        dcc.Slider(
            id='L--slider',
            min=0.1,
            max=10,
            value=5,
            # marks={0.1: '0.1', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1', 1.25: '1.25', 1.5: '1.5', 1.75: '1.75', 2: '2', 2.25: '2.25', 2.5: '2.5', 2.75: '2.75', 3: '3', 3.25: '3.25', 3.5: '3.5', 3.75: '3.75', 4: '4', 4.25: '4.25', 4.5: '4.5', 4.75: '4.75', 5: '5', 5.25: '5.25', 5.5: '5.5', 5.75: '5.75', 6: '6', 6.25: '6.25', 6.5: '6.5', 6.75: '6.75', 7: '7', 7.25: '7.25', 7.5: '7.5', 7.75: '7.75', 8: '8', 8.25: '8.25', 8.5: '8.5', 8.75: '8.75', 9: '9', 9.25: '9.25', 9.5: '9.5', 9.75: '9.75', 10: '10'},
            marks = {0.1: '0.1', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'},
            step=.1,
            tooltip={
                'placement': 'top'
                }
            ),
        dcc.Graph(
            id='phase-portrait',
            figure=fig
            )
    ]),
    mathjax_script
])

@app.callback(
    Output('phase-portrait','figure'),
    Input('L--slider','value'))
def update_graph(L_value):
    print("I'm alive!")
    fig = figMaker(3.2,5,.35,L_value,.1)
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
    








