#%% Full-Stack ML Analyses Dashboard
#%% Designed by: SUMERLabs from Sumertech

#%% Importing Front-End Libraries
import plotly.graph_objs as pgo

from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output

# Statistics & ML Libraries
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import linregress
from xgboost import XGBRegressor


#%% Step 1: Data Analyses
class Data:

    # Initialization
    def __init__(self,filename,percent):
        self.filename=filename
        self.percent=percent
        self.le=LabelEncoder()
        self.sc=StandardScaler()
        self.lower_limit=5000
        self.upper_limit=30000
        self.function1 = 'tanh'
        self.function2 = 'linear'

    # Opening the Dataframe
    def open_file(self):
        self.dataframe=pd.read_csv(self.filename)
        self.columns=list(self.dataframe.columns)

    # Label Encoding for Categorical Values
    def label_encode(self):
        self.le_list=self.le.fit(self.dataframe.iloc[:,5].values)
        self.le_list=self.le.transform(self.dataframe.iloc[:,5].values)

        self.new_dataframe=np.column_stack((self.dataframe.iloc[:,:5].values,self.le_list,self.dataframe.iloc[:,6:].values))
        self.new_dataframe=pd.DataFrame(self.new_dataframe,columns=self.columns)

    # First Analyses of the Data
    def analyse(self):
        # Creating the Correlation Matrix
        self.df_corr=self.dataframe.iloc[:,6:-1].corr()
        self.mask=np.triu(np.ones_like(self.df_corr, dtype=bool))
        self.df_mask=self.df_corr.mask(self.mask)

    # Data Preprocessing
    def data_cleanse(self):
        # Dropping NaN Values
        self.new_dataframe=self.new_dataframe.dropna()

        # Cleaning in between the Upper & Lower Limits
        self.new_dataframe=self.new_dataframe[self.new_dataframe.iloc[:,-1]>=self.lower_limit]
        self.new_dataframe=self.new_dataframe[self.new_dataframe.iloc[:,-1]<=self.upper_limit]

    # Data Seperating
    def separate(self):
        self.independent = self.new_dataframe.iloc[:, 5:-1]
        self.dependent = self.new_dataframe.iloc[:, -1]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.independent, self.dependent,
                                                                                test_size=self.percent, shuffle=True,
                                                                                random_state=None)

    # SVR Model Generation
    def svr_model(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

        self.regressor = SVR(kernel='rbf')
        self.regressor.fit(self.x_train, self.y_train)
        self.svr_reg = self.regressor.predict(self.x_test)

        self.svr_results = pd.DataFrame(np.column_stack((self.y_test, self.svr_reg)),
                                        columns=['Test Value (SVR)', 'Results (SVR)']).sort_values('Test Value (SVR)')

    # XGBR Model Generation
    def xgb_model(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

        self.regressor2=XGBRegressor()
        self.regressor2.fit(self.x_train, self.y_train)
        self.xgb_reg=self.regressor2.predict(self.x_test)

        self.xgb_results = pd.DataFrame(np.column_stack((self.y_test, self.xgb_reg)),
                                        columns=['Test Value (XGB)', 'Results (XGB)']).sort_values('Test Value (XGB)')

    # RF Model Generation
    def rf_model(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

        self.regressor3=RandomForestRegressor()
        self.regressor3.fit(self.x_train,self.y_train)
        self.rf_reg=self.regressor3.predict(self.x_test)

        self.rf_results = pd.DataFrame(np.column_stack((self.y_test, self.rf_reg)),
                                        columns=['Test Value (RF)', 'Results (RF)']).sort_values('Test Value (RF)')

    # DT Model Generation
    def dt_model(self):
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

        self.regressor4=DecisionTreeRegressor()
        self.regressor4.fit(self.x_train,self.y_train)
        self.dt_reg=self.regressor4.predict(self.x_test)

        self.dt_results = pd.DataFrame(np.column_stack((self.y_test, self.dt_reg)),
                                        columns=['Test Value (DT)', 'Results (DT)']).sort_values('Test Value (DT)')

    # Running the Whole Class
    def run_data(self):
        self.open_file()
        self.label_encode()
        self.analyse()
        self.data_cleanse()
        self.separate()
        self.svr_model()
        self.xgb_model()
        self.rf_model()
        self.dt_model()

        return self.dataframe,self.new_dataframe,\
               self.df_mask,self.svr_results,\
               self.xgb_results,self.rf_results,\
               self.dt_results

#%% Class Use
file='solar_power.csv'
percent=0.2
d=Data(file,percent)
old_df,new_df,df_corr,svr,xgb,rf,dt=d.run_data()
svr_slope,svr_intercept,svr_r_value,svr_p_value,svr_std_err=linregress(svr.iloc[:,0].values,svr.iloc[:,1].values)
xgb_slope,xgb_intercept,xgb_r_value,xgb_p_value,xgb_std_err=linregress(xgb.iloc[:,0].values,xgb.iloc[:,1].values)
rf_slope,rf_intercept,rf_r_value,rf_p_value,rf_std_err=linregress(rf.iloc[:,0].values,rf.iloc[:,1].values)
dt_slope,dt_intercept,dt_r_value,dt_p_value,dt_std_err=linregress(dt.iloc[:,0].values,dt.iloc[:,1].values)

#%% Step 2: App Design
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
                html.H1("RASAT ML Software"),
                html.Div([
                            dcc.Tabs(
                                id="tabs-with-classes-2",
                                value='tab-1',
                                parent_className='custom-tabs',
                                className='custom-tabs-container',
                                children=[
                                    dcc.Tab(
                                        label='Veri Önişleme',
                                        value='tab-1',
                                        className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                    dcc.Tab(
                                        label='Öznitelik Bilgisi',
                                        value='tab-2',
                                        className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                    dcc.Tab(
                                        label='Model Grafikleri',
                                        value='tab-3', className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                    dcc.Tab(
                                        label='Analiz Sonuçları',
                                        value='tab-4',
                                        className='custom-tab',
                                        selected_className='custom-tab--selected'
                                    ),
                                ]),
                            html.Div(id='tabs-content-classes-2')
                        ]),
                html.H4("Copyright of SUMERLabs by Sumer Technology")
                ])

@app.callback(Output('tabs-content-classes-2', 'children'),
              Input('tabs-with-classes-2', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Veri Ön Analizi'),
            html.Div([
                dcc.Graph(id='histogram',
                          style={'width': '90vh', 'height': '60vh'},
                          figure={'data': [
                              pgo.Histogram(x=new_df.iloc[:, -1],
                                            xbins=dict(start=0,
                                                       end=50000,
                                                       size=500),
                                            marker=dict(color='rgba(255,150,0,1)'),
                                            )],
                              'layout': pgo.Layout(title='Elektrik Enerjisi Üretimi Dağılımı',
                                                   xaxis={'title': 'Elde Edilen Güneş Enerjisi (kWh)'})}
                          ),

                dcc.Graph(id='boxplot',
                          style={'width': '90vh', 'height': '60vh'},
                          figure={'data': [
                              pgo.Box(
                                  y=new_df.iloc[:, -1],
                                  boxpoints='all',
                                  name='Güneş Enerjisi Periyodik Üretim Verisi',
                                  jitter=0.3,
                                  pointpos=0
                              )],
                              'layout': pgo.Layout(title='Elektrik Verisi Aykırı Değer Analizi',
                                                   yaxis={'title': 'Elektrik Verisi İstatistik Analizi'})}
                          )

                ],style={'border':"5px black solid"})
            ])

    elif tab == 'tab-2':
        return html.Div([
            html.H3('Öznitelik Bilgisi'),
            html.Div([
                dcc.Graph(id='feature-map',
                          style={'width': '60vh', 'height': '60vh'},
                          figure={'data': [
                              pgo.Heatmap(
                                  z=df_corr,
                                  x=df_corr.columns.values,
                                  y=df_corr.columns.values,
                                  colorscale='Jet',
                                  zmin=-1,
                                  zmax=1
                              )],
                              'layout': pgo.Layout(title='Elektrik Verisi Öznitelik Seçimi')}
                          )
            ],style={'border':"5px black solid"})
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H3('Model Analizleri'),
            html.Div([
                dcc.Graph(id='ml-model',
                          style={'width': '100vh', 'height': '80vh'},
                          figure={'data': [
                              pgo.Scatter(
                                  name='SVR Modeli=>y={:.3f}x+{:.3f}'.format(svr_slope, svr_intercept),
                                  x=svr.iloc[:, 0].values,
                                  y=svr.iloc[:, 1].values,
                                  mode='markers'
                              ),

                              pgo.Scatter(
                                  name='XGBoost Modeli=>y={:.3f}x+{:.3f}'.format(xgb_slope, xgb_intercept),
                                  x=xgb.iloc[:, 0].values,
                                  y=xgb.iloc[:, 1].values,
                                  mode='markers'
                              ),

                              pgo.Scatter(
                                  name='Rastgele Orman Modeli=>y={:.3f}x+{:.3f}'.format(rf_slope, rf_intercept),
                                  x=rf.iloc[:, 0].values,
                                  y=rf.iloc[:, 1].values,
                                  mode='markers'
                              ),

                              pgo.Scatter(
                                  name='Karar Ağaçları Modeli=>y={:.3f}x+{:.3f}'.format(dt_slope, dt_intercept),
                                  x=rf.iloc[:, 0].values,
                                  y=rf.iloc[:, 1].values,
                                  mode='markers'
                              )

                          ],
                              'layout': pgo.Layout(title='Yapay Öğrenme Model Sonuçları',
                                                   xaxis={'title': 'Test Verisi (kWh)'},
                                                   yaxis={'title': 'Model Tahmini (kWh)'},
                                                   )}
                          )

            ],style={'border':"5px black solid"})
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Model R2 Skorları'),
            html.Div([
                html.Div(children=[
                    html.H5('Destek Vektör Makineleri'),
                    html.H5(' % {:.2f}'.format(100 * svr_r_value),
                            style={'fontWeight': 'bold', 'color': '#011aeb'}),
                ],style={'border':"5px black solid"}),


                html.Div(children=[
                    html.H5('Karar Ağaçları'),
                    html.H5('% {:.2f}'.format(100 * dt_r_value),
                            style={'fontWeight': 'bold', 'color': '#00aeef'}),
                ],style={'border':"5px black solid"}),

                html.Div(children=[
                    html.H5('Rastgele Orman'),
                    html.H5('% {:.2f}'.format(100 * rf_r_value),
                            style={'fontWeight': 'bold', 'color': 'orange'}),

                ],style={'border':"5px black solid"}),

                html.Div(children=[
                    html.H5('XGBoost'),
                    html.H5('% {:.2f}'.format(100 * xgb_r_value),
                            style={'fontWeight': 'bold', 'color': 'red'}),

                ],style={'border':"5px black solid"})
            ])
        ])

#%% Main
app.run_server(debug=True)