import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Load your dataset
spacex_df = pd.read_csv("dataset_part_2.csv")

# MAX & MIN Payload
max_payload = spacex_df['PayloadMass'].max()
min_payload = spacex_df['PayloadMass'].min()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Encode Categorical Columns
le_site = LabelEncoder()
le_booster = LabelEncoder()
le_orbit = LabelEncoder()

spacex_df['LaunchSite_enc'] = le_site.fit_transform(spacex_df['LaunchSite'])
spacex_df['BoosterVersion_enc'] = le_booster.fit_transform(spacex_df['BoosterVersion'])
spacex_df['Orbit_enc'] = le_orbit.fit_transform(spacex_df['Orbit'])

# Features and Target
X = spacex_df[['PayloadMass', 'LaunchSite_enc', 'BoosterVersion_enc', 'Orbit_enc']]
y = spacex_df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)
print("Logistic Regression Accuracy:", log_accuracy)

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
print("Decision Tree Accuracy:", tree_accuracy)

# -----------------------------
# DASH APP STARTS
# -----------------------------

app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    
    dcc.Dropdown(
        id='site-dropdown',
        options=[{'label': 'All Sites', 'value': 'ALL'}] +
                [{'label': site, 'value': site} for site in spacex_df['LaunchSite'].unique()],
        value='ALL',
        placeholder="Select a Launch Site",
        searchable=True
    ),
    html.Br(),

    dcc.Graph(id='success-pie-chart'),
    html.Br(),

    html.P("Payload Range (kg):"),

    dcc.RangeSlider(
        id='payload-slider',
        min=min_payload,
        max=max_payload,
        step=100,
        marks={int(min_payload): str(int(min_payload)),
               int(max_payload): str(int(max_payload))},
        value=[min_payload, max_payload]
    ),
    html.Br(),

    dcc.Graph(id='success-payload-scatter-chart')
])

# Pie Chart Callback
@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def get_pie_chart(selected_site):

    if selected_site == 'ALL':
        fig = px.pie(
            spacex_df,
            names='LaunchSite',
            values='Class',
            title='Total Success Launches by Site'
        )
    else:
        filtered_df = spacex_df[spacex_df['LaunchSite'] == selected_site]
        fig = px.pie(
            filtered_df,
            names='Class',
            title=f'Success vs Failure at {selected_site}'
        )
    return fig

# Scatter Plot Callback
@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'),
     Input('payload-slider', 'value')]
)
def update_scatter(selected_site, payload_range):

    low, high = payload_range

    df = spacex_df[(spacex_df['PayloadMass'] >= low) & (spacex_df['PayloadMass'] <= high)]

    if selected_site != 'ALL':
        df = df[df['LaunchSite'] == selected_site]

    fig = px.scatter(
        df,
        x='PayloadMass',
        y='Class',
        color='BoosterVersion',
        title='Correlation Between Payload and Launch Outcome'
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
