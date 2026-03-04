from graphviz import Digraph

dot = Digraph(comment='Global Financial Crisis Prediction System')

dot.attr(rankdir='LR', size='16,8', dpi='300')

# DATA LAYER
with dot.subgraph(name='cluster_data') as c:
    c.attr(label='Data Layer', style='filled', color='lightgrey')
    c.node('A', 'Macroeconomic Data\n(FRED API)', shape='box', style='filled', fillcolor='#AED6F1')

# PROCESSING LAYER
with dot.subgraph(name='cluster_processing') as c:
    c.attr(label='Data Processing Layer', style='filled', color='lightgrey')
    c.node('B', 'Data Cleaning\nNormalization', shape='box', style='filled', fillcolor='#F9E79F')
    c.node('C', 'Feature Engineering\nTime Series Features', shape='box', style='filled', fillcolor='#F9E79F')

# MODEL LAYER
with dot.subgraph(name='cluster_model') as c:
    c.attr(label='AI Model Layer', style='filled', color='lightgrey')
    c.node('D', 'LSTM Deep Learning Model', shape='box', style='filled', fillcolor='#ABEBC6')

# PREDICTION LAYER
with dot.subgraph(name='cluster_prediction') as c:
    c.attr(label='Prediction Layer', style='filled', color='lightgrey')
    c.node('E', 'Crisis Probability\nPrediction', shape='box', style='filled', fillcolor='#F5B041')

# APPLICATION LAYER
with dot.subgraph(name='cluster_app') as c:
    c.attr(label='Application Layer', style='filled', color='lightgrey')
    c.node('F', 'Streamlit Dashboard\nVisualization', shape='box', style='filled', fillcolor='#F5B7B1')

# Connections
dot.edge('A','B')
dot.edge('B','C')
dot.edge('C','D')
dot.edge('D','E')
dot.edge('E','F')

dot.render('architecture', format='png', cleanup=True)

print("Professional AI architecture diagram created!")