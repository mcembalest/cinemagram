print("loading...")
import dash
from dash import html, dcc, callback_context
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import numpy as np
import os
from PIL import Image
import base64
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import umap
import torch
from transformers import AutoModel, AutoImageProcessor

app = dash.Dash(__name__)

nomic_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")

source_dir = "data/dhmis4"
image_filenames = sorted([f for f in os.listdir(source_dir) if f.endswith(".png")])
frame_indices = np.arange(len(image_filenames))

if os.path.exists(f"{source_dir}/gram.npy"):
    gram = np.load(f"{source_dir}/gram.npy")
else:
    gram = []
    for filename in image_filenames:
        print(f"Encoding frame {filename}")
        img = Image.open(f"{source_dir}/{filename}")
        img_emb = nomic_model(**processor(img, return_tensors="pt"))
        gram.append(torch.nn.functional.normalize(img_emb.last_hidden_state[:, 0], p=2, dim=1).detach().numpy().flatten())
    np.save(f"{source_dir}/gram.npy", gram)

print("PCA, tSNE, and UMAP...")
reduc_titles = ["Nomic_Top2", "Nomic_PCA", "Nomic_t-SNE", "Nomic_UMAP"]
colors = ['red', 'blue', 'green', 'purple']
gram_pca = PCA(n_components=2).fit_transform(gram)
gram_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(gram)
gram_umap = umap.UMAP().fit_transform(gram)
grams = [gram, gram_pca, gram_tsne, gram_umap]

print("plotting...")
fig = make_subplots(
    rows=3, 
    cols=4, 
    column_widths=[0.25, 0.25, 0.25, 0.25], 
    row_heights = [0.1, 0.2, 0.6],
    specs=[[{"colspan": 4}, None, None, None], [{}]*4, [{'type': 'scene'}]*4],
    subplot_titles=["Film Time-Sorted"] + reduc_titles,
    horizontal_spacing=0.02,
    vertical_spacing=0.05 
)

fig.add_trace(
    go.Scatter(
        x=frame_indices, 
        y=np.zeros(len(frame_indices)), 
        mode='markers', 
        marker=dict(size=5), 
        name="Film Time-Sorted",
        customdata=frame_indices,
        hoverinfo='text', 
        text=[f'Frame {idx}' for idx in frame_indices]
    ), 
    row=1, col=1)

scaler = MinMaxScaler()
for i, (data_og, name, color) in enumerate(zip(grams, reduc_titles, colors)):
    data = scaler.fit_transform(data_og)
    
    fig.add_trace(go.Scatter(
        x=data[:, 0], 
        y=data[:, 1], 
        mode='markers', 
        name=name, 
        customdata=frame_indices, 
        marker=dict(size=5, color=color), 
        hoverinfo='text', 
        text=[f'Frame {idx}' for idx in frame_indices]
    ), row=2, col=i+1)

    fig.add_trace(go.Scatter3d(
        x=data[:, 0], 
        y=data[:, 1], 
        z=frame_indices, 
        mode='markers+lines', 
        name=name, 
        customdata=frame_indices, 
        marker=dict(size=5, color=color), 
        hoverinfo='text', 
        text=[f'Frame {idx}' for idx in frame_indices]
    ), row=3, col=i+1)

start_angles = [{"eye" : {"x":0.5,"y":2.5,"z":0.5}}]*4

fig.update_layout(
    width=1250, 
    height=1000, 
    showlegend=False,
    scene_camera = start_angles[0],
    scene2_camera = start_angles[0],
    scene3_camera = start_angles[0],
    scene4_camera = start_angles[0],
    margin=dict(l=10, r=10, t=50, b=10)
)

fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_scenes(
    xaxis=dict(showticklabels=False, title=''),
    yaxis=dict(showticklabels=False, title=''),
    zaxis=dict(showticklabels=False, title='Frame')
)

@app.callback(
    Output('main-graph', 'figure'),
    [Input('main-graph', 'hoverData')],
    [State('main-graph', 'figure')]
)
def update_marker_size(hoverData, fig):
    if hoverData:
        point_index = hoverData['points'][0]['customdata']
        for trace in fig['data']:
            if 'customdata' in trace:
                sizes = [20 if idx == point_index else 5 for idx in trace['customdata']]
                trace['marker']['size'] = sizes
    return fig

@app.callback(
    Output('image-display', 'src'),
    [Input('main-graph', 'hoverData')]
)
def display_image(hoverData):
    if hoverData is not None:
        image_path = os.path.join(source_dir, image_filenames[hoverData['points'][0]['customdata']])
    else:
        image_path = "data/start.png"
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
    return f"data:image/png;base64,{encoded_image}"

app.layout = html.Div([
    html.Div([
        html.Img(id='image-display', style={'height': '300px', 'width': '400px'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
    dcc.Graph(id='main-graph', figure=fig)
], style={'display': 'flex', 'flex-direction': 'row'})
print("ready")
if __name__ == '__main__':
    app.run_server(debug=True)
    
