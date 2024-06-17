print("loading...")
import dash
from dash import html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import numpy as np
import os
from PIL import Image
import base64
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch
from transformers import AutoModel, AutoImageProcessor

app = dash.Dash(__name__)

nomic_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")

source_dir = "data/dhmis4"  # Update your directory as needed
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

# PCA, t-SNE, UMAP transformations
print("pca...")
gram_pca = PCA(n_components=2).fit_transform(gram)
print("tsne...")
gram_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(gram)
print("umap...")
gram_umap = umap.UMAP().fit_transform(gram)
print("plotting...")
# Creating the plots
fig = make_subplots(rows=2, cols=4, column_widths=[0.25, 0.25, 0.25, 0.25],
                    specs=[[{"colspan": 4}, None, None, None], [{}]*4],
                    subplot_titles=("Scrubber", "Nomic_Top2", "Nomic_PCA", "Nomic_t-SNE", "Nomic_UMAP"))
scrubber = go.Scatter(x=frame_indices, y=np.zeros(len(frame_indices)), mode='markers', marker=dict(size=5), name="Scrubber")
fig.add_trace(scrubber, row=1, col=1)

for i, (data, name) in enumerate(zip([gram, gram_pca, gram_tsne, gram_umap], ["Nomic_Top2", "Nomic_PCA", "Nomic_t-SNE", "Nomic_UMAP"]), start=1):
    scatter = go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers', name=name)
    fig.add_trace(scatter, row=2, col=i)

fig.update_layout(height=800, showlegend=False)

@app.callback(
    Output('image-display', 'src'),
    [Input('main-graph', 'hoverData')]
)
def display_image(hoverData):
    if hoverData is not None:
        point_index = hoverData['points'][0]['pointIndex']
        image_path = os.path.join(source_dir, image_filenames[point_index])
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        return f"data:image/png;base64,{encoded_image}"
    return None

app.layout = html.Div([
    dcc.Graph(id='main-graph', figure=fig),
    html.Img(id='image-display', style={'max-height': '300px', 'max-width': '300px'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
