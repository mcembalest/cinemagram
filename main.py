import numpy as np
import os
import pandas as pd
import plotly.express as px
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

nomic_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")

source_dir = "data/dhmis4"

if os.path.exists(f"{source_dir}/gram.npy"):
    with open(f"{source_dir}/gram.npy", "rb") as f:
        gram = np.load(f)
else:
    gram = []
    for filename in os.listdir(source_dir):
        if "png" in filename:
            idx = filename.split("_")[1].split(".png")[0]
            print("Encoding frame", idx)
            img_embs = nomic_model(** processor(Image.open(f"{source_dir}/{filename}"), return_tensors="pt"))
            gram.append(F.normalize(img_embs.last_hidden_state[:, 0], p=2, dim=1).detach().numpy().flatten())
    with open(f'{source_dir}/gram.npy', 'wb') as f:
        np.save(f, gram)

pca = PCA(n_components=2)
pca.fit(gram)
gram_2d = pca.transform(gram)
df = pd.DataFrame({"frame" : np.arange(len(gram)), "dim1" : gram_2d[:,0], "dim2" : gram_2d[:,1]})
fig = px.line_3d(x=df['dim1'], y=df['dim2'], z=df['frame'])
fig.show()