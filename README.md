# Cinemagram

## Cinemantic Trajectories

A `cinemagram` is a visualization of the frames of a video, grouping frames by visuo-semantic similarity. Here are four different 2-D cinemagrams for the same video, as well as accompanying 3-D cinemagrams sorted vertically by time. 

https://github.com/mcembalest/cinemagram/assets/70534565/ece8ae31-1647-4cb4-a4c2-cecda2e443ab

## How this was made

I use the `nomic-embed-vision-1.5` encoder to embed each video frame as a 768-dimensional vector, and compare a few ways of viewing the vectors in 2D. Each 3-D plot uses time as the z-axis, moving from the bottom to the top.

`Nomic_Top2` refers to just selecting the first two dimensions of the 768 dimensions. This is mostly noise without the other dimensions, even though the model was trained [resizable Matryoshka representations](https://blog.nomic.ai/posts/nomic-embed-matryoshka), since 2 dimensions is too much of a reduction (their website mentions 64 as the lowest useful number of dimensions).

The other three plots show three different dimensionality reduction techniques for converting the 768-D vectors into 2-D vectors: `PCA`, `t-SNE`, and `UMAP`. Each of these techniques has their benefits and limitations, but each can identify at least some visuo-semantic structure in the video: `PCA` roughly finds a cluster representing the credits sequence, `t-SNE` finds a clear cluster representing duck, and `UMAP` finds clear clusters representing red guy and the kitchen. 

To convert a video into frames (`frame_0000.png`, `frame_0001.png`, etc) I used ffmpeg:

```
ffmpeg -i input.mp4 -vf fps=1 output_dir/frame_%04d.png
```
