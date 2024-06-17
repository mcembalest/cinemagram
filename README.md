# Cinemagram

## Cinemantic Trajectories

A `cinemagram` is a visualization of the progression of a video. I use the `nomic-embed-vision-1.5` encoder to embed each frame from the video as a 768-dimensional vector, PCA to reduce the vectors into two dimensions, and Plotly to interact with the rendered cinemagram.

## Making video frames

Convert a video into frames (`frame_0000.png`, `frame_0001.png`, etc) using ffmpeg:

```
ffmpeg -i input.mp4 -vf fps=1 output_dir/frame_%04d.png
```
