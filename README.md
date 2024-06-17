# Cinemagram

## Cinemantic Trajectories

A `cinemagram` is a visualization of the progression of a video. I use the `nomic-embed-vision-1.5` encoder to embed each video frame as a 768-dimensional vector, PCA to reduce the vectors into two dimensions, and arrange the vectors over time:



https://github.com/mcembalest/cinemagram/assets/70534565/1048da5e-88a7-4e37-bbc5-b1200c7b4556



## Making video frames

To convert a video into frames (`frame_0000.png`, `frame_0001.png`, etc) I used ffmpeg:

```
ffmpeg -i input.mp4 -vf fps=1 output_dir/frame_%04d.png
```
