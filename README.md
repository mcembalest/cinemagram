# Cinemagram

## Cinemantic Trajectories

A `cinemagram` is a visualization of the progression of a video. I use the `nomic-embed-vision-1.5` encoder to embed each video frame as a 768-dimensional vector, and view the cinemagram using a few different techniques reducing the vectors to 2D:



https://github.com/mcembalest/cinemagram/assets/70534565/ece8ae31-1647-4cb4-a4c2-cecda2e443ab









## Making video frames

To convert a video into frames (`frame_0000.png`, `frame_0001.png`, etc) I used ffmpeg:

```
ffmpeg -i input.mp4 -vf fps=1 output_dir/frame_%04d.png
```
