#!/usr/bin/bash


ffmpeg \
-framerate 10 -loop 1 -t 1.5 -i hippocampus.9.training.png \
-framerate 10 -loop 1 -t 1.5 -i hippocampus.5.training.png \
-framerate 10 -loop 1 -t 1.5 -i hippocampus.6.training.png \
-framerate 10 -loop 1 -t 1.5 -i hippocampus.9-inference.png \
-framerate 10 -loop 1 -t 1.5 -i hippocampus.5.inference.png \
-framerate 10 -loop 1 -t 1.5 -i hippocampus.5.inference.png \
-filter_complex \
"[1]format=rgba,fade=d=0.3:t=in:alpha=1,setpts=PTS-STARTPTS+4/TB[f0]; \
 [2]format=rgba,fade=d=0.3:t=in:alpha=1,setpts=PTS-STARTPTS+8/TB[f1]; \
 [3]format=rgba,fade=d=0.3:t=in:alpha=1,setpts=PTS-STARTPTS+12/TB[f2]; \
 [4]format=rgba,fade=d=0.3:t=in:alpha=1,setpts=PTS-STARTPTS+16/TB[f3]; \
 [5]format=rgba,fade=d=0.3:t=in:alpha=1,setpts=PTS-STARTPTS+20/TB[f4]; \
 [0][f0]overlay[bg1]; \
 [bg1][f1]overlay[bg2]; \
 [bg2][f2]overlay[bg3]; \
 [bg3][f3]overlay[bg4]; \
 [bg4][f4]overlay,split[v0][v1]; \
 [v0]palettegen[p];[v1][p]paletteuse[v]" -map "[v]" mnist.gif
