#!/usr/bin/env bash
set -euo pipefail

VID="${1:?Usage: $0 <video_file>}"

# 2 fps × 16 s = 32 frames per chunk (last chunk may have fewer)
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VID" | awk '{print int($1)}')

rm -rf video_chunks && mkdir -p video_chunks
i=0
for start in $(seq 0 16 $DUR); do
  mkdir -p video_chunks/chunk_$i
  ffmpeg -y -loglevel error -ss $start -t 16 -i "$VID" \
    -vf "fps=2,scale=448:448:force_original_aspect_ratio=decrease,pad=448:448:(ow-iw)/2:(oh-ih)/2" \
    -q:v 3 video_chunks/chunk_$i/frame_%03d.jpg
  i=$((i+1))
done
