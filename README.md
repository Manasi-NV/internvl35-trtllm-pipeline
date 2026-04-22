# InternVL3.5-8B FP8 Video Benchmark

One-file benchmark that measures **per-video latency** and **videos/second throughput** for the chunked-pipeline pattern (long video split into ~16 s chunks, each sent as a separate chat request with N sampled frames). Designed for NVIDIA Hopper/Blackwell GPUs.

## Prereqs

- NVIDIA GPU with compute capability ≥ 9.0 (H100 / H200 / B200) — FP8 support
- `uv`, `ffmpeg`
- A video file to benchmark against (any H.264/H.265 mp4 works)

## 1. Install vLLM and download the model

```bash
uv venv --python 3.12 vllm-env
source vllm-env/bin/activate
uv pip install -U vllm --torch-backend cu128 Pillow httpx
uv tool install "huggingface_hub[cli]"
hf download brandonbeiler/InternVL3_5-8B-FP8-Dynamic
```

## 2. Start the vLLM server

**Alerting / novel-video workload (recommended for capacity planning):**
```bash
vllm serve brandonbeiler/InternVL3_5-8B-FP8-Dynamic \
  --served-model-name internvl3_5-8b \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 32}' \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

**Interactive / repeat-query workload (caches on):** drop the last two flags and add `--mm-processor-cache-gb 8`.

Wait for `Application startup complete.` in the logs before running the benchmark.

## 3. Prepare chunks from your video

Chunks are just directories of JPEGs, one per sampled frame. This example splits a video into 16-second chunks at 2 fps (32 frames each), letter-boxed to 448×448:

```bash
VID=/path/to/your_video.mp4
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
```

## 4. Run the benchmark

**Per-video latency** (single video, serial vs parallel chunk firing, prints model's reply for each chunk):

```bash
python bench_chunked.py --mode latency
```

**Throughput sweep** (multiple concurrent videos, reports videos/s and per-video p50/p95/p99):

```bash
for c in 1 2 4 8; do
  python bench_chunked.py --mode throughput --strategy parallel --concurrency $c --duration 45
done
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | required | `latency` or `throughput` |
| `--strategy` | `parallel` | `parallel` fires all chunks at once; `serial` fires them sequentially |
| `--concurrency` | 1 | Number of full videos in flight at once (throughput mode) |
| `--duration` | 45 | Throughput benchmark window, seconds |
| `--max-tokens` | 128 | Output token cap per chunk. Drop to 16 for yes/no alerting prompts. |
| `--prompt` | generic | The text prompt sent with each chunk's frames |
| `--chunks-root` | `./video_chunks` | Directory containing `chunk_0/`, `chunk_1/`, … subdirs |
| `--model` | `internvl3_5-8b` | Must match server's `--served-model-name` |
| `--base-url` | `http://127.0.0.1:8000` | vLLM server address |

## Reading the output

- **Videos/s** — sustained throughput, complete videos per second across the run.
- **Per-video p50** — median wall-clock time per video (submission → final token).
- **Input tok/s** — GPU's raw input-token processing rate. On H200 with FP8 this tops out near 25k tok/s with caches off.

The throughput **knee** is typically at concurrency 4 — past that, videos/s barely grows while per-video latency grows linearly. Plan production capacity at the knee.

## Current numbers (H200, 3 × 32-frame chunks, caches off)

| Concurrency | Videos/s | Per-video p50 |
|---:|---:|---:|
| 1 | 0.72 | 1.34 s |
| 4 | **1.06** | 3.89 s |
| 8 | 1.17 | 7.20 s |

Rule of thumb: **~1 video/s per H200** for novel 40-second clips at this chunking.
