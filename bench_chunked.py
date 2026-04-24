#!/usr/bin/env python3
"""
Chunked-pipeline benchmark.

Three strategies for sending a video to the model:

  serial   – pre-split chunks sent one after another (chunks-root required)
  parallel – pre-split chunks fired concurrently    (chunks-root required)
  whole    – entire video sent as a single video_url request (--video required;
             vLLM handles frame extraction internally via --media-io-kwargs)

All three modes report the same set of metrics so results are directly
comparable:
  latency mode   – TTFT, E2E, wall-clock per video (serial+parallel also show
                   per-chunk breakdown)
  throughput mode – videos/s, req/s, input tok/s, output tok/s, p50/p95/p99
                   wall-clock per video

Dataset mode (--videos-dir):
  Iterates over every .mp4 in the given folder using the whole strategy,
  prints per-video metrics, then prints aggregate stats across the dataset.

For --strategy whole the server must be launched with:
  --allowed-local-media-path <dir>   (if using a file:// URL)
  --max-num-batched-tokens <N>       (>= num_frames * tokens_per_frame to avoid
                                      running the vision encoder twice per req)
"""
import argparse, asyncio, base64, glob, json, os, statistics, time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx

def load_chunk(d: str) -> List[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(d, "frame_*.jpg"))):
        with open(p, "rb") as f:
            out.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode())
    return out

def video_file_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://", "file://", "data:")):
        return path_or_url
    return "file://" + os.path.abspath(path_or_url)

@dataclass
class R:
    ok: bool; ttft: float = 0.0; e2e: float = 0.0
    input_tokens: int = 0; output_tokens: int = 0; err: str = ""; text: str = ""

async def one(c, base, model, frames, prompt, max_toks, capture=False):
    """Send one chat request with frames as image_url items."""
    body = {"model": model, "max_tokens": max_toks, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content":
                          [{"type": "text", "text": prompt}] +
                          [{"type": "image_url", "image_url": {"url": f}} for f in frames]}]}
    return await _post_streaming(c, base, body, capture)

async def one_video(c, base, model, video_url, num_frames, prompt, max_toks, capture=False):
    """Send one chat request with the full video as a video_url item.

    num_frames is forwarded to vLLM's media_io_kwargs so the server samples
    exactly that many frames uniformly.  Set num_frames=-1 to use the server
    default.
    """
    media_kwargs = {} if num_frames == -1 else {"video": {"num_frames": num_frames}}
    body = {"model": model, "max_tokens": max_toks, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": video_url}},
            ]}]}
    if media_kwargs:
        body["media_io_kwargs"] = media_kwargs
    return await _post_streaming(c, base, body, capture)


async def _post_streaming(c, base, body, capture):
    t0 = time.perf_counter(); ttft = None; inp = 0; out = 0; buf = []
    try:
        async with c.stream("POST", f"{base}/v1/chat/completions",
                            json=body, timeout=600.0) as r:
            if r.status_code != 200:
                return R(ok=False, err=f"HTTP {r.status_code}: {(await r.aread())[:200]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try: j = json.loads(d)
                except json.JSONDecodeError: continue
                if j.get("usage"):
                    u = j["usage"]; inp = u.get("prompt_tokens", inp); out = u.get("completion_tokens", out)
                ch = j.get("choices") or []
                if ch:
                    dd = ch[0].get("delta") or {}
                    piece = dd.get("content") or dd.get("reasoning")
                    if piece:
                        if ttft is None: ttft = time.perf_counter() - t0
                        if capture: buf.append(piece)
    except Exception as e:
        return R(ok=False, err=f"{type(e).__name__}: {e}")
    return R(ok=True, ttft=ttft or 0.0, e2e=time.perf_counter() - t0,
             input_tokens=inp, output_tokens=out, text="".join(buf))


async def run_video_serial(c, base, model, chunks, prompt, max_toks, capture=False):
    t0 = time.perf_counter()
    per_chunk: List[R] = []
    for frames in chunks:
        per_chunk.append(await one(c, base, model, frames, prompt, max_toks, capture))
    return time.perf_counter() - t0, per_chunk


async def run_video_parallel(c, base, model, chunks, prompt, max_toks, capture=False):
    t0 = time.perf_counter()
    tasks = [asyncio.create_task(one(c, base, model, f, prompt, max_toks, capture))
             for f in chunks]
    per_chunk: List[R] = await asyncio.gather(*tasks)
    return time.perf_counter() - t0, per_chunk


async def run_video_whole(c, base, model, video_url, num_frames, prompt, max_toks, capture=False):
    t0 = time.perf_counter()
    r = await one_video(c, base, model, video_url, num_frames, prompt, max_toks, capture)
    return time.perf_counter() - t0, [r]

async def start_profile(c: httpx.AsyncClient, base_url: str) -> None:
    print("Starting profiler...")
    r = await c.post(f"{base_url}/start_profile")
    print("Profiler started" if r.status_code == 200 else f"Failed to start profiler: HTTP {r.status_code}")


async def stop_profile(c: httpx.AsyncClient, base_url: str) -> None:
    print("Stopping profiler...")
    r = await c.post(f"{base_url}/stop_profile")
    print("Profiler stopped" if r.status_code == 200 else f"Failed to stop profiler: HTTP {r.status_code}")


def pct(xs, p):
    xs = sorted(xs)
    return 0.0 if not xs else xs[max(0, min(len(xs)-1, int(round(p/100*(len(xs)-1)))))]


def _print_aggregate(label: str, values: List[float], fmt: str = ".2f") -> None:
    if not values:
        return
    mean = statistics.mean(values)
    print(f"  {label:<18} mean={mean:{fmt}}  "
          f"p50={pct(values,50):{fmt}}  p95={pct(values,95):{fmt}}  p99={pct(values,99):{fmt}}")


def _print_throughput(results, total_wall):
    ok_videos = [r for r in results if all(p.ok for p in r[1])]
    if not ok_videos:
        print("  NO fully-OK videos")
        for _, per in results[:2]:
            for p in per:
                if not p.ok: print(f"  err: {p.err}")
        return
    video_walls = [w for w, _ in ok_videos]
    all_reqs    = [p for _, per in ok_videos for p in per]
    tot_in  = sum(p.input_tokens  for p in all_reqs)
    tot_out = sum(p.output_tokens for p in all_reqs)
    print(f"  videos_ok={len(ok_videos)} (requests={len(all_reqs)}) wall={total_wall:.1f}s")
    print(f"  videos/s        = {len(ok_videos)/total_wall:.2f}")
    print(f"  req/s           = {len(all_reqs)/total_wall:.2f}")
    print(f"  input  tok/s    = {tot_in/total_wall:.0f}")
    print(f"  output tok/s    = {tot_out/total_wall:.0f}")
    print(f"  per-video wall  p50/p95/p99 = "
          f"{pct(video_walls,50):.2f} / {pct(video_walls,95):.2f} / {pct(video_walls,99):.2f} s")

async def dataset_mode(args, video_paths: List[str]) -> None:
    print(f"\n=== DATASET LATENCY ({len(video_paths)} videos, "
          f"num_frames={args.num_frames}, max_tokens={args.max_tokens}) ===")
    print(f"  prompt: {args.prompt!r}\n")

    # (name, wall, R)
    rows: List[Tuple[str, float, R]] = []

    async with httpx.AsyncClient() as c:
        if args.profile:
            await start_profile(c, args.base_url)

        for path in video_paths:
            name = os.path.basename(path)
            url  = video_file_url(path)
            wall, per = await run_video_whole(
                c, args.base_url, args.model, url,
                args.num_frames, args.prompt, args.max_tokens,
                capture=True,
            )
            r = per[0]
            rows.append((name, wall, r))
            if r.ok:
                print(f"  {name}")
                print(f"    in_tok={r.input_tokens:5d}  out_tok={r.output_tokens:4d}  "
                      f"TTFT={r.ttft*1000:7.1f}ms  E2E={r.e2e:6.2f}s  wall={wall:.2f}s")
                if r.text:
                    print(f"    -> {r.text.strip()[:160]!r}")
            else:
                print(f"  {name}  FAILED: {r.err}")

        if args.profile:
            await stop_profile(c, args.base_url)

    ok = [(n, w, r) for n, w, r in rows if r.ok]
    n_ok, n_total = len(ok), len(rows)
    print(f"\n=== AGGREGATE ({n_ok}/{n_total} ok) ===")
    if not ok:
        return

    ttfts_ms = [r.ttft * 1000 for _, _, r in ok]
    e2es     = [r.e2e          for _, _, r in ok]
    walls    = [w              for _, w, _ in ok]
    in_toks  = [r.input_tokens  for _, _, r in ok]
    out_toks = [r.output_tokens for _, _, r in ok]

    _print_aggregate("TTFT (ms)",   ttfts_ms, fmt=".1f")
    _print_aggregate("E2E (s)",     e2es)
    _print_aggregate("wall (s)",    walls)
    _print_aggregate("input tokens",  in_toks,  fmt=".0f")
    _print_aggregate("output tokens", out_toks, fmt=".0f")

async def latency_mode(args, chunks):
    if args.strategy == "whole":
        print(f"\n=== WHOLE-VIDEO LATENCY (1 request, num_frames={args.num_frames}, "
              f"max_tokens={args.max_tokens}) ===")
        print(f"  video: {args.video_url}")
        async with httpx.AsyncClient() as c:
            if args.profile:
                await start_profile(c, args.base_url)
            wall, per = await run_video_whole(c, args.base_url, args.model, args.video_url,
                                              args.num_frames, args.prompt, args.max_tokens,
                                              capture=True)
            r = per[0]
            if r.ok:
                print(f"  in_tok={r.input_tokens:5d} out_tok={r.output_tokens:4d} "
                      f"TTFT={r.ttft*1000:6.1f}ms E2E={r.e2e:5.2f}s "
                      f"wall={wall:.2f}s")
                print(f"  -> {r.text.strip()[:240]!r}")
            else:
                print(f"  FAILED: {r.err}")
            if args.profile:
                await stop_profile(c, args.base_url)
        return

    print(f"\n=== PER-CHUNK LATENCY (1 video, {len(chunks)} chunks, max_tokens={args.max_tokens}) ===")
    async with httpx.AsyncClient() as c:
        if args.profile:
            await start_profile(c, args.base_url)

        wall, per = await run_video_serial(c, args.base_url, args.model, chunks,
                                           args.prompt, args.max_tokens, capture=True)
        for i, r in enumerate(per):
            print(f"  chunk {i}: in_tok={r.input_tokens:5d} out_tok={r.output_tokens:4d} "
                  f"TTFT={r.ttft*1000:6.1f}ms E2E={r.e2e:5.2f}s -> "
                  f"{r.text.strip()[:120]!r}")
        print(f"  SERIAL   wall-clock per video = {wall:.2f}s")

        wall_p, per_p = await run_video_parallel(c, args.base_url, args.model, chunks,
                                                 args.prompt, args.max_tokens)
        ttfts = [r.ttft for r in per_p]; e2es = [r.e2e for r in per_p]
        print(f"  PARALLEL wall-clock per video = {wall_p:.2f}s  "
              f"(chunk TTFT p50/p99={pct(ttfts,50)*1000:.0f}/{pct(ttfts,99)*1000:.0f}ms, "
              f"E2E max={max(e2es):.2f}s)")

        if args.profile:
            await stop_profile(c, args.base_url)


async def throughput_mode(args, chunks):
    if args.strategy == "whole":
        print(f"\n=== VIDEO THROUGHPUT (conc={args.concurrency} videos, dur={args.duration}s, "
              f"strategy=whole, num_frames={args.num_frames}) ===")
        conn_limit = args.concurrency * 2
    else:
        print(f"\n=== VIDEO THROUGHPUT (conc={args.concurrency} videos, dur={args.duration}s, "
              f"strategy={args.strategy}, chunks_per_video={len(chunks)}) ===")
        conn_limit = args.concurrency * len(chunks) * 2

    results: List[Tuple[float, List[R]]] = []
    stop_at = time.perf_counter() + args.duration
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=conn_limit,
            max_keepalive_connections=conn_limit)) as c:

        async def worker():
            while time.perf_counter() < stop_at:
                async with sem:
                    if args.strategy == "whole":
                        wall, per = await run_video_whole(
                            c, args.base_url, args.model, args.video_url,
                            args.num_frames, args.prompt, args.max_tokens)
                    elif args.strategy == "serial":
                        wall, per = await run_video_serial(
                            c, args.base_url, args.model, chunks,
                            args.prompt, args.max_tokens)
                    else:
                        wall, per = await run_video_parallel(
                            c, args.base_url, args.model, chunks,
                            args.prompt, args.max_tokens)
                    results.append((wall, per))

        if args.profile:
            await start_profile(c, args.base_url)
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        await asyncio.gather(*tasks, return_exceptions=True)
        total_wall = time.perf_counter() - t0
        if args.profile:
            await stop_profile(c, args.base_url)

    _print_throughput(results, total_wall)

async def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="internvl3_5-8b")
    p.add_argument("--mode", choices=["latency", "throughput"],
                   help="required unless --videos-dir is used")
    p.add_argument("--strategy", choices=["serial", "parallel", "whole"], default="parallel",
                   help="serial/parallel: use pre-split frame chunks (--chunks-root); "
                        "whole: send entire video as video_url (--video / --videos-dir)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="concurrent videos in throughput mode")
    p.add_argument("--duration", type=float, default=45.0,
                   help="throughput mode window in seconds")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt", default="Briefly describe what is happening in this video segment.")
    p.add_argument("--profile", action="store_true",
                   help="enable nsys profiling via /start_profile and /stop_profile endpoints")

    # chunk-based args
    p.add_argument("--chunks-root", default="./video_chunks",
                   help="directory of chunk_* subdirs (serial/parallel strategies)")

    # whole-video args
    p.add_argument("--video", default=None,
                   help="video file path or URL for --strategy whole (single video)")
    p.add_argument("--videos-dir", default=None,
                   help="directory of .mp4 files; iterate over all with whole strategy "
                        "and report per-video + aggregate metrics (dataset mode)")
    p.add_argument("--num-frames", type=int, default=32,
                   help="frames to sample per video (whole strategy); -1 = server default")

    a = p.parse_args()

    if a.videos_dir:
        video_paths = sorted(
            os.path.join(a.videos_dir, f)
            for f in os.listdir(a.videos_dir)
            if f.lower().endswith(".mp4")
        )
        if not video_paths:
            raise SystemExit(f"no .mp4 files found in {a.videos_dir}")
        print(f"[dataset mode: {len(video_paths)} videos in {a.videos_dir}]")
        await dataset_mode(a, video_paths)
        return

    if not a.mode:
        p.error("--mode is required when --videos-dir is not used")

    if a.strategy == "whole":
        if not a.video:
            p.error("--video is required when --strategy whole")
        a.video_url = video_file_url(a.video)
        chunks = []
        print(f"[whole-video mode: {a.video_url}, num_frames={a.num_frames}]")
    else:
        a.video_url = None
        chunks = [load_chunk(d) for d in sorted(glob.glob(f"{a.chunks_root}/chunk_*"))]
        if not chunks:
            raise SystemExit(f"no chunks found in {a.chunks_root}")
        print(f"[loaded {len(chunks)} chunks, frame counts: {[len(c) for c in chunks]}]")

    if a.mode == "latency":
        await latency_mode(a, chunks)
    else:
        await throughput_mode(a, chunks)


asyncio.run(main())
