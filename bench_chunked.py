#!/usr/bin/env python3
"""
Chunked-pipeline benchmark.

Given a directory of chunk subdirs each containing frame JPEGs, this sends
one chat request per chunk and measures:
  - per-chunk latency (TTFT, E2E)
  - per-video wall clock  (serial vs parallel firing of the chunks)
  - throughput at various concurrent-video levels

Matches the production pattern where a long video is pre-split into 16-s
chunks sampled at 2 fps (e.g. 32 frames each), and each chunk gets its own
model call.
"""
import argparse, asyncio, base64, glob, json, os, statistics, time
from dataclasses import dataclass
from typing import List, Tuple

import httpx


def load_chunk(d: str) -> List[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(d, "frame_*.jpg"))):
        with open(p, "rb") as f:
            out.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode())
    return out


@dataclass
class R:
    ok: bool; ttft: float = 0.0; e2e: float = 0.0
    input_tokens: int = 0; output_tokens: int = 0; err: str = ""; text: str = ""


async def one(c, base, model, frames, prompt, max_toks, capture=False):
    body = {"model": model, "max_tokens": max_toks, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role":"user","content":
                          [{"type":"text","text":prompt}] +
                          [{"type":"image_url","image_url":{"url":f}} for f in frames]}]}
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


def pct(xs, p):
    xs = sorted(xs)
    return 0.0 if not xs else xs[max(0, min(len(xs)-1, int(round(p/100*(len(xs)-1)))))]


async def start_profile(c: httpx.AsyncClient, base_url: str) -> None:
    print("Starting profiler...")
    profile_input = {"api_url": f"{base_url}/start_profile"}
    print(f"profile_input: {profile_input}")
    r = await c.post(profile_input["api_url"])
    if r.status_code == 200:
        print("Profiler started")
    else:
        print(f"Failed to start profiler: HTTP {r.status_code}")


async def stop_profile(c: httpx.AsyncClient, base_url: str) -> None:
    print("Stopping profiler...")
    profile_input = {"api_url": f"{base_url}/stop_profile"}
    print(f"profile_input: {profile_input}")
    r = await c.post(profile_input["api_url"])
    if r.status_code == 200:
        print("Profiler stopped")
    else:
        print(f"Failed to stop profiler: HTTP {r.status_code}")


async def run_video_serial(c, base, model, chunks, prompt, max_toks, capture=False):
    """Send chunks one after another. Wall-clock = sum of per-chunk E2E."""
    t0 = time.perf_counter()
    per_chunk: List[R] = []
    for frames in chunks:
        per_chunk.append(await one(c, base, model, frames, prompt, max_toks, capture))
    wall = time.perf_counter() - t0
    return wall, per_chunk


async def run_video_parallel(c, base, model, chunks, prompt, max_toks, capture=False):
    """Fire all chunks of one video concurrently. Wall-clock = max per-chunk E2E (+ scheduler queueing)."""
    t0 = time.perf_counter()
    tasks = [asyncio.create_task(one(c, base, model, f, prompt, max_toks, capture))
             for f in chunks]
    per_chunk: List[R] = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    return wall, per_chunk


async def latency_mode(args, chunks):
    print(f"\n=== PER-CHUNK LATENCY (1 video, {len(chunks)} chunks, max_tokens={args.max_tokens}) ===")
    async with httpx.AsyncClient() as c:
        if args.profile:
            await start_profile(c, args.base_url)
        # First video serial — print per-chunk reply so user sees the model understood each chunk
        wall, per = await run_video_serial(c, args.base_url, args.model, chunks,
                                           args.prompt, args.max_tokens, capture=True)
        for i, r in enumerate(per):
            size = (r.input_tokens or 0)
            print(f"  chunk {i}: in_tok={size:5d} out_tok={r.output_tokens:4d} "
                  f"TTFT={r.ttft*1000:6.1f}ms E2E={r.e2e:5.2f}s -> "
                  f"{r.text.strip()[:120]!r}")
        print(f"  SERIAL   wall-clock per video = {wall:.2f}s")

        # Same video, parallel
        wall_p, per_p = await run_video_parallel(c, args.base_url, args.model, chunks,
                                                 args.prompt, args.max_tokens)
        ttfts = [r.ttft for r in per_p]; e2es = [r.e2e for r in per_p]
        print(f"  PARALLEL wall-clock per video = {wall_p:.2f}s  "
              f"(chunk TTFT p50/p99={pct(ttfts,50)*1000:.0f}/{pct(ttfts,99)*1000:.0f}ms, "
              f"E2E max={max(e2es):.2f}s)")

        if args.profile:
            await stop_profile(c, args.base_url)


async def throughput_mode(args, chunks):
    print(f"\n=== VIDEO THROUGHPUT (conc={args.concurrency} videos, dur={args.duration}s, "
          f"strategy={args.strategy}, chunks_per_video={len(chunks)}) ===")
    results: List[Tuple[float, List[R]]] = []
    stop_at = time.perf_counter() + args.duration
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency*len(chunks)*2,
                                                      max_keepalive_connections=args.concurrency*len(chunks)*2)) as c:
        async def worker():
            while time.perf_counter() < stop_at:
                async with sem:
                    if args.strategy == "serial":
                        wall, per = await run_video_serial(c, args.base_url, args.model, chunks,
                                                           args.prompt, args.max_tokens)
                    else:
                        wall, per = await run_video_parallel(c, args.base_url, args.model, chunks,
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

    ok_videos = [r for r in results if all(p.ok for p in r[1])]
    if not ok_videos:
        print(" NO fully-OK videos")
        for _, per in results[:2]:
            for p in per:
                if not p.ok: print(f"  err: {p.err}")
        return
    video_walls = [w for w, _ in ok_videos]
    all_chunks = [p for _, per in ok_videos for p in per]
    tot_in = sum(p.input_tokens for p in all_chunks)
    tot_out = sum(p.output_tokens for p in all_chunks)
    print(f"  videos_ok={len(ok_videos)} (chunks={len(all_chunks)}) wall={total_wall:.1f}s")
    print(f"  videos/s        = {len(ok_videos)/total_wall:.2f}")
    print(f"  chunk req/s     = {len(all_chunks)/total_wall:.2f}")
    print(f"  input  tok/s    = {tot_in/total_wall:.0f}")
    print(f"  output tok/s    = {tot_out/total_wall:.0f}")
    print(f"  per-video wall  p50/p95/p99 = {pct(video_walls,50):.2f} / {pct(video_walls,95):.2f} / {pct(video_walls,99):.2f} s")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="internvl3_5-8b")
    p.add_argument("--chunks-root", default="./video_chunks")
    p.add_argument("--mode", choices=["latency","throughput"], required=True)
    p.add_argument("--strategy", choices=["serial","parallel"], default="parallel",
                   help="how to fire chunks within a single video")
    p.add_argument("--concurrency", type=int, default=1, help="concurrent videos in throughput mode")
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt", default="Briefly describe what is happening in this video segment.")
    p.add_argument("--profile", action="store_true",
                   help="Enable nsys profiling via /start_profile and /stop_profile endpoints.")
    a = p.parse_args()
    chunks = [load_chunk(d) for d in sorted(glob.glob(f"{a.chunks_root}/chunk_*"))]
    if not chunks:
        raise SystemExit(f"no chunks in {a.chunks_root}")
    print(f"[loaded {len(chunks)} chunks, frame counts: {[len(c) for c in chunks]}]")
    if a.mode == "latency":
        await latency_mode(a, chunks)
    else:
        await throughput_mode(a, chunks)


asyncio.run(main())
