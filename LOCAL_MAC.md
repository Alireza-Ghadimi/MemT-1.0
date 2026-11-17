# MemT-1.0-Nano on this Mac Studio

This directory contains the upstream MemT-1.0 text source at Git tree
`9ce3ec690ce9d0ce264ba9c8fa4e002e22faacd0` plus two local launchers.
The GitHub clone was blocked by this session's network sandbox, so this is a
source snapshot, not a Git checkout. Upstream's images and paper PDF are omitted.

## Mac deployment

The official `inference/start_memt_nano.sh` uses SGLang, Linux, and two NVIDIA
GPUs. `mac_server.py` is an experimental Apple Silicon route using the model's
Hugging Face Transformers code and PyTorch MPS. Stop other large models before loading it: the
checkpoint is about 70 GB and this Mac has 96 GB unified memory.

For MemT and JiuwenSwarm's terminal UI in one Terminal, run this from the
`Mem-Jiuwen` project root:

```bash
./run-memt-tui.sh
```

The launcher starts MemT if needed, verifies its API, configures JiuwenSwarm,
starts the JiuwenSwarm backend if needed, and opens the TUI in the foreground.
It builds the TUI only if the build or its dependencies are missing. Services
started by the launcher stop when the TUI exits; services that were already
running are left alone. To have the full stack owned by this one Terminal,
stop any separately running MemT and JiuwenSwarm instances first.

From a normal macOS Terminal:

```bash
cd /Users/alirezaghadimi/Mem-Jiuwen/memt-1.0
./deploy_mac_nano.sh setup
./deploy_mac_nano.sh serve
```

`setup` creates `.venv-mac`, installs pinned dependencies, and downloads the
checkpoint at Hugging Face commit
`30fe0b22422c4cc4c87475f98db5c8baa5c1a902` into
`checkpoints/MemT-1.0-Nano`. It needs at least 85 GiB free. The download may
take a long time. If a transfer fails, run `./deploy_mac_nano.sh setup` again.
Completed files are retained in the checkpoint directory; the Hugging Face
progress bar still counts all repository files from the start. Use
`./deploy_mac_nano.sh status` to see completed shards and partial downloads.
The launcher retries Xet CAS transfer errors and lowers transfer concurrency.
Do not delete the checkpoint directory or start a second setup at the same
time. `serve` binds only to `127.0.0.1:30000`.

In another terminal:

```bash
cd /Users/alirezaghadimi/Mem-Jiuwen/memt-1.0
./deploy_mac_nano.sh health
curl http://127.0.0.1:30000/v1/models
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MemTensor/MemT-1.0-Nano","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

The OpenAI-style API base is `http://127.0.0.1:30000/v1`. The chat route
accepts text messages and function tools. It supports nonstreaming responses
and SSE framing, but SSE sends the completed answer in one chunk. It runs one
generation at a time. The OpenAI route uses the request's plaintext messages
and does not persist native MemT memory between requests.

## Native memory sessions

Use these routes to retain MemT's local memory state:

```bash
curl http://127.0.0.1:30000/memt/sessions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo"}'
curl http://127.0.0.1:30000/memt/sessions/demo/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"My cat is named Memmy.","enable_memory":true,"max_tokens":64}'
curl http://127.0.0.1:30000/memt/sessions/demo/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is my cat named?","enable_memory":true,"max_tokens":64}'
curl -X POST http://127.0.0.1:30000/memt/sessions/demo/save
```

Snapshots are stored in `memt-state/`. To restore one, create a session with
`{"session_id":"restored","state":"demo.safetensors"}`. These routes use
the checkpoint's own `create_session`, `chat`, and `save_state` methods.

## Linux deployment

On a separate Linux host with two NVIDIA GPUs, the supported route is:

```bash
./deploy_nano.sh setup
./deploy_nano.sh serve
```

That launcher uses the repository's official MemT SGLang server and exposes
`GET /memt/capabilities` on `127.0.0.1:30000`.

## Current validation

The upstream inference source was compared against its Git blob hashes. The Mac
server compiles, its API helpers were checked, and both launcher scripts pass
shell syntax checks. On November 16, 2026, all 26 checkpoint weight shards
were present. The Mac server loaded the model and accepted requests, but its
first generation call failed because Transformers returned a `BatchEncoding`
where the server expected a tensor. That generation path has been fixed and
tested with the local tokenizer and a fake model. A subsequent JiuwenSwarm
request reached model attention but exhausted MPS memory during a large
prefill. The server now clears unused MPS cache and uses Transformers' chunked
prefill for prompts over 512 tokens. The generation path has been validated
with a fake model; full inference
still needs a live check because this sandbox cannot access Metal or the
host's localhost.
