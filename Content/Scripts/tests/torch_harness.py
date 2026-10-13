"""Engine-free tests for unreal_torch: drives PythonCLI's runtime.py over frames the way UTorchSubsystem does.

    <PythonCLI>/ThirdParty/python/Win64/python.exe -I Content/Scripts/tests/torch_harness.py --site <venv site-packages> [--pythoncli <PythonCLI plugin dir>]
"""
import os
import shutil
import struct
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
PLUGIN = os.path.dirname(os.path.dirname(SCRIPTS))
args = sys.argv[1:]
pythoncli = args[args.index("--pythoncli") + 1] if "--pythoncli" in args else os.path.join(os.path.dirname(PLUGIN), "PythonCLI-Unreal")
site = [args[i + 1] for i, a in enumerate(args) if a == "--site"]  # several: e.g. the venv and an extra one (gymnasium)
sys.path.insert(0, os.path.join(pythoncli, "Content", "Python", "tests"))
import harness as H  # noqa: E402  (PythonCLI's harness: Bridge, check, frames)
from harness import F, check  # noqa: E402

FIXTURE = os.path.join(HERE, "torch_fixture.py")
O, S = "torch:test", "tests/torch_fixture.py"


def floats(values, shape):
    data = struct.pack(f"<{len(values)}f", *values)
    return data, {"_tensor": 0, "dtype": "float32", "shape": list(shape)}


def tensor_arg(buffers, values, shape, dtype="float32"):
    fmt = {"float32": "f", "uint8": "B"}[dtype]
    buffers.append(struct.pack(f"<{len(values)}{fmt}", *values))
    return {"_tensor": len(buffers) - 1, "dtype": dtype, "shape": list(shape)}


def unpack(reply_frame, name):
    """An output tensor of a reply as a flat list (float32/int64/...)."""
    h, b = reply_frame[1], reply_frame[2]
    desc = h["result"]["outputs"][name]
    data = H.table_entries(b)[desc["_tensor"]]
    fmt = {"float32": "f", "float64": "d", "int64": "q", "int32": "i", "uint8": "B", "bool": "?"}[desc["dtype"]]
    return list(struct.unpack(f"<{len(data) // struct.calcsize(fmt)}{fmt}", data)), desc["shape"]


def run(b, model, method, inputs, buffers=(), rows=None, options=None, timeout=60.0):
    return b.call(O, S, "_torch_run", [model, method, inputs, rows or [1], options or {}], list(buffers), timeout=timeout)


def env_step(q, rows, d, meta, sid, timeout=30.0):
    """One Lockstep step of an ATorchEnvironment-like stream: rows of [obs (d) | final obs (d) | r | term | trunc]."""
    flat = [v for row in rows for v in row]
    q.push(struct.pack(f"<{len(flat)}f", *flat), "float32", [len(rows), 2 * d + 3], meta=meta, sid=sid, timeout=timeout)
    end = time.time() + timeout
    with q.b.cv:
        while sid not in q.b.sids:
            if time.time() >= end:
                return None
            q.b.cv.wait(end - time.time())
        return q.b.sids.pop(sid)


def rl_tests(b):
    """unreal_torch.rl.UnrealVectorEnv over a simulated environment stream: spaces from the spec, steps,
    SAME_STEP autoreset (final_obs), reset with a seed, close (stop)."""
    runs = tempfile.mkdtemp(prefix="torch_runs_")
    b.clear()
    r = b.call(O, S, "_job_start", ["rl1", "rl_probe", {"steps": 2}, {"runs_dir": runs}], timeout=60)
    check("rl job started", r is not None and r[1]["ok"], r and r[1])
    info = {"call": "_torch_env", "env": "Probe", "name": "Probe", "obs_size": 2, "obs_names": ["a", "b"],
            "action": {"type": "continuous", "size": 1, "n": 2, "low": [-1.0], "high": [1.0]}}
    q = H.Producer(b, O, S, "env:Probe#1", 1, 2 * 7 * 4, policy="lockstep", shm=False, info=info)
    z = [0.0, 0.0]
    r1 = env_step(q, [[1, 2] + z + [0, 0, 0], [3, 4] + z + [0, 0, 0]], 2, {"reset": True, "step": 0}, 101)
    r2 = env_step(q, [[5, 6] + z + [1, 0, 0], [0, 0, 9, 9, 2, 1, 0]], 2, {"reset": False, "step": 1}, 102)
    r3 = env_step(q, [[7, 8] + z + [0, 0, 0], [1, 1] + z + [0, 0, 0]], 2, {"reset": False, "step": 2}, 103)
    r4 = env_step(q, [[0, 0] + z + [0, 0, 0], [1, 1] + z + [0, 0, 0]], 2, {"reset": True, "step": 3}, 104)
    acts = r1 and r1["ok"] and r1["result"]["actions"]
    check("step reply: actions (N, A) float32", bool(acts) and acts["shape"] == [2, 1] and acts["dtype"] == "float32", r1)
    check("reset(seed) asks Unreal for a reseeded reset", r3 is not None and r3["ok"] and r3["result"].get("reset") is True and r3["result"].get("seed") == 5, r3)
    check("close answers the pending step with stop", r4 is not None and r4["ok"] and r4["result"].get("stop") is True, r4)
    done = H.job_events(b, "rl1", {"completed", "failed", "stopped"}, timeout=60)
    res = (done[-1].get("result") or {}) if done else {}
    print(f"     gymnasium: {res.get('gymnasium')}, action space {res.get('action_space')}")
    check("env: spec -> spaces (2 envs, obs 2, Box action, batched (2, 1))", done and done[-1]["kind"] == "completed" and res.get("num_envs") == 2
          and res.get("obs_size") == 2 and res.get("batched_action_shape") == [2, 1] and "Box" in res.get("action_space", ""), done and done[-1])
    check("env: first obs, rewards, dones", res.get("first_obs") == [[1, 2], [3, 4]] and res.get("rewards") == [[1, 2], [0, 0]]
          and res.get("dones") == [[False, True], [False, False]], res)
    check("env: SAME_STEP autoreset (new episode's obs, final_obs in info)", res.get("finals") == [[None, [9, 9]]], res)
    check("env: reset(seed) returns the reset frame's obs", res.get("reset_obs") == [[0, 0], [1, 1]], res)
    q.close()
    shutil.rmtree(runs, ignore_errors=True)


def examples(b):
    """examples/digits.py: train briefly (MNIST download on first use), publishing into the live model; resume."""
    EO, ES = "torch:examples", "examples/digits.py"
    b.control(cmd="start", owner=EO, script=ES, path=os.path.join(SCRIPTS, "examples", "digits.py"))
    check("digits example started", b.status("started", EO, ES, timeout=120) is not None)
    sample = b.call(EO, ES, "_torch_run", ["digits", "sample", {"index": 7}, [1], {}], timeout=300)
    check("sample: an MNIST test image", sample is not None and sample[1]["ok"] and sample[1]["result"]["outputs"]["label"] == [9], sample and sample[1])
    img_desc = sample[1]["result"]["outputs"]["image"] if sample and sample[1]["ok"] else None
    img = H.table_entries(sample[2])[img_desc["_tensor"]] if img_desc else b""

    def read():
        r = b.call(EO, ES, "_torch_run", ["digits", "read", {"image": {"_tensor": 0, "dtype": "float32", "shape": [1, 256, 256, 1]}}, [1], {}], [img])
        return r[1]["result"]["outputs"] if r and r[1]["ok"] else {"text": [r and r[1].get("error")]}

    before = read()
    runs = tempfile.mkdtemp(prefix="torch_runs_")
    b.clear()
    t0 = time.time()
    r = b.call(EO, ES, "_job_start", ["t1", "train_digits", {"epochs": 1, "max_steps": 150}, {"runs_dir": runs}], timeout=60)
    done = H.job_events(b, "t1", {"completed", "failed", "stopped"}, timeout=600)
    final = done[-1] if done else {}
    acc = (final.get("result") or {}).get("accuracy", 0)
    check("train_digits: one short epoch reaches > 90% test accuracy", r is not None and r[1]["ok"] and final.get("kind") == "completed" and acc > 0.9, (r and r[1], final))
    print(f"     trained 150 steps in {time.time() - t0:.1f} s (incl. MNIST load): accuracy {acc:.1%}")
    after = read()
    check("the live model reads the digit after training (published weights)", after.get("digit") is not None and after["text"][0].startswith("Camera sees: 9"), (before, after))
    r = b.call(EO, ES, "_job_start", ["t2", "train_digits", {"epochs": 2, "max_steps": 50}, {"runs_dir": runs, "resume": True}], timeout=60)
    done = H.job_events(b, "t2", {"completed", "failed", "stopped"}, timeout=600)
    check("resume continues the run (from epoch 2)", r is not None and r[1]["ok"] and r[1]["result"]["resumed"] and done and done[-1]["kind"] == "completed"
          and len([e for e in done if e["kind"] == "metrics"]) == 1, (r and r[1], done and done[-1]))
    b.control(cmd="stopOwner", owner=EO)


def main():
    project = tempfile.mkdtemp(prefix="torch_harness_")
    b = H.Bridge(site_dirs=site, script_dirs=[SCRIPTS], project_dir=project)
    ready = b.status("ready", timeout=30)
    check("bridge ready", ready is not None)
    b.control(cmd="start", owner=O, script=S, path=FIXTURE)
    started = b.status("started", O, S, timeout=120)
    check("fixture started (imports torch)", started is not None, b.status("error", O, S, timeout=1))

    r = b.call(O, S, "_torch_list", [], timeout=60)
    names = {m["model"] for m in r[1]["result"]["models"]} if r and r[1]["ok"] else set()
    check("models listed with signatures", {"doubler", "rowwise", "imager", "counter", "sized", "faulty"} <= names, r and r[1])
    r = b.call(O, S, "_torch_signature", ["rowwise"])
    sig = r[1]["result"] if r and r[1]["ok"] else {}
    stats_m = next((m for m in sig.get("methods", []) if m["name"] == "stats"), {})
    check("signature: typed inputs", stats_m.get("inputs") == [{"name": "x", "required": True, "kind": "tensor", "dtype": "float32", "shape": [-1]},
                                                                {"name": "label", "required": False, "kind": "json", "type": "str"}] and not stats_m.get("batch"), stats_m)

    r = b.call(O, S, "_torch_load", ["doubler", {"device": "auto"}], timeout=120)
    check("load", r is not None and r[1]["ok"] and r[1]["result"]["device"].startswith(("cuda", "cpu", "xpu")), r and r[1])
    device = r[1]["result"]["device"] if r and r[1]["ok"] else "?"
    print(f"     device {device}, load {r[1]['result'].get('load_ms') if r and r[1]['ok'] else '?'} ms")

    # one request; then a stacked call (Unreal coalesced 3 requests of a tick)
    buf = []
    x = tensor_arg(buf, [1, 2, 3], [1, 3])
    r = run(b, "doubler", "double", {"x": x}, buf)
    out, shape = unpack(r, "output") if r and r[1]["ok"] else ([], [])
    check("batched method, one request", out == [2, 4, 6] and shape == [1, 3] and r[1]["result"]["per_row"] == ["output"], r and r[1])
    buf = []
    x = tensor_arg(buf, list(range(8)), [4, 2])
    r = run(b, "doubler", "double", {"x": x}, buf, rows=[1, 2, 1])
    out, shape = unpack(r, "output") if r and r[1]["ok"] else ([], [])
    check("stacked requests: rows kept, one pass", out == [v * 2 for v in range(8)] and shape == [4, 2] and r[1]["result"]["rows"] == 4, r and r[1])

    # micro-batching: concurrent calls meet in the batcher (the method sleeps so they queue up)
    run(b, "doubler", "set_delay", {"seconds": 0.05})
    b.call(O, S, "_torch_stats", ["doubler", True])
    ids = []
    for i in range(8):
        buf = []
        x = tensor_arg(buf, [float(i)], [1, 1])
        ids.append(b.event(O, S, "_torch_run", ["doubler", "double", {"x": x}, [1], {}], buf, call=True))
    replies = [b.reply(i, timeout=30) for i in ids]
    ok = all(r is not None and r[1]["ok"] for r in replies)
    vals = [unpack(r, "output")[0][0] for r in replies] if ok else []
    st = b.call(O, S, "_torch_stats", ["doubler"])
    passes = st[1]["result"]["passes"] if st and st[1]["ok"] else -1
    check("concurrent calls are micro-batched and each gets its own rows", ok and vals == [2.0 * i for i in range(8)] and 1 <= passes < 8, (passes, vals))
    print(f"     8 concurrent calls -> {passes} forward passes, batch sizes up to {max(r[1]['result']['batch'] for r in replies) if ok else '?'}")
    run(b, "doubler", "set_delay", {"seconds": 0.0})

    # an unbatched method gets one request at a time; JSON inputs and outputs
    buf = []
    x = tensor_arg(buf, [1, 2, 3, 4, 5, 6], [2, 3])
    r = run(b, "rowwise", "stats", {"x": x, "label": "a"}, buf, rows=[1, 1])
    sums, _ = unpack(r, "sum") if r and r[1]["ok"] else ([], [])
    res = r[1]["result"] if r and r[1]["ok"] else {}
    check("unbatched method runs per request; scalars come back per row", sums == [6.0, 15.0] and res["outputs"].get("n") == [3, 3]
          and res["outputs"].get("label") == ["a", "a"] and set(res["per_row"]) == {"sum", "n", "label"}, res)

    # images: uint8 BGRA (N, H, W, 4) from a render target, resized and gray; ImageNet normalization
    buf = []
    px = [10, 20, 30, 255] * (8 * 8) * 2  # two 8x8 BGRA images
    img = tensor_arg(buf, px, [2, 8, 8, 4], "uint8")
    r = run(b, "imager", "gray", {"image": img}, buf, rows=[1, 1], options={"layouts": {"image": "BGRA"}})
    means, _ = unpack(r, "mean") if r and r[1]["ok"] else ([], [])
    res = r[1]["result"] if r and r[1]["ok"] else {}
    check("ut.Image: BGRA uint8 -> gray, resized", len(means) == 2 and abs(means[0] - 20 / 255) < 1e-4 and res["outputs"].get("shape") == [1, 4, 4], res)
    buf = []
    img = tensor_arg(buf, [1.0, 0.0, 0.5] * 4, [1, 2, 2, 3])
    r = run(b, "imager", "imagenet", {"image": img}, buf)
    vals, shape = unpack(r, "output") if r and r[1]["ok"] else ([], [])
    expect = [(1.0 - 0.485) / 0.229, (0.0 - 0.456) / 0.224, (0.5 - 0.406) / 0.225]
    check("ut.Image: RGB float, ImageNet normalization", shape == [1, 3] and all(abs(a - e) < 1e-4 for a, e in zip(vals, expect)), (vals, shape))

    # errors
    r = run(b, "doubler", "nope", {})
    check("unknown method is an error", r is not None and not r[1]["ok"] and "no exported method 'nope'" in r[1]["error"], r and r[1])
    r = run(b, "doubler", "double", {"y": 1})
    check("unknown input is an error", r is not None and not r[1]["ok"] and "has no input y" in r[1]["error"], r and r[1])
    buf = []
    x = tensor_arg(buf, [1.0], [1, 1])
    r = run(b, "faulty", "boom", {"x": x}, buf)
    check("an exception in a batched method fails the call with its traceback", r is not None and not r[1]["ok"] and "model failed on purpose" in r[1]["error"]
          and "Traceback" in r[1].get("traceback", ""), r and r[1])
    buf = []
    x = tensor_arg(buf, [1.0, 2.0], [2, 1])
    r = run(b, "doubler", "double", {"x": x}, buf, rows=[1])
    check("row count mismatch is an error", r is not None and not r[1]["ok"] and "rows" in r[1]["error"], r and r[1])

    # warmup on load from fixed-size annotations
    b.call(O, S, "_torch_load", ["sized", {}], timeout=60)
    r = run(b, "sized", "warm_count", {})
    check("warmup ran the exported method once", r is not None and r[1]["ok"] and r[1]["result"]["outputs"]["output"] == [1], r and r[1])

    # weights: save, change, load back; then a script reload keeps the weights
    r = run(b, "counter", "bump", {"amount": 5.0})
    first = r[1]["result"]["outputs"]["output"][0] if r and r[1]["ok"] else None
    reloaded_src = open(FIXTURE, encoding="utf-8").read().replace("VERSION = 1", "VERSION = 2")
    tmp_dir = tempfile.mkdtemp(prefix="torch_fixture_")
    tmp_fixture = os.path.join(tmp_dir, "torch_fixture.py")
    with open(tmp_fixture, "w", encoding="utf-8") as f:
        f.write(reloaded_src)
    b.clear()
    b.control(cmd="reload", owner=O, script=S, path=tmp_fixture)
    check("script reloaded", b.status("started", O, S, timeout=60) is not None)
    r = run(b, "counter", "version", {})
    r2 = run(b, "counter", "bump", {"amount": 1.0})
    check("reload: new code, same weights", r is not None and r[1]["ok"] and r[1]["result"]["outputs"]["output"] == [2] and r2 is not None and r2[1]["ok"]
          and first is not None and abs(r2[1]["result"]["outputs"]["output"][0] - (first + 1.0)) < 1e-5, (first, r and r[1], r2 and r2[1]))
    r = b.call(O, S, "_torch_unload", ["counter"])
    check("unload", r is not None and r[1]["ok"], r and r[1])

    # streams bound to model methods: a camera-like stream (latest, shared memory, BGRA) and a Lockstep one
    b.clear()
    p = H.Producer(b, O, S, "cam#1", 2, 8 * 8 * 4, policy="latest", shm=True,
                   info={"call": "_torch_stream", "model": "imager", "method": "gray", "input": "image", "binding": 7, "name": "cam"})
    p.push(bytes([30, 20, 10, 255] * 64), "uint8", [8, 8, 4], layout="BGRA")
    ev = b.wait(lambda t, h, x: t == F.EVENT and h.get("name") == "_torch_stream_result", 60)
    res = ev[1]["args"][0] if ev else {}
    mean = struct.unpack("<f", H.table_entries(ev[2])[res["outputs"]["mean"]["_tensor"]])[0] if ev and "outputs" in res else None
    check("camera stream -> model method -> result event (binding, gray mean)", res.get("binding") == 7 and mean is not None and abs(mean - 20 / 255) < 1e-4, res)
    p.close()
    q = H.Producer(b, O, S, "agents#1", 1, 64, policy="lockstep", shm=False,
                   info={"call": "_torch_stream", "model": "doubler", "method": "double", "input": "x", "binding": 8, "name": "agents"})
    rep = q.step(struct.pack("<3f", 1, 2, 3), "float32", [3], sid=77)
    check("lockstep stream -> model method -> step reply", rep is not None and rep["ok"] and rep["result"]["outputs"]["output"]["shape"] == [1, 3], rep)
    q.close()

    # torch.compile (triton-windows): a 3-row batch is padded to 4, results match eager
    t0 = time.perf_counter()
    r = b.call(O, S, "_torch_load", ["compiled", {"compile": "default"}], timeout=300)
    buf = []
    x = tensor_arg(buf, [0.1 * i for i in range(12)], [3, 4])
    r = run(b, "compiled", "f", {"x": x}, buf, rows=[1, 1, 1], timeout=300)
    ref = run(b, "compiled", "reference", {"x": x}, buf, rows=[1, 1, 1])
    got, shape = unpack(r, "output") if r and r[1]["ok"] else ([], [])
    want, _ = unpack(ref, "output") if ref and ref[1]["ok"] else ([], [])
    check("compiled model: padded batch, same results as eager", shape == [3, 2] and len(got) == 6 and all(abs(a - b_) < 2e-3 for a, b_ in zip(got, want)), (got, want))
    print(f"     compile + first call: {time.perf_counter() - t0:.1f} s")

    # throughput: stacked calls of 64 rows
    buf = []
    x = tensor_arg(buf, [1.0] * 64, [64, 1])
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        r = run(b, "doubler", "double", {"x": x}, buf, rows=[1] * 64)
    dt = time.perf_counter() - t0
    check("stacked 64-row calls", r is not None and r[1]["ok"])
    print(f"     {n * 64 / dt:.0f} requests/s in 64-row stacked calls ({dt / n * 1000:.2f} ms per call, engine-free)")

    rl_tests(b)

    if "--examples" in args:
        examples(b)

    b.control(cmd="exit", grace=500)
    b.status("exiting")
    b.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(project, ignore_errors=True)
    print(f"\n{sum(H.results)}/{len(H.results)} passed")
    sys.exit(0 if all(H.results) else 1)


if __name__ == "__main__":
    main()
