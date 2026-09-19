"""One-off: seed the fresh hackathon volume with the public model weights already cached
in the old tribe-model-cache volume, so today's app never re-downloads ~20 GB.

    modal run scripts/seed_volume.py
"""
import modal

app = modal.App("brain-twin-seed")
old = modal.Volume.from_name("tribe-model-cache")
new = modal.Volume.from_name("brain-twin-cache", create_if_missing=True)

MODELS = [
    "models--facebook--tribev2",
    "models--facebook--vjepa2-vitg-fpc64-256",
    "models--facebook--w2v-bert-2.0",
    "models--meta-llama--Llama-3.2-3B",
]


@app.function(volumes={"/old": old, "/new": new}, timeout=3600, cpu=4)
def seed():
    import os, shutil, time
    src_hub = "/old/huggingface/hub"
    dst_hub = "/new/huggingface/hub"
    os.makedirs(dst_hub, exist_ok=True)
    t0 = time.time()
    for m in MODELS:
        s, d = f"{src_hub}/{m}", f"{dst_hub}/{m}"
        if os.path.exists(d):
            print(f"skip (exists): {m}")
            continue
        print(f"copying {m} ...", flush=True)
        shutil.copytree(s, d, symlinks=True)
        new.commit()
        print(f"  done {m} in {time.time()-t0:.0f}s", flush=True)
    total = 0
    for root, _, files in os.walk(dst_hub):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    print(f"new volume hub size: {total/1e9:.1f} GB")
    return total


@app.local_entrypoint()
def main():
    print(seed.remote())
