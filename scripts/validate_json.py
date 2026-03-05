import json, glob

for f in glob.glob("dataset/**/**/*.json", recursive=True):
    try:
        json.load(open(f))
        print(f"OK: {f}")
    except Exception as e:
        print(f"ERROR: {f} — {e}")
