#!/usr/bin/env python3
"""Fill `sha256` and `revision` fields in models/*.json from the Hugging Face API.

Every curated GGUF must carry a content hash so the client can verify a
download before serving it (integrity gate). The hash comes from the HF API's
LFS metadata (`lfs.oid` IS the file's SHA-256) — no download required.
Each hashed entry also pins `revision` (the repo's current commit): downloads
then resolve against that commit instead of the moving `main`, so a repo
force-push either serves the pinned bytes or fails loudly — never a silent
swap. Hash and revision are recorded together on purpose: they describe the
same observation of the repo.

Where the fields land:
  - models with a `variants` table: `sha256` + `revision` per variant;
  - models without variants (sources[0].repo + default_gguf): top-level.

A file the API cannot find is reported and left WITHOUT a hash — never guess.
(Some large-model manifests rename the file locally; those need the hash of
the actual source parts recorded by hand at curation time.)

Usage:
  python3 scripts/manifest_sha256.py [models/foo.json ...]   # default: models/*.json
  python3 scripts/manifest_sha256.py --strict                # missing hash = exit 1

Honors HF_ENDPOINT (default https://huggingface.co) for mirror users.
"""

import glob
import json
import os
import re
import sys
import urllib.request

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def api_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "idletoken-manifest-sha256"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r), r.headers.get("Link", "")


def api_revision(repo: str) -> str:
    """The repo's current main commit — the revision downloads get pinned to."""
    info, _ = api_json(f"{ENDPOINT}/api/models/{repo}")
    sha = info.get("sha") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"{repo}: API returned no usable commit sha ({sha!r})")
    return sha


def api_tree(repo: str) -> dict:
    """Map of path -> sha256 (and basename -> sha256) for a repo's LFS files."""
    url = f"{ENDPOINT}/api/models/{repo}/tree/main?recursive=true"
    out: dict[str, str] = {}
    while url:
        entries, link = api_json(url)
        for e in entries:
            oid = (e.get("lfs") or {}).get("oid")
            if oid:
                out[e["path"]] = oid
                out.setdefault(os.path.basename(e["path"]), oid)
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def insert_after(d: dict, anchor: str, key: str, value):
    """Rebuild dict with key placed right after anchor (readability of diffs)."""
    if anchor not in d:
        d[key] = value
        return d
    out = {}
    for k, v in d.items():
        out[k] = v
        if k == anchor:
            out[key] = value
    d.clear()
    d.update(out)
    return d


def main() -> int:
    strict = "--strict" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(glob.glob("models/*.json"))
    trees: dict[str, dict] = {}
    revisions: dict[str, str] = {}
    missing: list[str] = []
    for p in paths:
        with open(p) as f:
            man = json.load(f)
        changed = False

        def hash_of(repo: str, gguf: str):
            if not repo or not gguf:
                return None
            if repo not in trees:
                try:
                    trees[repo] = api_tree(repo)
                    revisions[repo] = api_revision(repo)
                except Exception as e:
                    print(f"  ! {repo}: API error: {e}", file=sys.stderr)
                    trees[repo] = {}
            return trees[repo].get(gguf)

        def pin(d: dict, anchor: str, repo: str, h: str) -> bool:
            """Write sha256 + revision next to `anchor`; True when anything changed."""
            touched = False
            if d.get("sha256") != h:
                insert_after(d, anchor, "sha256", h)
                touched = True
            rev = revisions.get(repo)
            if rev and d.get("revision") != rev:
                insert_after(d, "sha256", "revision", rev)
                touched = True
            return touched

        for v in man.get("variants") or []:
            h = hash_of(v.get("repo", ""), v.get("gguf", ""))
            if h is None:
                missing.append(f"{p}: variant {v.get('quant')} ({v.get('gguf')})")
            else:
                changed = pin(v, "gguf", v["repo"], h) or changed
        if not man.get("variants"):
            repo = (man.get("sources") or [{}])[0].get("repo", "")
            gguf = man.get("default_gguf", "")
            if repo and gguf:
                h = hash_of(repo, gguf)
                if h is None:
                    missing.append(f"{p}: default_gguf ({gguf}) not in {repo}")
                else:
                    changed = pin(man, "default_gguf", repo, h) or changed

        if changed:
            with open(p, "w") as f:
                json.dump(man, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"updated {p}")

    if missing:
        print("\nNo hash found for (left unhashed — record by hand at curation time):")
        for m in missing:
            print(f"  - {m}")
        if strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
