from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gdown
import requests
from gdown.download_folder import _download_and_parse_google_drive_link, _get_session

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_external" / "pff" / "raw"
FOLDER_ID = "1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/98.0.4758.102 Safari/537.36")
EXPORT = {
    "application/vnd.google-apps.document": ("document", "pdf"),
    "application/vnd.google-apps.spreadsheet": ("spreadsheets", "csv"),
    "application/vnd.google-apps.presentation": ("presentation", "pdf"),
}


def listing(refresh=False):
    p = RAW / "listing.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    sess, _ = _get_session(proxy=None, use_cookies=False, user_agent=UA, cookies_file=None)
    root = _download_and_parse_google_drive_link(sess=sess, folder_id=FOLDER_ID, quiet=True,
                                                 verify=True, timeout=60)
    out = []

    def walk(f, path):
        pth = f"{path}/{f.name}" if path else f.name
        out.append(dict(id=f.id, path=pth, type=f.type))
        for c in f.children or []:
            walk(c, pth)

    walk(root, "")
    RAW.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out


def local_path(entry) -> Path:
    parts = entry["path"].split("/")[1:]
    parts = [re.sub(r"[^\w.,\- ]", "_", s) for s in parts]
    dst = RAW.joinpath(*parts)
    if entry["type"] in EXPORT:
        dst = dst.with_name(dst.name + "." + EXPORT[entry["type"]][1])
    return dst


def direct(fid: str, tmp: Path):
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            r.close()
            gdown.download(id=fid, output=str(tmp), quiet=True, resume=True)
            return
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def fetch(entry, dst: Path, tries=2) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    for k in range(tries):
        try:
            if entry["type"] in EXPORT:
                kind, fmt = EXPORT[entry["type"]]
                url = f"https://docs.google.com/{kind}/d/{entry['id']}/export?format={fmt}"
                r = requests.get(url, timeout=120, headers={"User-Agent": UA})
                r.raise_for_status()
                tmp.write_bytes(r.content)
            else:
                direct(entry["id"], tmp)
            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.rename(dst)
                return True
        except Exception as e:  # noqa: BLE001 - Drive quota / transient errors
            msg = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
            wait = 10 * (k + 1)
            print(f"  retry {k + 1}/{tries} {entry['path']}: {msg} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="substrings of the Drive path to fetch")
    ap.add_argument("--all-versions", action="store_true")
    ap.add_argument("--refresh-listing", action="store_true")
    ap.add_argument("--passes", type=int, default=6, help="passes over the failed files")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (Drive throttles per connection)")
    a = ap.parse_args()
    files = [e for e in listing(a.refresh_listing) if e["type"] != "application/vnd.google-apps.folder"]
    if not a.all_versions:
        files = [e for e in files if not re.search(r"Event Data/[^/]*20\d\d/", e["path"])
                 or e["path"].endswith(".pdf")]
    if a.only:
        files = [e for e in files if any(s in e["path"] for s in a.only)]
    files.sort(key=lambda e: ("Tracking Data" in e["path"] and e["path"].endswith(".bz2"), e["path"]))
    todo = files
    for ps in range(a.passes):
        failed = []

        def one(ie):
            i, e = ie
            dst = local_path(e)
            if dst.exists() and dst.stat().st_size > 0:
                return None
            ok = fetch(e, dst)
            sz = dst.stat().st_size / 1e6 if ok else 0
            print(f"[pass {ps + 1} {i + 1}/{len(todo)}] {'ok ' if ok else 'FAILED'} {e['path']} {sz:.1f} MB",
                  flush=True)
            return None if ok else dict(e, local=str(dst), url=f"https://drive.google.com/file/d/{e['id']}/view")

        with ThreadPoolExecutor(a.workers) as ex:
            failed = [r for r in ex.map(one, enumerate(todo)) if r is not None]
        if not failed:
            break
        todo = [f for f in files if f["id"] in {x["id"] for x in failed}]
        if ps + 1 < a.passes:
            wait = min(300 * 2 ** ps, 3600)
            print(f"pass {ps + 1}: {len(failed)} failed; sleeping {wait}s before retrying them", flush=True)
            time.sleep(wait)
    fp = RAW / "failed.json"
    old = [f for f in (json.loads(fp.read_text()) if fp.exists() else [])
           if not Path(f["local"]).exists()]
    keep = {f["id"]: f for f in old + failed}
    fp.write_text(json.dumps(list(keep.values()), indent=1))
    print(f"done; {len(failed)} failed this run", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
