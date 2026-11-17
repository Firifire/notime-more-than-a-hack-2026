import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data_external"
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}

PATHCRF = ("https://github.com/hyunsungkim-ds/pathcrf", "1b7c6f85a6b78c92e4543fde72ca2b86a933f997")
ELASTIC = ("https://github.com/hyunsungkim-ds/elastic", "e245224")
METRICA = ("https://github.com/metrica-sports/sample-data", "e706dd506b360d69d9d123d5b8026e7294b13996")
SKILLCORNER_SHA = "4340d274572876239c154c90bc507a9b3250a656"
SKILLCORNER_IDS = ("1874553 1886347 1899585 1925299 1927964 1953632 1959846 1986691 1996435 1996436 2006229 "
                   "2006363 2007448 2007721 2010085 2011166 2013725 2015213 2016236 2017461").split()
IDSSE_ARTICLE = "https://api.figshare.com/v2/articles/28196177"
STV2_REPO, STV2_REV = "atomscott/soccertrack-v2", "v1.2"
STV2_PATTERNS = ["README.md", "raw/*/*_tracker_box_data.xml", "raw/*/*_tracker_box_metadata.xml",
                 "raw/*/*_player_nodes.csv"]


def get(url, dst, md5=None):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    if md5 and hashlib.md5(tmp.read_bytes()).hexdigest() != md5:
        tmp.unlink()
        raise RuntimeError("md5 mismatch: %s" % url)
    tmp.rename(dst)
    print("  got", dst.relative_to(ROOT), flush=True)


def git_at(url, sha, dst):
    dst = Path(dst)
    if not (dst / ".git").exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", sha], check=True)
    print("  %s @ %s -> %s" % (url, sha[:7], dst.relative_to(ROOT)), flush=True)


def idsse():
    d = EXT / "idsse"
    d.mkdir(parents=True, exist_ok=True)
    art = json.load(urllib.request.urlopen(urllib.request.Request(IDSSE_ARTICLE, headers=UA), timeout=60))
    (d / "article.json").write_text(json.dumps(art, indent=1))
    lines = []
    for f in art["files"]:
        get(f["download_url"], d / f["name"], f.get("supplied_md5") or f.get("computed_md5"))
        lines.append("%s %s %s" % (f["download_url"], f["name"], f.get("supplied_md5")))
    (d / "files.txt").write_text("\n".join(lines) + "\n")


def pathcrf():
    git_at(*PATHCRF, EXT / "pathcrf" / "repo")


def elastic():
    repo = EXT / "elastic" / "repo"
    git_at(*ELASTIC, repo)
    gt = EXT / "elastic" / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    for p in sorted((repo / "benchmark" / "gt").glob("*.parquet")):
        (gt / p.name).write_bytes(p.read_bytes())


def metrica():
    git_at(*METRICA, EXT / "metrica" / "repo")


def skillcorner():
    base = EXT / "skillcorner"
    for mid in SKILLCORNER_IDS:
        rel = "data/matches/%s/%s" % (mid, mid)
        get("https://raw.githubusercontent.com/SkillCorner/opendata/%s/%s_match.json" % (SKILLCORNER_SHA, rel),
            base / (rel + "_match.json"))
        get("https://raw.githubusercontent.com/SkillCorner/opendata/%s/%s_phases_of_play.csv" % (SKILLCORNER_SHA, rel),
            base / (rel + "_phases_of_play.csv"))
        get("https://media.githubusercontent.com/media/SkillCorner/opendata/%s/%s_tracking_extrapolated.jsonl"
            % (SKILLCORNER_SHA, rel), base / (rel + "_tracking_extrapolated.jsonl"))
    (base / "commit.txt").write_text(json.dumps({"sha": SKILLCORNER_SHA}) + "\n")


def stv2():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("stv2 needs huggingface_hub (pip install huggingface_hub)")
    snapshot_download(STV2_REPO, repo_type="dataset", revision=STV2_REV, allow_patterns=STV2_PATTERNS,
                      local_dir=str(EXT / "raw" / "soccertrack_v2"))
    print("  %s @ %s -> data_external/raw/soccertrack_v2" % (STV2_REPO, STV2_REV), flush=True)


def pff():
    subprocess.run([sys.executable, "-I", str(ROOT / "src" / "external" / "download_pff.py")], check=True)


STATSBOMB_SHA = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
SC2020_SB_EVENTS = (303470, 303596)


def skillcorner2020():
    subprocess.run([sys.executable, "-I", str(ROOT / "src" / "external" / "download_open3.py"), "skillcorner2020"],
                   check=True)
    for mid in SC2020_SB_EVENTS:
        get("https://raw.githubusercontent.com/statsbomb/open-data/%s/data/events/%d.json" % (STATSBOMB_SHA, mid),
            EXT / "statsbomb" / "data" / "events" / ("%d.json" % mid))


def statsbomb():
    base = "https://raw.githubusercontent.com/statsbomb/open-data/%s/data" % STATSBOMB_SHA
    d = EXT / "statsbomb" / "data"
    get(base + "/matches/43/106.json", d / "matches" / "43" / "106.json")
    ids = [x["match_id"] for x in json.loads((d / "matches" / "43" / "106.json").read_text())]
    for mid in ids:
        get(base + "/events/%d.json" % mid, d / "events" / ("%d.json" % mid))
    (EXT / "statsbomb" / "wc2022_done.txt").write_text("statsbomb/open-data %s: matches/43/106.json + %d event files\n"
                                                         % (STATSBOMB_SHA, len(ids)))


SOURCES = {"idsse": idsse, "pathcrf": pathcrf, "elastic": elastic, "metrica": metrica,
           "skillcorner": skillcorner, "stv2": stv2, "pff": pff, "skillcorner2020": skillcorner2020,
           "statsbomb": statsbomb}


def main():
    names = sys.argv[1:] or list(SOURCES)
    bad = [n for n in names if n not in SOURCES]
    if bad:
        raise SystemExit("unknown source(s) %s; choose from %s" % (bad, list(SOURCES)))
    for n in names:
        print("[download] %s" % n, flush=True)
        SOURCES[n]()


if __name__ == "__main__":
    main()
