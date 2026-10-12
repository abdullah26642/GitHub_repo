# %% [markdown]
# # Faithful-by-design explanations for chest X-ray classification — revision experiments (COVID-QU-Ex)
#
# **One click:** `Runtime → Change runtime type → A100 GPU (High-RAM if offered)`, then `Runtime → Run all`.
#
# What this notebook does, end to end and without manual steps:
# 1. Downloads **COVID-QU-Ex** (Tahir et al., 2021; Kaggle `anasmohammedtahir/covidqu`, CC BY-SA 4.0) **directly from Kaggle to the Colab VM** — nothing passes through Google Drive or your computer.
# 2. Audits the data: official train/val/test split, class counts, image sizes, exact and near-duplicate leakage across splits.
# 3. Trains 5 ImageNet-pretrained backbones (ResNet-34, ResNet-50, VGG-19, DenseNet-121, ConvNeXt-Tiny) with the torchvision **default head** and with the **Conv → Flatten → Linear** head, over 3 seeds; plus head ablations (truncated backbone; the Lamprou et al. replacement block), class-imbalance ablations (natural and induced imbalance) and lung-masked training.
# 4. Classification statistics: mean ± SD over seeds, 95 % bootstrap CIs, McNemar tests, calibration, shortcut tests (lung-only / lungs-removed inputs).
# 5. Explanations with 8 attribution methods + a random baseline; exact numerical completeness checks; faithfulness (AOPC, deletion/insertion), robustness (Max-Sensitivity), HAAS, model-randomization sanity checks, plausibility against lung masks and expert COVID-19 infection masks; paired Wilcoxon tests with Holm correction.
# 6. A logit-averaging ensemble whose HiResCAM explanation is exactly complete.
# 7. Every table (CSV + one Excel workbook) and figure (PNG 300 dpi + PDF) is written to Google Drive.
#
# **Resumable:** if Colab disconnects, reconnect and run all cells again — completed runs and evaluations are skipped.
# **Recommended order:** run once with `RUN_MODE = "smoke"` (about 20 min) to validate the pipeline on the real data, then set `RUN_MODE = "full"`.

# %%
# ================================ USER SETTINGS ================================
RUN_MODE = "full"          # "smoke" (pipeline test, ~20 min) | "lite" (~3 h) | "full" (~6-8 h on A100)
USE_DRIVE = True           # write results to Google Drive (recommended)
DRIVE_DIR = "/content/drive/MyDrive/CXR_XAI_Revision_2026"
KAGGLE_DATASET = "anasmohammedtahir/covidqu"   # COVID-QU-Ex
SAVE_ALL_CHECKPOINTS_TO_DRIVE = False          # seed-0 checkpoints are always saved (needed for the XAI stage)
# ===============================================================================

# %%
import os, sys, json, time, math, random, shutil, zipfile, hashlib, subprocess, platform, datetime, warnings, gc, traceback, copy
from collections import OrderedDict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

RUN_MODE = os.environ.get("CXR_RUN_MODE", RUN_MODE)
LOCAL_TEST = os.environ.get("CXR_LOCAL_TEST", "0") == "1"
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models as tvm
from torchvision.models.convnext import LayerNorm2d
from PIL import Image
import matplotlib
if not IN_COLAB:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import dctn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

assert RUN_MODE in ("smoke", "lite", "full"), RUN_MODE
PRESETS = {
    "full": dict(seeds=[0, 1, 2], max_epochs=20, patience=5, batch=128, steps_cap=None, eval_cap=None,
                 head_ablation=True, balance_ablation=True, lung_masked=True,
                 n_xai_per_class=100, n_ms_per_class=40, aopc_grids=[4, 8, 14], ms_radii=[0.02, 0.05, 0.10],
                 ms_samples=10, ig_steps=32, rise_masks=2000, delins_steps=28, boot=2000, n_examples=6),
    "lite": dict(seeds=[0], max_epochs=15, patience=4, batch=128, steps_cap=None, eval_cap=None,
                 head_ablation=True, balance_ablation=True, lung_masked=True,
                 n_xai_per_class=50, n_ms_per_class=20, aopc_grids=[8, 14], ms_radii=[0.02, 0.05, 0.10],
                 ms_samples=8, ig_steps=32, rise_masks=1000, delins_steps=28, boot=1000, n_examples=6),
    "smoke": dict(seeds=[0], max_epochs=2, patience=2, batch=32, steps_cap=12, eval_cap=300,
                  head_ablation=True, balance_ablation=True, lung_masked=True,
                  n_xai_per_class=3, n_ms_per_class=2, aopc_grids=[4, 8], ms_radii=[0.05],
                  ms_samples=2, ig_steps=4, rise_masks=64, delins_steps=8, boot=100, n_examples=2),
}
CFG = dict(PRESETS[RUN_MODE])
CFG.update(dict(
    run_mode=RUN_MODE, img_size=256, input_size=224,
    lr_backbone=1e-4, lr_head=1e-3, weight_decay=1e-4, warmup_epochs=1, grad_clip=5.0,
    aug_rot_deg=10.0, aug_scale=0.10, aug_translate=0.05, aug_brightness=0.10, aug_contrast=0.10,
    eval_batch=512, xai_batch=8, fwd_chunk=512, scorecam_chunk=512, rise_chunk=500, rise_s=7, rise_p=0.5,
    tie_eps=1e-6, dup_hamming=6, dup_ncc=0.97, covid10_fraction=0.10, pointing_tol_px=15,
    pretrained=os.environ.get("CXR_PRETRAINED", "1") == "1",
))
if LOCAL_TEST:
    CFG.update(dict(eval_batch=64, fwd_chunk=64, scorecam_chunk=64, rise_chunk=32, xai_batch=2))
PRETRAINED = CFG["pretrained"]

# ----------------------------------------------------------------------------- paths
DRIVE_OK = False
if IN_COLAB and USE_DRIVE and not LOCAL_TEST:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_OK = True
    except Exception as e:
        print("WARNING: Google Drive could not be mounted:", e)
if IN_COLAB and USE_DRIVE and not LOCAL_TEST and not DRIVE_OK:
    raise SystemExit("Google Drive is NOT mounted, so earlier results cannot be found and everything would start from zero. "
                     "Run this cell again and approve the Google Drive permission pop-up.")
if LOCAL_TEST:
    OUT_ROOT = Path(os.environ["CXR_OUT"])
    WORK = Path(os.environ.get("CXR_WORK", str(OUT_ROOT / "_work")))
elif DRIVE_OK:
    OUT_ROOT, WORK = Path(DRIVE_DIR), Path("/content/work")
else:
    OUT_ROOT, WORK = Path("/content/CXR_XAI_results"), Path("/content/work")
OUT = OUT_ROOT / f"results_{RUN_MODE}"
DIRS = {k: OUT / k for k in ["runs", "tables", "figures", "xai", "audit", "logs", "checkpoints"]}
for d in list(DIRS.values()) + [WORK, WORK / "ckpt", WORK / "cache"]:
    d.mkdir(parents=True, exist_ok=True)
LOG_FILE = DIRS["logs"] / "log.txt"


def log(*a):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:                      # a transient Google Drive error must never stop the run
        print(f"  (log file write failed: {e!r})", flush=True)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


TABLES = OrderedDict()


def save_table(df, name, index=False):
    df.to_csv(DIRS["tables"] / f"{name}.csv", index=index)
    TABLES[name] = df.reset_index() if index else df
    return df


# ----------------------------------------------------------------------------- device and precision
if not torch.cuda.is_available():
    raise SystemExit("No GPU found. In Colab: Runtime -> Change runtime type -> A100 GPU.")
DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if not any(k in GPU_NAME for k in ("A100", "H100", "H200", "L4", "RTX PRO", "B200")) and not LOCAL_TEST:
    print(f"WARNING: GPU is {GPU_NAME}. The notebook works, but an A100 is strongly recommended for the time budget.")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class tf32_off:
    """Context manager: full float32 precision (TF32 disabled) for the few computations where it matters."""

    def __enter__(self):
        self.s = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.s


ENV = dict(time=str(datetime.datetime.now()), run_mode=RUN_MODE, python=sys.version.split()[0], platform=platform.platform(),
           torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__, pandas=pd.__version__,
           cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=GPU_NAME, gpu_mem_gb=round(GPU_MEM_GB, 1),
           amp_dtype=str(AMP_DTYPE), cpu_count=os.cpu_count(), in_colab=IN_COLAB, drive=DRIVE_OK, out=str(OUT))
try:
    import psutil
    ENV["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
except Exception:
    pass
write_json(DIRS["logs"] / "environment.json", ENV)
write_json(DIRS["logs"] / "config.json", CFG)
log("Environment:", json.dumps(ENV))
log("Config:", json.dumps(CFG))
log("Expected wall-clock on one A100:", {"smoke": "about 20-30 min", "lite": "about 3-4 h", "full": "about 6-8 h"}[RUN_MODE],
    "| results folder:", OUT)

# %% [markdown]
# ## 1. Data: COVID-QU-Ex, downloaded cloud-to-cloud from Kaggle
# Uses the public Kaggle API endpoint (no login needed for this public dataset); falls back to `kagglehub` if needed.
# Only the **Lung Segmentation Data** (all 33,920 images, official split, lung masks) and the **Infection Segmentation Data**
# (expert COVID-19 infection masks; every image of this subset also belongs to the same split of the main set) are used.

# %%
CLASSES = ["COVID-19", "Non-COVID", "Normal"]
CLS_SHORT = ["COVID-19", "Non-COVID", "Normal"]
SPLITS = ["Train", "Val", "Test"]
EXPECTED = {"Train": (7658, 7208, 6849), "Val": (1903, 1802, 1712), "Test": (2395, 2253, 2140)}
IMG_SIZE, INPUT = CFG["img_size"], CFG["input_size"]


def find_root(base):
    base = Path(base)
    for dirpath, dirnames, _ in os.walk(base):
        p = Path(dirpath)
        if p.name == "Lung Segmentation Data" and p.parent.name == "Lung Segmentation Data":
            return p.parent.parent
        if len(p.relative_to(base).parts) >= 5:
            dirnames[:] = []
    raise FileNotFoundError(f"'Lung Segmentation Data' not found under {base}")


def get_dataset_root():
    if LOCAL_TEST:
        return Path(os.environ["CXR_DATA_ROOT"])
    root_file = WORK / "dataset_root.txt"
    if root_file.exists() and Path(root_file.read_text().strip()).exists():
        return Path(root_file.read_text().strip())
    data_dir = WORK / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = WORK / "covidqu.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
    root = None
    try:
        t0 = time.time()
        if not (zpath.exists() and zipfile.is_zipfile(zpath)):
            log("Downloading COVID-QU-Ex from Kaggle to the Colab VM ...")
            r = subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-sS", "-o", str(zpath), url])
            if r.returncode != 0 or not zipfile.is_zipfile(zpath):
                raise RuntimeError(f"curl exit code {r.returncode}")
        log(f"Zip: {zpath.stat().st_size / 1e9:.2f} GB ({time.time() - t0:.0f} s). Extracting ...")
        t0 = time.time()
        ok = False
        if shutil.which("unzip"):
            r = subprocess.run(["unzip", "-q", "-o", str(zpath), "-d", str(data_dir)])
            ok = r.returncode in (0, 1)
        if not ok:
            with zipfile.ZipFile(zpath) as z:
                z.extractall(data_dir)
        root = find_root(data_dir)
        log(f"Extracted in {time.time() - t0:.0f} s -> {root}")
    except Exception as e:
        log("Direct Kaggle download failed:", repr(e), "-> trying kagglehub")
        import kagglehub
        root = find_root(Path(kagglehub.dataset_download(KAGGLE_DATASET)))
    root_file.write_text(str(root))
    return root


def build_index(root):
    L = root / "Lung Segmentation Data" / "Lung Segmentation Data"
    I = root / "Infection Segmentation Data" / "Infection Segmentation Data"
    rows = []
    for s in SPLITS:
        for ci, c in enumerate(CLASSES):
            idir, mdir = L / s / c / "images", L / s / c / "lung masks"
            names = sorted(n for n in os.listdir(idir) if n.lower().endswith(".png"))
            for n in names:
                ip = I / s / c / "infection masks" / n
                if not (mdir / n).exists():
                    raise FileNotFoundError(f"missing lung mask for {s}/{c}/{n}")
                rows.append((s.lower(), ci, c, n, str(idir / n), str(mdir / n), str(ip) if ip.exists() else ""))
    df = pd.DataFrame(rows, columns=["split", "label", "cls", "fname", "img_path", "lung_path", "inf_path"])
    counts = {s: tuple(int(((df.split == s.lower()) & (df.label == i)).sum()) for i in range(3)) for s in SPLITS}
    log("Counts per split (COVID-19, Non-COVID, Normal):", counts)
    if not LOCAL_TEST and counts != EXPECTED:
        log("WARNING: counts differ from the Kaggle listing verified on 2026-10-10:", EXPECTED)
    return df


def _load_png(path, is_mask):
    with Image.open(path) as im:
        size, mode = im.size, im.mode
        im = im.convert("L")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR)
        a = np.asarray(im, dtype=np.uint8)
    if is_mask:
        a = (a > (0 if a.max() <= 1 else 127)).astype(np.uint8)
    return a, size, mode


def load_arrays(df):
    cache = WORK / "cache"
    fi, fl, finf, fidx = cache / "images_u8.npy", cache / "lungs_u8.npy", cache / "inf_u8.npy", cache / "index.csv"
    if all(p.exists() for p in (fi, fl, finf, fidx)):
        old = pd.read_csv(fidx, keep_default_na=False)
        if len(old) == len(df) and (old.fname.values == df.fname.values).all() and (old.split.values == df.split.values).all():
            log("Loading decoded arrays from the VM cache")
            return np.load(fi), np.load(fl), np.load(finf), old
    t0 = time.time()
    n = len(df)
    imgs = np.empty((n, IMG_SIZE, IMG_SIZE), np.uint8)
    lungs = np.empty((n, IMG_SIZE, IMG_SIZE), np.uint8)
    sizes, modes = Counter(), Counter()
    workers = min(32, (os.cpu_count() or 4) * 2)

    def work(i):
        a, sz, md = _load_png(df.img_path.iat[i], False)
        m, _, _ = _load_png(df.lung_path.iat[i], True)
        imgs[i] = a; lungs[i] = m
        return sz, md
    with ThreadPoolExecutor(workers) as ex:
        for sz, md in ex.map(work, range(n), chunksize=64):
            sizes[f"{sz[0]}x{sz[1]}"] += 1; modes[md] += 1
    inf_rows = np.where(df.inf_path.values != "")[0]
    infs = np.zeros((max(1, len(inf_rows)), IMG_SIZE, IMG_SIZE), np.uint8)

    def work_inf(j):
        infs[j] = _load_png(df.inf_path.iat[inf_rows[j]], True)[0]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work_inf, range(len(inf_rows))))
    df = df.copy()
    df["inf_pos"] = -1
    df.loc[df.index[inf_rows], "inf_pos"] = np.arange(len(inf_rows))
    df["has_inf"] = False
    if len(inf_rows):
        df.loc[df.index[inf_rows], "has_inf"] = infs[: len(inf_rows)].reshape(len(inf_rows), -1).max(1) > 0
    df["lung_frac"] = lungs.reshape(n, -1).mean(1)
    np.save(fi, imgs); np.save(fl, lungs); np.save(finf, infs)
    df.to_csv(fidx, index=False)
    write_json(DIRS["audit"] / "image_formats.json", dict(original_sizes=dict(sizes), modes=dict(modes)))
    log(f"Decoded {n} images + masks in {time.time() - t0:.0f} s; original sizes: {dict(sizes.most_common(5))}; modes: {dict(modes)}")
    return imgs, lungs, infs, df


DATA_ROOT = get_dataset_root()
DF = build_index(DATA_ROOT)
IMG, LUNG, INF, DF = load_arrays(DF)
DF["inf_pos"] = DF["inf_pos"].astype(int)
DF["has_inf"] = DF["has_inf"].astype(str).str.lower().isin(["true", "1"])
LAB_NP = DF.label.values.astype(np.int64)
TRAIN_IDX = np.where(DF.split.values == "train")[0]
VAL_IDX = np.where(DF.split.values == "val")[0]
TEST_IDX = np.where(DF.split.values == "test")[0]
log(f"train {len(TRAIN_IDX)} | val {len(VAL_IDX)} | test {len(TEST_IDX)} | infection masks: {int(DF.has_inf.sum())} non-empty")

# %% [markdown]
# ## 2. Data audit: exact and near-duplicate leakage across splits
# COVID-QU-Ex has no patient identifiers, so patient-level separation cannot be verified directly. As the closest available
# check, every test image is compared exhaustively with every training and validation image (and every validation image with
# every training image): exact duplicates (identical pixels) and near-duplicates (a reference image with 64-bit DCT perceptual-hash
# Hamming distance ≤ 6 **and** normalized cross-correlation ≥ 0.97 of the 64×64 thumbnails). Results are reported on the full
# official test set and on the de-duplicated test set.

# %%
def phash_bits(imgs, chunk=2048):
    out = np.empty((len(imgs), 64), bool)
    f = IMG_SIZE // 32
    for i in range(0, len(imgs), chunk):
        x = imgs[i:i + chunk].astype(np.float32).reshape(-1, 32, f, 32, f).mean(axis=(2, 4))
        d = dctn(x, type=2, axes=(1, 2), norm="ortho")[:, :8, :8].reshape(-1, 64)
        out[i:i + chunk] = d > np.median(d[:, 1:], axis=1, keepdims=True)
    return out


def thumbs(imgs, idx):
    f = IMG_SIZE // 64
    t = imgs[idx].astype(np.float32).reshape(-1, 64, f, 64, f).mean(axis=(2, 4)).reshape(len(idx), -1)
    t -= t.mean(1, keepdims=True)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)


def dedup_audit():
    f = DIRS["audit"] / "dedup_per_image.csv"
    if f.exists():
        return pd.read_csv(f)
    t0 = time.time()
    md5 = np.array([hashlib.md5(IMG[i].tobytes()).hexdigest() for i in range(len(IMG))])
    H = torch.from_numpy(phash_bits(IMG)).to(DEVICE).float() * 2 - 1          # (N,64) +-1 codes
    TH = torch.cat([torch.from_numpy(thumbs(IMG, np.arange(i, min(i + 4096, len(IMG))))) for i in range(0, len(IMG), 4096)]).to(DEVICE)
    rows = []
    with tf32_off():                                   # exhaustive comparison: every query against every reference image
        for query_split, ref_splits in (("test", ("train", "val")), ("val", ("train",))):
            q = np.where(DF.split.values == query_split)[0]
            r = np.where(DF.split.isin(ref_splits).values)[0]
            rt = torch.from_numpy(r).to(DEVICE)
            Hr, Tr = H[rt], TH[rt]
            md5_ref = {}
            for j in r:
                md5_ref.setdefault(md5[j], j)
            for s in range(0, len(q), 1024):
                qq = q[s:s + 1024]
                qt = torch.from_numpy(qq).to(DEVICE)
                ham = (64 - H[qt] @ Hr.T) / 2
                ncc = TH[qt] @ Tr.T
                ok = (ham <= CFG["dup_hamming"]) & (ncc >= CFG["dup_ncc"])
                j = torch.where(ok, ncc + 2, ncc).argmax(1)            # a qualifying match if any, otherwise the most similar image
                near, nmatch = ok.any(1).cpu().numpy(), ok.sum(1).cpu().numpy()
                hb = ham.gather(1, j[:, None])[:, 0].cpu().numpy()
                nb = ncc.gather(1, j[:, None])[:, 0].cpu().numpy()
                mrow = r[j.cpu().numpy()]
                for k in range(len(qq)):
                    exact = md5[qq[k]] in md5_ref
                    m = md5_ref[md5[qq[k]]] if exact else mrow[k]
                    rows.append(dict(row=int(qq[k]), split=query_split, exact_dup=bool(exact), near_dup=bool(near[k] or exact),
                                     n_matching_references=int(nmatch[k]), hamming=int(round(float(hb[k]))), ncc=float(nb[k]), match_row=int(m),
                                     match_split=DF.split.iat[m], match_fname=DF.fname.iat[m],
                                     label=int(LAB_NP[qq[k]]), match_label=int(LAB_NP[m])))
    del H, TH
    torch.cuda.empty_cache()
    res = pd.DataFrame(rows)
    res.to_csv(f, index=False)
    log(f"Duplicate audit done in {time.time() - t0:.0f} s")
    return res


DUP = dedup_audit()
DF["dup_flag"] = False
DF.loc[DUP.row.values, "dup_flag"] = DUP.near_dup.values
TEST_DEDUP_IDX = TEST_IDX[~DF.dup_flag.values[TEST_IDX]]
audit = []
for s in ("val", "test"):
    d = DUP[DUP.split == s]
    audit.append(dict(split=s, n=len(d), exact_duplicates=int(d.exact_dup.sum()), near_duplicates=int(d.near_dup.sum()),
                      near_duplicates_with_different_label=int((d.near_dup & (d.label != d.match_label)).sum()),
                      near_dup_COVID=int((d.near_dup & (d.label == 0)).sum()), near_dup_NonCOVID=int((d.near_dup & (d.label == 1)).sum()),
                      near_dup_Normal=int((d.near_dup & (d.label == 2)).sum())))
save_table(pd.DataFrame(audit), "T01b_duplicate_audit")
log("Duplicate audit:", audit)
DF[["split", "label", "cls", "fname", "has_inf", "dup_flag", "lung_frac"]].to_csv(DIRS["audit"] / "split_manifest.csv", index=False)

tab = []
for s in SPLITS:
    for ci, c in enumerate(CLASSES):
        m = (DF.split == s.lower()) & (DF.label == ci)
        tab.append(dict(split=s, cls=c, images=int(m.sum()), with_infection_mask=int((m & DF.has_inf).sum()),
                        near_duplicate_in_earlier_split=int((m & DF.dup_flag).sum()), mean_lung_area_fraction=round(float(DF.lung_frac[m].mean()), 3)))
save_table(pd.DataFrame(tab), "T01_dataset")
log(f"De-duplicated test set: {len(TEST_DEDUP_IDX)} of {len(TEST_IDX)} images")

# %% [markdown]
# ## 3. GPU-resident data pipeline
# All decoded images (uint8, 256×256) live in GPU memory; augmentation (rotation ±10°, scale ±10 %, translation ±5 %,
# brightness/contrast ±10 %; **no horizontal flip**, which would create anatomically implausible laterality) and resizing to
# 224×224 run on the GPU with one `grid_sample` call per batch, so the A100 is never starved by the CPU.

# %%
ON_GPU_DATA = GPU_MEM_GB >= 20
T_IMG = torch.from_numpy(IMG)
T_LUNG = torch.from_numpy(LUNG)
T_INF = torch.from_numpy(INF)
if ON_GPU_DATA:
    T_IMG, T_LUNG, T_INF = T_IMG.to(DEVICE), T_LUNG.to(DEVICE), T_INF.to(DEVICE)
LABELS = torch.from_numpy(LAB_NP)
LABELS_DEV = LABELS.to(DEVICE)
INF_POS = torch.from_numpy(DF.inf_pos.values.astype(np.int64))
FILL = float(np.mean([IMG[TRAIN_IDX[i:i + 2048]].mean() for i in range(0, len(TRAIN_IDX), 2048)]) / 255.0)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
log(f"Data on GPU: {ON_GPU_DATA}; mean training intensity (fill value) = {FILL:.4f}")


def _identity_theta(B):
    th = torch.zeros(B, 2, 3, device=DEVICE)
    th[:, 0, 0] = 1
    th[:, 1, 1] = 1
    return th


def _rand(B, gen, *shape):
    return torch.rand((B,) + shape, device=DEVICE, generator=gen) * 2 - 1


def fetch(idx, train=False, lung_mode=None, gen=None, with_lung=False):
    """Rows -> (B,1,224,224) float in [0,1] and an optional lung mask, both on the GPU."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    x = T_IMG[idx.to(T_IMG.device)].to(DEVICE, non_blocking=True).float().div_(255.0).unsqueeze(1)
    B = x.shape[0]
    if train:
        ang = _rand(B, gen) * math.radians(CFG["aug_rot_deg"])
        sc = 1 + _rand(B, gen) * CFG["aug_scale"]
        tx, ty = _rand(B, gen) * CFG["aug_translate"] * 2, _rand(B, gen) * CFG["aug_translate"] * 2
        c, s = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([c, -s, tx], 1), torch.stack([s, c, ty], 1)], 1)
    else:
        theta = _identity_theta(B)
    grid = F.affine_grid(theta, (B, 1, INPUT, INPUT), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    lung = None
    if lung_mode is not None or with_lung:
        m = T_LUNG[idx.to(T_LUNG.device)].to(DEVICE, non_blocking=True).float().unsqueeze(1)
        lung = F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False) > 0.5
    if train:
        b = 1 + _rand(B, gen, 1, 1, 1) * CFG["aug_brightness"]
        cf = 1 + _rand(B, gen, 1, 1, 1) * CFG["aug_contrast"]
        mu = x.mean((2, 3), keepdim=True)
        x = ((x - mu) * cf + mu) * b
    x = x.clamp(0, 1)
    if lung_mode == "lung_only":
        x = torch.where(lung, x, torch.full_like(x, FILL))
    elif lung_mode == "no_lung":
        x = torch.where(lung, torch.full_like(x, FILL), x)
    return x, lung


def fetch_inf(idx):
    idx = torch.as_tensor(idx, dtype=torch.long)
    pos = INF_POS[idx]
    out = torch.zeros(len(idx), 1, INPUT, INPUT, dtype=torch.bool, device=DEVICE)
    valid = pos >= 0
    if valid.any():
        m = T_INF[pos[valid].to(T_INF.device)].to(DEVICE).float().unsqueeze(1)
        grid = F.affine_grid(_identity_theta(m.shape[0]), (m.shape[0], 1, INPUT, INPUT), align_corners=False)
        out[valid.to(DEVICE)] = F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False) > 0.5
    return out


def to_input(x01):
    return ((x01.expand(-1, 3, -1, -1) - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def cap_idx(idx, cap, seed=0):
    if not cap or len(idx) <= cap:
        return idx
    rng = np.random.RandomState(seed)
    per = []
    for c in range(3):
        ic = idx[LAB_NP[idx] == c]
        per.append(rng.choice(ic, size=min(len(ic), cap // 3), replace=False))
    return np.sort(np.concatenate(per))


# augmentation sanity figure
try:
    g = torch.Generator(device=DEVICE); g.manual_seed(0)
    demo = np.array([TRAIN_IDX[np.where(LAB_NP[TRAIN_IDX] == c)[0][0]] for c in range(3)])
    xa, _ = fetch(np.repeat(demo, 4), train=True, gen=g)
    fig, axes = plt.subplots(3, 4, figsize=(6.4, 4.9))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(xa[k, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1); ax.axis("off")
        if k % 4 == 0:
            ax.set_title(CLASSES[k // 4], fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(DIRS["figures"] / "S_augmentation_examples.png", dpi=200); plt.close(fig)
except Exception as e:
    log("augmentation figure skipped:", repr(e))

# %% [markdown]
# ## 4. Models: backbones × heads
# * **default** – the torchvision head (ResNet / DenseNet: GAP → Linear; ConvNeXt: GAP → LayerNorm → Linear; VGG-19: 3-layer MLP).
# * **flat** – the faithful head: the backbone's final feature map **A** → Flatten → one Linear layer (backbone unchanged).
# * **flat_trunc** – the modification described in the original manuscript: last normalisation/activation (and, for VGG-19,
#   the last max-pool) removed, then Flatten → Linear.
# * **flat_lamprou** – ResNets only: the last residual block replaced by the plain block of Lamprou et al. (2024) (MIT-licensed
#   repository `vggls/medical_xai`), then Flatten → Linear.
#
# Every model is written as `features(x) → A` and `head(A) → logits`; all CAM methods are computed at **A**, the tensor that
# feeds the head, using exact autograd of the head (no hooks).

# %%
BACKBONES = ["resnet34", "resnet50", "vgg19", "densenet121", "convnext_tiny"]
BB_LABEL = {"resnet34": "ResNet-34", "resnet50": "ResNet-50", "vgg19": "VGG-19", "densenet121": "DenseNet-121", "convnext_tiny": "ConvNeXt-T"}
HEAD_LABEL = {"default": "Default head", "flat": "Flatten head", "flat_trunc": "Flatten, truncated backbone", "flat_lamprou": "Flatten, Lamprou block"}
HEAD_DESC = {
    "default": "torchvision head (GAP->Linear; ConvNeXt GAP->LN->Linear; VGG-19 MLP)",
    "flat": "backbone unchanged; final feature map -> Flatten -> Linear",
    "flat_trunc": "last normalization/activation removed (VGG-19: last ReLU and max-pool removed) -> Flatten -> Linear",
    "flat_lamprou": "last residual block replaced by plain conv block (Lamprou et al. 2024) -> Flatten -> Linear",
}


class CXRNet(nn.Module):
    def __init__(self, features, head):
        super().__init__()
        self.features, self.head = features, head

    def forward(self, x):
        return self.head(self.features(x))


class TruncBasicBlock(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.conv2, self.downsample = b.conv1, b.bn1, b.relu, b.conv2, b.downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        return self.conv2(self.relu(self.bn1(self.conv1(x)))) + idt


class TruncBottleneck(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.conv1, self.bn1, self.conv2, self.bn2, self.conv3 = b.conv1, b.bn1, b.conv2, b.bn2, b.conv3
        self.relu, self.downsample = b.relu, b.downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.conv3(out) + idt


def lamprou_block(backbone):
    if backbone == "resnet34":
        return nn.Sequential(OrderedDict([("conv1", nn.Conv2d(512, 512, 3, 1, 1, bias=False)), ("bn1", nn.BatchNorm2d(512)),
                                          ("relu", nn.ReLU(inplace=True)), ("conv2", nn.Conv2d(512, 512, 3, 1, 1, bias=False))]))
    return nn.Sequential(OrderedDict([("conv1", nn.Conv2d(2048, 512, 1, bias=False)), ("bn1", nn.BatchNorm2d(512)),
                                      ("conv2", nn.Conv2d(512, 512, 3, 1, 1, bias=False)), ("bn2", nn.BatchNorm2d(512)),
                                      ("conv3", nn.Conv2d(512, 2048, 1, bias=False))]))


def _weights(backbone):
    if not PRETRAINED:
        return None
    return {"resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1, "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2,
            "vgg19": tvm.VGG19_Weights.IMAGENET1K_V1, "densenet121": tvm.DenseNet121_Weights.IMAGENET1K_V1,
            "convnext_tiny": tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1}[backbone]


def heads_for(backbone):
    return ["default", "flat", "flat_trunc"] + (["flat_lamprou"] if backbone.startswith("resnet") else [])


def build_model(backbone, head, num_classes=3):
    """Returns (model, info). info['final'] is the new classifier; info['new'] lists other newly initialised layers."""
    w = _weights(backbone)
    new = []
    if backbone.startswith("resnet"):
        m = getattr(tvm, backbone)(weights=w)
        layer4 = m.layer4
        if head == "flat_trunc":
            last = layer4[-1]
            layer4[-1] = TruncBasicBlock(last) if isinstance(last, tvm.resnet.BasicBlock) else TruncBottleneck(last)
        elif head == "flat_lamprou":
            layer4[-1] = lamprou_block(backbone)
            new.append(f"features.7.{len(layer4) - 1}.")
        feats = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, layer4)
        C, hw, last_stage = (512 if backbone == "resnet34" else 2048), 7, ["features.7"]
        hd = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(C, num_classes)) if head == "default" else \
            nn.Sequential(nn.Flatten(1), nn.Linear(C * hw * hw, num_classes))
        linear = True
    elif backbone == "densenet121":
        m = tvm.densenet121(weights=w)
        C, hw = 1024, 7
        if head == "flat_trunc":
            f = m.features
            del f.norm5
            feats, last_stage = f, ["features.denseblock4"]
        else:
            feats, last_stage = nn.Sequential(m.features, nn.ReLU(inplace=False)), ["features.0.denseblock4", "features.0.norm5"]
        hd = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(C, num_classes)) if head == "default" else \
            nn.Sequential(nn.Flatten(1), nn.Linear(C * hw * hw, num_classes))
        linear = True
    elif backbone == "vgg19":
        m = tvm.vgg19(weights=w)
        C = 512
        if head == "flat_trunc":
            feats, hw = m.features[:-2], 14
            last_stage = [f"features.{i}" for i in (28, 30, 32, 34)]
        else:
            feats, hw = m.features, 7
            last_stage = [f"features.{i}" for i in (28, 30, 32, 34)]
        if head == "default":
            cls = m.classifier
            cls[6] = nn.Linear(4096, num_classes)
            hd = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten(1), cls)
            linear = False
        else:
            hd = nn.Sequential(nn.Flatten(1), nn.Linear(C * hw * hw, num_classes))
            linear = True
    elif backbone == "convnext_tiny":
        m = tvm.convnext_tiny(weights=w)
        C, hw = 768, 7
        ln = m.classifier[0]
        if head == "default":
            feats, last_stage = m.features, ["features.7"]
            hd = nn.Sequential(nn.AdaptiveAvgPool2d(1), ln, nn.Flatten(1), nn.Linear(C, num_classes))
            linear = False
        elif head == "flat":
            feats, last_stage = nn.Sequential(m.features, ln), ["features.0.7"]
            hd = nn.Sequential(nn.Flatten(1), nn.Linear(C * hw * hw, num_classes))
            linear = True
        else:
            feats, last_stage = m.features, ["features.7"]
            hd = nn.Sequential(nn.Flatten(1), nn.Linear(C * hw * hw, num_classes))
            linear = True
    else:
        raise ValueError(backbone)
    final = [m_ for m_ in hd.modules() if isinstance(m_, nn.Linear)][-1]
    nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)          # zero-initialised classifier: logits start at 0
    final_name = [n_ for n_, m_ in hd.named_modules() if m_ is final][0]
    model = CXRNet(feats, hd)
    flat_positions = 1 if head == "default" else hw * hw
    info = dict(backbone=backbone, head=head, A_shape=(C, hw, hw), fc_inputs=(C if head == "default" and backbone != "vgg19" else C * hw * hw),
                head_linear=linear, gap_first=(head == "default" and backbone != "vgg19"),
                new=new, final=f"head.{final_name}.", flat_positions=flat_positions, last_stage=last_stage)
    return model, info


def param_groups(model, info, lr_b, lr_h, wd):
    """Three learning rates: pretrained backbone (lr_b), other new layers (lr_h), final classifier (lr_h / number of
    spatial positions it reads). The last rule equalises the per-step change of the logits between a GAP head (C inputs)
    and a Flatten head (C*h*w inputs of similar magnitude); without it, Adam moves a Flatten head's logits h*w times faster."""
    groups = OrderedDict()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(info["final"]):
            kind, lr = "final", lr_h / info["flat_positions"]
        elif any(name.startswith(pfx) for pfx in info["new"]):
            kind, lr = "new", lr_h
        else:
            kind, lr = "backbone", lr_b
        groups.setdefault((kind, p.ndim > 1), dict(params=[], lr=lr, weight_decay=wd if p.ndim > 1 else 0.0))["params"].append(p)
    return list(groups.values())


CAM_METHODS =["gradcam", "hirescam", "gradcampp", "xgradcam", "layercam"]


def cam_raw_maps(model, x, target, methods):
    """CAM-family raw maps at the head input A (pytorch-grad-cam definitions), plus logits and the head reference s(A=0).
    Everything here runs in exact float32 (TF32 off): TF32 rounding of the feature extractor changes the maps by up to
    ~1e-2 after the cancellation-prone channel sums, whereas in exact float32 they equal pytorch-grad-cam to ~1e-6."""
    with tf32_off():
        with torch.no_grad():
            A = model.features(x.float()).float()
        A = A.detach().requires_grad_(True)
        logits = model.head(A)
        sel = logits.gather(1, target.view(-1, 1)).sum()
        g, = torch.autograd.grad(sel, A)
        with torch.no_grad():
            s0 = model.head(torch.zeros_like(A[:1]))
    A, g, logits = A.detach(), g.detach(), logits.detach()
    out = {}
    if "gradcam" in methods:
        out["gradcam"] = (g.mean((2, 3), keepdim=True) * A).sum(1)
    if "hirescam" in methods:
        out["hirescam"] = (g * A).sum(1)
    if "gradcampp" in methods:
        g2 = g * g; g3 = g2 * g
        sA = A.sum((2, 3), keepdim=True)
        aij = g2 / (2 * g2 + sA * g3 + 1e-6)
        aij = torch.where(g != 0, aij, torch.zeros_like(aij))
        out["gradcampp"] = ((g.clamp(min=0) * aij).sum((2, 3), keepdim=True) * A).sum(1)
    if "xgradcam" in methods:
        sA = A.sum((2, 3), keepdim=True)
        out["xgradcam"] = ((g * A / (sA + 1e-7)).sum((2, 3), keepdim=True) * A).sum(1)
    if "layercam" in methods:
        out["layercam"] = (g.clamp(min=0) * A).sum(1)
    sc = logits.gather(1, target.view(-1, 1)).squeeze(1)
    s0c = s0[0][target]
    return out, logits, sc, s0c, A


def unit_tests():
    rows = []
    seed_all(0)
    for bb in BACKBONES:
        for hd in heads_for(bb):
            m, info = build_model(bb, hd)
            with torch.no_grad():                    # non-trivial head weights/biases for the identity checks
                for lin in m.head.modules():
                    if isinstance(lin, nn.Linear):
                        lin.weight.normal_(0, 0.01); lin.bias.normal_(0, 0.1)
            m = m.to(DEVICE).eval()
            x = torch.randn(2, 3, INPUT, INPUT, device=DEVICE)
            with torch.no_grad():
                A = m.features(x)
            assert tuple(A.shape[1:]) == info["A_shape"], (bb, hd, A.shape)
            t = torch.tensor([0, 2], device=DEVICE)
            with tf32_off():
                raw, lg, sc, s0c, _ = cam_raw_maps(m, x, t, ["gradcam", "hirescam"])
            delta = sc - s0c
            err_h = ((raw["hirescam"].sum((1, 2)) - delta).abs() / (delta.abs() + 1e-6)).max().item()
            err_g = ((raw["gradcam"].sum((1, 2)) - delta).abs() / (delta.abs() + 1e-6)).max().item()
            same = ((raw["gradcam"] - raw["hirescam"]).abs().max() / (raw["hirescam"].abs().max() + 1e-12)).item()
            n_par = sum(p.numel() for p in m.parameters())
            n_head = sum(p.numel() for p in m.head.parameters())
            rows.append(dict(backbone=BB_LABEL[bb], head=hd, head_description=HEAD_DESC[hd], target_layer_shape="x".join(map(str, info["A_shape"])),
                             fc_input_features=info["fc_inputs"], total_params_M=round(n_par / 1e6, 2), head_params=n_head,
                             head_affine_in_A=info["head_linear"],
                             theory_hirescam_complete=info["head_linear"],
                             theory_gradcam_equals_hirescam=info["gap_first"],
                             check_hirescam_relerr=err_h, check_gradcam_relerr=err_g,
                             check_gradcam_hirescam_maxdiff=same))
            if info["head_linear"]:
                assert err_h < 1e-3, f"HiResCAM completeness failed for {bb}/{hd}: {err_h}"
            if info["gap_first"]:
                assert same < 1e-4, f"Grad-CAM != HiResCAM although the head starts with GAP ({bb}/{hd}): {same}"
            del m; torch.cuda.empty_cache()
    return pd.DataFrame(rows)


ARCH = unit_tests()
save_table(ARCH, "T02_architectures")
log("Unit tests passed: completeness and Grad-CAM/HiResCAM identities hold where theory predicts.")
print(ARCH[["backbone", "head", "target_layer_shape", "fc_input_features", "total_params_M", "theory_hirescam_complete", "theory_gradcam_equals_hirescam"]].to_string())

# %% [markdown]
# ## 5. Explanations: methods
# All CAM methods follow the definitions of `pytorch-grad-cam` v1.5.7 (ReLU → per-image min-max → bilinear resize), computed at the
# head input **A**. Score-CAM follows the same library (softmax over channel scores). Integrated Gradients: black baseline, 32
# midpoint steps. RISE: 2,000 masks, 7×7 grid, p = 0.5 (Petsiuk et al., 2018). Random: smooth random 7×7 map (lower bound).

# %%
ALL_METHODS = ["gradcam", "hirescam", "gradcampp", "xgradcam", "layercam", "scorecam", "ig", "rise", "random"]
METHOD_LABEL = {"gradcam": "Grad-CAM", "hirescam": "HiResCAM", "gradcampp": "Grad-CAM++", "xgradcam": "XGrad-CAM", "layercam": "LayerCAM",
                "scorecam": "Score-CAM", "ig": "Integrated Gradients", "rise": "RISE", "random": "Random"}
BASELINE = ((torch.zeros(1, 3, 1, 1, device=DEVICE) - MEAN) / STD)   # black image in normalized space


def minmax(r):
    r = r - r.amin((1, 2), keepdim=True)
    return r / (r.amax((1, 2), keepdim=True) + 1e-7)


def postprocess(raw):
    """pytorch-grad-cam 1.5.7 post-processing: ReLU -> min-max -> bilinear resize -> min-max (BaseCAM.aggregate_multi_layers)."""
    r = minmax(raw.float().clamp(min=0))
    if r.shape[-1] != INPUT:
        r = F.interpolate(r.unsqueeze(1), size=(INPUT, INPUT), mode="bilinear", align_corners=False).squeeze(1)
    return minmax(r)


@torch.no_grad()
def forward_probs(fwd, xs):
    out = []
    for s in range(0, xs.shape[0], CFG["fwd_chunk"]):
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            out.append(torch.softmax(fwd(xs[s:s + CFG["fwd_chunk"]].contiguous(memory_format=torch.channels_last)).float(), 1))
    return torch.cat(out)


@torch.no_grad()
def scorecam_raw(fwd, x, target, A):
    maps = []
    C = A.shape[1]
    for i in range(A.shape[0]):
        up = F.interpolate(A[i:i + 1], size=(INPUT, INPUT), mode="bilinear", align_corners=True)[0]
        mn, mx = up.amin((1, 2), keepdim=True), up.amax((1, 2), keepdim=True)
        up = (up - mn) / (mx - mn + 1e-8)
        sc = []
        for c0 in range(0, C, CFG["scorecam_chunk"]):
            inp = (x[i:i + 1] * up[c0:c0 + CFG["scorecam_chunk"]].unsqueeze(1)).contiguous(memory_format=torch.channels_last)
            sc.append(fwd(inp).float()[:, target[i]])          # fp32: the channel softmax is sensitive to logit rounding
        wgt = torch.softmax(torch.cat(sc), 0)
        maps.append((wgt.view(C, 1, 1) * A[i]).sum(0))
    return torch.stack(maps)


def make_rise_masks(N, s, p, seed=2018):
    g = torch.Generator(); g.manual_seed(seed)
    cell = math.ceil(INPUT / s); up = (s + 1) * cell
    grid = (torch.rand(N, 1, s, s, generator=g) < p).float()
    big = F.interpolate(grid, size=(up, up), mode="bilinear", align_corners=False)
    sx = torch.randint(0, cell, (N,), generator=g); sy = torch.randint(0, cell, (N,), generator=g)
    return torch.stack([big[i, 0, sx[i]:sx[i] + INPUT, sy[i]:sy[i] + INPUT] for i in range(N)]).to(DEVICE, torch.float16)


@torch.no_grad()
def rise_map(fwd, x, target, masks):
    N, p = masks.shape[0], CFG["rise_p"]
    out = []
    for i in range(x.shape[0]):
        sal = torch.zeros(INPUT, INPUT, device=DEVICE)
        for c0 in range(0, N, CFG["rise_chunk"]):
            m = masks[c0:c0 + CFG["rise_chunk"]].float().unsqueeze(1)
            pr = forward_probs(fwd, x[i:i + 1] * m)[:, target[i]]
            sal += (pr.view(-1, 1, 1) * m[:, 0]).sum(0)
        out.append(sal / (N * p))
    return torch.stack(out)


def integrated_gradients(fwd, x, target, steps, max_batch):
    B = x.shape[0]
    x0 = BASELINE.expand_as(x)
    alphas = (torch.arange(steps, device=DEVICE, dtype=torch.float32) + 0.5) / steps
    total = torch.zeros_like(x)
    per = max(1, max_batch // B)
    for a0 in range(0, steps, per):
        al = alphas[a0:a0 + per]; k = len(al)
        xs = (x0.unsqueeze(1) + al.view(1, k, 1, 1, 1) * (x - x0).unsqueeze(1)).reshape(B * k, *x.shape[1:])
        xs = xs.contiguous(memory_format=torch.channels_last).requires_grad_(True)
        lg = fwd(xs)
        sel = lg.gather(1, target.repeat_interleave(k).view(-1, 1)).sum()
        gr, = torch.autograd.grad(sel, xs)
        total += gr.view(B, k, *x.shape[1:]).sum(1)
    return ((x - x0) * total / steps).sum(1)


def random_maps(rows):
    r = []
    for rw in rows:
        g = torch.Generator(); g.manual_seed(int(rw) + 7)
        r.append(torch.rand(1, 1, 7, 7, generator=g))
    r = torch.cat(r).to(DEVICE)
    return minmax(F.interpolate(r, size=(INPUT, INPUT), mode="bilinear", align_corners=False).squeeze(1))


class Explainable:
    """Wraps a single CXRNet or a logit-averaging ensemble of CXRNets with a common explain() interface."""

    def __init__(self, members, name, lung_mode=None):
        self.members = members
        self.name, self.lung_mode = name, lung_mode
        self.is_ensemble = len(members) > 1

    def fwd(self, x):
        if not self.is_ensemble:
            return self.members[0](x)
        return torch.stack([m(x).float() for m in self.members]).mean(0)

    def cam(self, x, target, methods):
        """Raw CAM-family maps. For an ensemble with logit averaging, d(mean logit)/dA_k = (1/K) d(logit_k)/dA_k, so the
        ensemble map is the mean of the member maps (resampled sum-preservingly to the finest member grid) and HiResCAM
        stays exactly complete: sum(map) = mean_k(logit_k - head_k(0))."""
        K = len(self.members)
        parts, sc, s0, A_last = [], 0.0, 0.0, None
        for mem in self.members:
            raw, lg, scm, s0m, A = cam_raw_maps(mem, x, target, methods)
            parts.append(raw)
            sc, s0, A_last = sc + scm / K, s0 + s0m / K, A
        if K == 1:
            return parts[0], sc, s0, A_last
        g = max(v.shape[-1] for p in parts for v in p.values())
        acc = {}
        for p in parts:
            for k, v in p.items():
                f = g // v.shape[-1]
                if v.shape[-1] * f != g:
                    raise ValueError("ensemble members need explanation grids that divide the finest grid")
                if f > 1:
                    v = v.repeat_interleave(f, 1).repeat_interleave(f, 2) / (f * f)
                acc[k] = acc.get(k, 0) + v / K
        return acc, sc, s0, A_last

    def explain(self, x, target, methods, rows, ctx, with_raw=True):
        vis, raw, info = {}, {}, {}
        cm = [m for m in methods if m in CAM_METHODS]
        if cm or "scorecam" in methods:
            raw, sc, s0, A = self.cam(x, target, cm if cm else ["hirescam"])
            raw = {k: v for k, v in raw.items() if k in cm}
            info["score"], info["s0"] = sc, s0
            if "scorecam" in methods and not self.is_ensemble:
                raw["scorecam"] = scorecam_raw(self.fwd, x, target, A)
        for k, v in raw.items():
            vis[k] = postprocess(v)
        if "ig" in methods:
            a = integrated_gradients(self.fwd, x, target, CFG["ig_steps"], ctx["ig_batch"])
            vis["ig"] = postprocess(a)
            info["ig_sum"] = a.sum((1, 2)).detach()
        if "rise" in methods:
            vis["rise"] = minmax(rise_map(self.fwd, x, target, ctx["rise_masks"]))
        if "random" in methods:
            vis["random"] = random_maps(rows)
        return vis, (raw if with_raw else {}), info


def gaussian_blur(x, klen=11, sig=5.0):
    ax = torch.arange(klen, device=x.device, dtype=torch.float32) - klen // 2
    k1 = torch.exp(-ax ** 2 / (2 * sig ** 2)); k1 = k1 / k1.sum()
    k2 = (k1[:, None] * k1[None, :]).view(1, 1, klen, klen).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, k2, padding=klen // 2, groups=x.shape[1])


def rank_map(scores, gen):
    """scores (B,L) -> rank of every element (0 = most important), ties broken at random (reproducibly)."""
    s = scores + CFG["tie_eps"] * torch.rand(scores.shape, device=scores.device, generator=gen)
    order = s.argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(scores.shape[1], device=scores.device).expand_as(order))
    return rank


def aopc_curves(fwd, x, vis, target, grid, noise, gen):
    """MoRF region perturbation (Samek et al. 2017; Lamprou et al. 2024): all grid*grid regions replaced one by one with noise."""
    B, r = x.shape[0], INPUT // grid
    L = grid * grid
    rank = rank_map(F.avg_pool2d(vis.unsqueeze(1), r).flatten(1), gen)
    rmap = rank.view(B, 1, grid, grid).repeat_interleave(r, 2).repeat_interleave(r, 3)
    t = torch.arange(L + 1, device=DEVICE).view(-1, 1, 1, 1)
    curves = torch.empty(B, L + 1, device=DEVICE)
    for i in range(B):
        xs = torch.where(rmap[i:i + 1] < t, noise[i:i + 1], x[i:i + 1])
        curves[i] = forward_probs(fwd, xs)[:, target[i]]
    return curves


def delins_curves(fwd, x, vis, target, blurred, gen):
    """Deletion (zero substrate) and insertion (blur substrate) games (Petsiuk et al. 2018)."""
    B, n = x.shape[0], CFG["delins_steps"]
    step = math.ceil(INPUT * INPUT / n)
    rank = rank_map(vis.flatten(1), gen).view(B, 1, INPUT, INPUT)
    t = (torch.arange(n + 1, device=DEVICE) * step).view(-1, 1, 1, 1)
    dc, ic = torch.empty(B, n + 1, device=DEVICE), torch.empty(B, n + 1, device=DEVICE)
    for i in range(B):
        m = rank[i:i + 1] < t
        dc[i] = forward_probs(fwd, torch.where(m, torch.zeros_like(x[i:i + 1]), x[i:i + 1]))[:, target[i]]
        ic[i] = forward_probs(fwd, torch.where(m, x[i:i + 1], blurred[i:i + 1]))[:, target[i]]
    return dc, ic


def norm_auc(c):
    return (c.sum(1) - c[:, 0] / 2 - c[:, -1] / 2) / (c.shape[1] - 1)


def plausibility(vis, lung, inf, has_inf):
    tot = vis.flatten(1).sum(1) + 1e-12
    out = dict(lung_mass=((vis * lung[:, 0]).flatten(1).sum(1) / tot), lung_area=lung[:, 0].float().flatten(1).mean(1))
    B = vis.shape[0]
    rma = torch.full((B,), float("nan"), device=DEVICE); rra = rma.clone(); pg = rma.clone(); pgt = rma.clone()
    if has_inf.any():
        im = inf[:, 0]
        tol = CFG["pointing_tol_px"]
        dil = F.max_pool2d(im.float().unsqueeze(1), 2 * tol + 1, stride=1, padding=tol)[:, 0] > 0
        flat = vis.flatten(1)
        am = flat.argmax(1)
        for i in torch.where(has_inf)[0].tolist():
            Kp = int(im[i].sum())
            if Kp == 0:
                continue
            rma[i] = (vis[i] * im[i]).sum() / tot[i]
            top = flat[i].topk(Kp).indices
            rra[i] = im[i].flatten()[top].float().mean()
            pg[i] = im[i].flatten()[am[i]].float()
            pgt[i] = dil[i].flatten()[am[i]].float()
    out.update(inf_mass=rma, inf_rank_acc=rra, pointing=pg, pointing_tol=pgt)
    return out


def spearman_rows(a, b):
    a = F.adaptive_avg_pool2d(a.unsqueeze(1), 56).flatten(1).cpu().numpy()
    b = F.adaptive_avg_pool2d(b.unsqueeze(1), 56).flatten(1).cpu().numpy()
    out = []
    for i in range(len(a)):
        if a[i].std() < 1e-12 or b[i].std() < 1e-12:
            out.append(float("nan"))
        else:
            out.append(float(stats.spearmanr(a[i], b[i])[0]))
    return out


def reinit_(module, seed):
    torch.manual_seed(seed)
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if hasattr(m, "reset_running_stats"):
            m.reset_running_stats()


# Optional cross-check against the reference library (pytorch-grad-cam); never required for the main results.
def crosscheck_pytorch_grad_cam(model, x, target):
    try:
        import importlib.util
        if importlib.util.find_spec("pytorch_grad_cam") is None:
            if LOCAL_TEST:
                return None
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "grad-cam==1.5.7"], check=True, timeout=600)
            importlib.invalidate_caches()
        from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, XGradCAM, LayerCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as e:
        log("pytorch-grad-cam cross-check skipped:", repr(e))
        return None
    ours, _, _, _, _ = cam_raw_maps(model, x, target, CAM_METHODS)
    rows = []
    m32 = copy.deepcopy(model).float().eval()
    for name, cls in [("gradcam", GradCAM), ("hirescam", HiResCAM), ("gradcampp", GradCAMPlusPlus), ("xgradcam", XGradCAM), ("layercam", LayerCAM)]:
        try:
            with tf32_off():
                cam = cls(model=m32, target_layers=[m32.features])
                ref = torch.from_numpy(cam(input_tensor=x.float(), targets=[ClassifierOutputTarget(int(t)) for t in target])).to(DEVICE)
            mine = postprocess(ours[name])
            diff = (mine - ref).abs().max().item()
            corr = float(np.corrcoef(mine.flatten().cpu().numpy(), ref.flatten().cpu().numpy())[0, 1])
            rows.append(dict(method=METHOD_LABEL[name], max_abs_diff=diff, pearson_r=corr))
        except Exception as e:
            rows.append(dict(method=METHOD_LABEL[name], max_abs_diff=float("nan"), pearson_r=float("nan"), note=repr(e)[:200]))
    del m32
    return pd.DataFrame(rows)


def xai_selftest(run_result):
    """Runs every explanation method and metric once, on six test images, with the first trained Flatten-head model,
    plus the cross-check against pytorch-grad-cam. Any problem stops the notebook within minutes instead of hours."""
    f = DIRS["logs"] / "xai_selftest.json"
    if f.exists():
        return
    t0 = time.time()
    ck = torch.load(run_result["ckpt"], map_location="cpu", weights_only=True)
    m, _ = build_model(run_result["backbone"], run_result["head"])
    m.load_state_dict(ck["state_dict"])
    m = m.to(DEVICE).eval().to(memory_format=torch.channels_last)
    rows = np.concatenate([TEST_IDX[LAB_NP[TEST_IDX] == c][:2] for c in range(3)])
    inf_rows = TEST_IDX[DF.has_inf.values[TEST_IDX] & (LAB_NP[TEST_IDX] == 0)]
    if len(inf_rows):
        rows[0] = inf_rows[0]
    ex = Explainable([m], "selftest")
    ctx = dict(rise_masks=make_rise_masks(32, CFG["rise_s"], CFG["rise_p"]), ig_batch=16)
    x01, lung = fetch(rows, train=False, with_lung=True)
    x = to_input(x01)
    with torch.no_grad():
        target = ex.fwd(x).float().argmax(1)
    saved_steps = CFG["ig_steps"]
    CFG["ig_steps"] = 4
    out = {}
    try:
        vis, raw, info = ex.explain(x, target, ALL_METHODS, rows, ctx)
        gen = torch.Generator(device=DEVICE); gen.manual_seed(0)
        noise = to_input(torch.rand(len(rows), 1, INPUT, INPUT, device=DEVICE, generator=gen))
        blurred = gaussian_blur(x)
        inf = fetch_inf(rows)
        has_inf = torch.as_tensor(DF.has_inf.values[rows], device=DEVICE)
        for k in ALL_METHODS:
            v = vis[k]
            if tuple(v.shape) != (len(rows), INPUT, INPUT) or not torch.isfinite(v).all():
                raise RuntimeError(f"self-test: invalid map for {k}: shape {tuple(v.shape)}")
            c = aopc_curves(ex.fwd, x, v, target, min(CFG["aopc_grids"]), noise, gen)
            dc, ic = delins_curves(ex.fwd, x, v, target, blurred, gen)
            pl = plausibility(v, lung, inf, has_inf)
            out[k] = dict(aopc=float((c[:, :1] - c).mean()), deletion=float(norm_auc(dc).mean()), insertion=float(norm_auc(ic).mean()),
                          lung_mass=float(pl["lung_mass"].mean()))
        delta = info["score"] - info["s0"]
        comp = float(((raw["hirescam"].flatten(1).sum(1) - delta).abs() / (delta.abs() + 1e-8)).max())
        out["hirescam_completeness_relerr"] = comp
        if comp > 1e-3:
            raise RuntimeError(f"self-test: HiResCAM completeness error {comp:.3e} on a Flatten-head model (expected < 1e-3)")
        rm = copy.deepcopy(m)
        reinit_(rm.get_submodule("head"), 1)
        v2, _, _ = Explainable([rm.eval()], "rand").explain(x, target, ["hirescam"], rows, ctx)
        out["randomization_spearman_head"] = spearman_rows(vis["hirescam"], v2["hirescam"])
        del rm
        cc = crosscheck_pytorch_grad_cam(m, x[:4], target[:4])
        if cc is not None:
            save_table(cc, "S_crosscheck_pytorch_grad_cam")
            out["crosscheck"] = cc.to_dict("records")
            fin = np.isfinite(cc.max_abs_diff.values)
            worst = float(np.nanmax(cc.max_abs_diff.values)) if fin.any() else float("nan")
            min_r = float(np.nanmin(cc.pearson_r.values)) if fin.any() else float("nan")
            log(f"pytorch-grad-cam cross-check: max |difference| {worst:.2e}, min Pearson r {min_r:.6f}\n{cc.to_string(index=False)}")
            if fin.any() and min_r < 0.999:          # a real implementation mismatch (numerical noise keeps r > 0.9999)
                raise RuntimeError(f"self-test: CAM maps disagree with pytorch-grad-cam (min r {min_r:.4f}, max diff {worst:.3e})")
            if fin.any() and worst > 1e-3:
                log("NOTE: cross-check difference above 1e-3 but correlation ~1 -> floating-point effects only; continuing.")
    finally:
        CFG["ig_steps"] = saved_steps
    out.update(seconds=time.time() - t0, run=run_result["run_id"])
    write_json(f, out)
    log(f"XAI self-test passed in {time.time() - t0:.0f} s (HiResCAM completeness rel. error {comp:.1e}); details: {f}")
    del m, ex
    gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Training
# Full fine-tuning with AdamW: pretrained layers lr 1e-4; new layers lr 1e-3; the zero-initialised final classifier lr 1e-3
# divided by the number of spatial positions it reads (1 for GAP/MLP heads, 49 or 196 for Flatten heads, which keeps the
# per-step change of the logits comparable between heads); weight decay 1e-4 on weights only; 1-epoch linear warm-up then
# cosine decay; batch 128; bf16 autocast; gradient clipping at 5. After the first Flatten-head model, an **XAI self-test** runs
# every method and metric once (and cross-checks the CAMs against pytorch-grad-cam) so that problems surface within minutes.
# **Model selection: the epoch with the lowest unweighted validation
# cross-entropy; training stops after `patience` epochs without improvement** (single, explicit criterion).

# %%
def make_spec(backbone, head, seed, group, balance="none", imbalance="natural", lung_mode=None):
    rid = f"{backbone}__{head}__s{seed}__{balance}__{imbalance}__{lung_mode or 'full'}"
    return dict(run_id=rid, backbone=backbone, head=head, seed=seed, group=group, balance=balance, imbalance=imbalance, lung_mode=lung_mode)


def make_runs():
    runs = [make_spec(bb, hd, 0, "main") for bb in BACKBONES for hd in ("default", "flat")]
    if CFG["head_ablation"]:
        runs += [make_spec(bb, "flat_trunc", 0, "head_ablation") for bb in BACKBONES]
        runs += [make_spec(bb, "flat_lamprou", 0, "head_ablation") for bb in ("resnet34", "resnet50")]
    if CFG["balance_ablation"]:
        runs += [make_spec("resnet50", "flat", 0, "balance_natural", balance=b) for b in ("weights", "sampler", "both")]
        runs += [make_spec("resnet50", "flat", 0, "balance_induced", balance=b, imbalance="covid10") for b in ("none", "weights", "sampler", "both")]
    if CFG["lung_masked"]:
        runs += [make_spec(bb, "flat", 0, "lung_masked", lung_mode="lung_only") for bb in ("resnet50", "densenet121")]
    runs += [make_spec(bb, hd, s, "main") for s in CFG["seeds"] if s != 0 for bb in BACKBONES for hd in ("default", "flat")]
    return runs


RUNS = make_runs()
log(f"{len(RUNS)} training runs planned")


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece_score(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def cls_metrics(logits, y):
    p = softmax_np(np.asarray(logits, np.float64))
    yhat = p.argmax(1)
    cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    out = dict(acc=float((yhat == y).mean()), bal_acc=float(rec.mean()), macro_f1=float(f1.mean()), ece=ece_score(p, y))
    try:
        out["macro_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except Exception:
        out["macro_auc"] = float("nan")
    for c in range(3):
        k = ["covid", "noncovid", "normal"][c]
        out[f"recall_{k}"], out[f"precision_{k}"], out[f"f1_{k}"] = float(rec[c]), float(prec[c]), float(f1[c])
        try:
            out[f"auc_{k}"] = float(roc_auc_score((y == c).astype(int), p[:, c]))
        except Exception:
            out[f"auc_{k}"] = float("nan")
    return out


@torch.no_grad()
def predict(model, idx, lung_mode=None):
    model.eval()
    out = []
    for s in range(0, len(idx), CFG["eval_batch"]):
        x01, _ = fetch(idx[s:s + CFG["eval_batch"]], train=False, lung_mode=lung_mode)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            lg = model(to_input(x01))
        out.append(lg.float().cpu())
    return torch.cat(out).numpy()


def train_one(spec):
    rdir = DIRS["runs"] / spec["run_id"]
    rdir.mkdir(parents=True, exist_ok=True)
    done = rdir / "result.json"
    if done.exists():                         # resume: reuse a finished run only if its artifacts are intact
        try:
            res = json.load(open(done))
            with np.load(rdir / "logits.npz") as z:
                assert {"val", "test", "val_idx", "test_idx"} <= set(z.files), "logits.npz incomplete"
            if res.get("ckpt_on_drive"):
                assert os.path.exists(res["ckpt"]) and os.path.getsize(res["ckpt"]) == res.get("ckpt_bytes"), "checkpoint missing/incomplete"
            return res
        except Exception as e:
            log(f"{spec['run_id']}: stored result is not usable ({e!r}) -> training this run again")
    t_start = time.time()
    seed_all(spec["seed"])
    gen = torch.Generator(device=DEVICE); gen.manual_seed(1000 + spec["seed"])
    gcpu = torch.Generator(); gcpu.manual_seed(2000 + spec["seed"])
    model, info = build_model(spec["backbone"], spec["head"])
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    tr = TRAIN_IDX.copy()
    if spec["imbalance"] == "covid10":
        rng = np.random.RandomState(spec["seed"])
        cov = tr[LAB_NP[tr] == 0]
        keep = rng.choice(cov, size=int(round(CFG["covid10_fraction"] * len(cov))), replace=False)
        tr = np.sort(np.concatenate([tr[LAB_NP[tr] != 0], keep]))
    counts = np.bincount(LAB_NP[tr], minlength=3)
    cw = counts.sum() / (3.0 * counts)                      # mean-normalised inverse class frequency
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE) if spec["balance"] in ("weights", "both") else None)
    samp_p = torch.tensor((1.0 / counts)[LAB_NP[tr]], dtype=torch.double) if spec["balance"] in ("sampler", "both") else None
    opt = torch.optim.AdamW(param_groups(model, info, CFG["lr_backbone"], CFG["lr_head"], CFG["weight_decay"]), fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
    B = CFG["batch"]
    steps = max(1, len(tr) // B)
    if CFG["steps_cap"]:
        steps = min(steps, CFG["steps_cap"])
    total, warm = steps * CFG["max_epochs"], steps * CFG["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    val_idx = cap_idx(VAL_IDX, CFG["eval_cap"])
    yv = LAB_NP[val_idx]
    tr_t = torch.from_numpy(tr)
    hist, best, best_ep, best_state, bad = [], float("inf"), 0, None, 0
    sampled = np.zeros(3, np.int64)
    for ep in range(CFG["max_epochs"]):
        model.train()
        t0 = time.time()
        order = tr_t[torch.multinomial(samp_p, len(tr), replacement=True, generator=gcpu)] if samp_p is not None else tr_t[torch.randperm(len(tr), generator=gcpu)]
        loss_sum = torch.zeros((), device=DEVICE); correct = torch.zeros((), device=DEVICE); n = 0
        for s in range(steps):
            idx = order[s * B:(s + 1) * B]
            if ep == 0:
                sampled += np.bincount(LAB_NP[idx.numpy()], minlength=3)
            x01, _ = fetch(idx, train=True, lung_mode=spec["lung_mode"], gen=gen)
            y = LABELS_DEV[idx.to(DEVICE)]
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                logits = model(to_input(x01))
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.detach() * len(idx); correct += (logits.argmax(1) == y).sum(); n += len(idx)
        tl, ta = (loss_sum / n).item(), (correct / n).item()
        if not math.isfinite(tl):
            raise RuntimeError(f"non-finite training loss at epoch {ep + 1}")
        vlog = predict(model, val_idx, spec["lung_mode"])
        vl = float(F.cross_entropy(torch.from_numpy(vlog), torch.from_numpy(yv)).item())
        vm = cls_metrics(vlog, yv)
        hist.append(dict(epoch=ep + 1, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=vm["acc"], val_macro_f1=vm["macro_f1"],
                         lr_backbone=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        improved = vl < best - 1e-4
        if improved:
            best, best_ep, bad = vl, ep + 1, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"{spec['run_id']} ep {ep + 1}/{CFG['max_epochs']} train {tl:.4f}/{ta:.3f} val {vl:.4f}/{vm['acc']:.3f} F1 {vm['macro_f1']:.3f} "
            f"({time.time() - t0:.0f}s){' *' if improved else ''}")
        if bad >= CFG["patience"]:
            break
    model.load_state_dict(best_state)
    test_idx = cap_idx(TEST_IDX, CFG["eval_cap"])
    preds = dict(val=predict(model, val_idx, spec["lung_mode"]), test=predict(model, test_idx, spec["lung_mode"]))
    if spec["lung_mode"] is None:
        preds["test_lung_only"] = predict(model, test_idx, "lung_only")
        preds["test_no_lung"] = predict(model, test_idx, "no_lung")
    else:
        preds["test_full_image"] = predict(model, test_idx, None)
        preds["test_no_lung"] = predict(model, test_idx, "no_lung")
    np.savez_compressed(rdir / "logits.npz", val_idx=val_idx, test_idx=test_idx, **preds)
    pd.DataFrame(hist).to_csv(rdir / "history.csv", index=False)
    to_drive = spec["seed"] == 0 or SAVE_ALL_CHECKPOINTS_TO_DRIVE
    ckpt = (DIRS["checkpoints"] if to_drive else WORK / "ckpt") / f"{spec['run_id']}.pt"
    tmp = WORK / "ckpt" / f"_tmp_{spec['run_id']}.pt"
    torch.save({"state_dict": best_state, "spec": spec}, tmp)
    shutil.move(str(tmp), str(ckpt))
    tm = cls_metrics(preds["test"], LAB_NP[test_idx])
    res = dict(spec, status="done", best_epoch=best_ep, epochs_run=len(hist), best_val_loss=best,
               val=cls_metrics(preds["val"], yv), test=tm, train_counts=counts.tolist(), class_weights=cw.tolist(),
               sampled_epoch1=sampled.tolist(), seconds=time.time() - t_start, ckpt=str(ckpt), ckpt_on_drive=bool(to_drive),
               ckpt_bytes=os.path.getsize(ckpt), steps_per_epoch=steps, n_params=sum(p.numel() for p in model.parameters()))
    write_json(done, res)
    log(f"DONE {spec['run_id']}: best epoch {best_ep}, test acc {tm['acc']:.4f}, macro-F1 {tm['macro_f1']:.4f}, AUC {tm['macro_auc']:.4f} ({res['seconds'] / 60:.1f} min)")
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return res


RESULTS, FAILED = {}, {}
SELFTEST_RID = next((s["run_id"] for s in RUNS if s["head"] == "flat" and s["group"] == "main"), None)
t_all = time.time()
for i, spec in enumerate(RUNS):
    try:
        RESULTS[spec["run_id"]] = train_one(spec)
    except Exception as e:
        FAILED[spec["run_id"]] = repr(e)
        log(f"FAILED {spec['run_id']}: {e!r}\n{traceback.format_exc()}")
        gc.collect(); torch.cuda.empty_cache()
    if i == 0 and spec["run_id"] in RESULTS:
        log(f"First run took {RESULTS[spec['run_id']]['seconds'] / 60:.1f} min")
    if spec["run_id"] == SELFTEST_RID and SELFTEST_RID in RESULTS:
        xai_selftest(RESULTS[SELFTEST_RID])          # deliberately NOT caught: a failure here stops the notebook early
log(f"Training stage finished: {len(RESULTS)} done, {len(FAILED)} failed, {(time.time() - t_all) / 3600:.2f} h")
write_json(DIRS["logs"] / "failed_runs.json", FAILED)

# %% [markdown]
# ## 7. Classification results: seeds, 95 % bootstrap CIs, paired tests, calibration, shortcut tests

# %%
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.edgecolor": "#898781", "axes.linewidth": 0.6, "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 110, "savefig.bbox": "tight"})
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
HEAD_COL = {"default": PAL[0], "flat": PAL[1], "flat_trunc": PAL[2], "flat_lamprou": PAL[3]}
INK2 = "#52514e"


def savefig(fig, name):
    fig.savefig(DIRS["figures"] / f"{name}.png", dpi=300)
    fig.savefig(DIRS["figures"] / f"{name}.pdf")
    plt.close(fig)


def safe(fn):
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            log(f"[analysis] {fn.__name__} failed: {e!r}\n{traceback.format_exc()}")
            return None
    wrapper.__name__ = fn.__name__
    return wrapper


def load_logits(rid):
    with np.load(DIRS["runs"] / rid / "logits.npz") as z:
        return {k: z[k] for k in z.files}


def try_logits(rid):
    try:
        return load_logits(rid)
    except Exception as e:
        log(f"[analysis] logits of {rid} unreadable: {e!r} (run skipped)")
        return None


def boot_metrics(y, p, B, seed=0, auc=True):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.randint(0, n, size=(B, n))
    yhat = p.argmax(1)
    corr = (yhat == y)
    acc = corr[idx].mean(1)
    f1s, bal = [], []
    for c in range(3):
        tp = ((yhat == c) & (y == c))[idx].sum(1); fp = ((yhat == c) & (y != c))[idx].sum(1); fn = ((yhat != c) & (y == c))[idx].sum(1)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1s.append(2 * prec * rec / np.maximum(prec + rec, 1e-12)); bal.append(rec)
    out = dict(acc=acc, macro_f1=np.mean(f1s, 0), bal_acc=np.mean(bal, 0))
    if auc:
        aucs = []
        for b in range(min(B, 1000)):
            try:
                aucs.append(roc_auc_score(y[idx[b]], p[idx[b]], multi_class="ovr", average="macro", labels=[0, 1, 2]))
            except Exception:
                pass
        out["macro_auc"] = np.array(aucs)
    return out


def ci_str(v, lo, hi, d=4):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def macro_f1_np(y, yhat):
    f = []
    for c in range(3):
        tp = ((yhat == c) & (y == c)).sum(); fp = ((yhat == c) & (y != c)).sum(); fn = ((yhat != c) & (y == c)).sum()
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f.append(2 * pr * rc / max(pr + rc, 1e-12))
    return float(np.mean(f))


def boot_mean_ci(v, B, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    m = v[rng.randint(0, len(v), size=(B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(np.where(ok, p, np.inf))
    m = ok.sum()
    run = 0.0
    for rank, i in enumerate(idx[:m]):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


def wilcoxon_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), 0
    nz = d[np.abs(d) > 1e-12]
    if len(nz) == 0:
        return 1.0, 0.0, len(d)
    r = stats.rankdata(np.abs(nz))
    rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())
    try:
        p = float(stats.wilcoxon(nz).pvalue)
    except Exception:
        p = float("nan")
    return p, rb, len(d)


done_res = {k: v for k, v in RESULTS.items() if v.get("status") == "done"}


@safe
def classification_tables():
    rows = []
    for rid, r in done_res.items():
        try:
            lg = load_logits(rid)
        except Exception as e:
            log(f"[analysis] logits of {rid} unreadable: {e!r} (run skipped in the tables)")
            continue
        for subset in ("test", "test_dedup"):
            idx = lg["test_idx"]
            logits = lg["test"]
            if subset == "test_dedup":
                keep = ~DF.dup_flag.values[idx]
                idx, logits = idx[keep], logits[keep]
            m = cls_metrics(logits, LAB_NP[idx])
            rows.append(dict(run_id=rid, group=r["group"], backbone=r["backbone"], head=r["head"], seed=r["seed"], balance=r["balance"],
                             imbalance=r["imbalance"], lung_mode=r["lung_mode"] or "full", subset=subset, n=len(idx),
                             best_epoch=r["best_epoch"], epochs_run=r["epochs_run"], minutes=r["seconds"] / 60, **m))
    allm = pd.DataFrame(rows)
    save_table(allm, "T03a_classification_all_runs")
    main = allm[(allm.group == "main")]
    agg = []
    for (bb, hd, sub), g in main.groupby(["backbone", "head", "subset"]):
        d = dict(backbone=BB_LABEL[bb], head=hd, subset=sub, n_seeds=len(g))
        for k in ["acc", "bal_acc", "macro_f1", "macro_auc", "ece", "recall_covid", "recall_noncovid", "recall_normal"]:
            d[f"{k}_mean"] = g[k].mean(); d[f"{k}_sd"] = g[k].std(ddof=1) if len(g) > 1 else float("nan")
        agg.append(d)
    save_table(pd.DataFrame(agg), "T03_classification_mean_sd_over_seeds")
    # bootstrap CIs (seed 0) and default-vs-Flatten paired tests (every seed; Holm correction across backbones within a seed)
    ci_rows, pair_rows = [], []
    B = CFG["boot"]
    for seed in CFG["seeds"]:
        for bb in BACKBONES:
            ps = {}
            for hd in ("default", "flat"):
                rid = make_spec(bb, hd, seed, "main")["run_id"]
                if rid not in done_res:
                    continue
                try:
                    lg = load_logits(rid)
                except Exception as e:
                    log(f"[analysis] logits of {rid} unreadable: {e!r}")
                    continue
                idx, p = lg["test_idx"], softmax_np(lg["test"].astype(np.float64))
                y = LAB_NP[idx]
                if seed == CFG["seeds"][0]:
                    bm = boot_metrics(y, p, B)
                    m = cls_metrics(lg["test"], y)
                    d = dict(backbone=BB_LABEL[bb], head=hd, seed=seed)
                    for k in ["acc", "bal_acc", "macro_f1", "macro_auc"]:
                        lo, hi = np.percentile(bm[k], [2.5, 97.5])
                        d[k] = ci_str(m[k], lo, hi)
                    ci_rows.append(d)
                ps[hd] = (y, p, idx)
            if len(ps) == 2 and np.array_equal(ps["default"][2], ps["flat"][2]):
                y, pd_, _ = ps["default"]
                pf = ps["flat"][1]
                cd, cf = pd_.argmax(1) == y, pf.argmax(1) == y
                b, c = int((cd & ~cf).sum()), int((~cd & cf).sum())
                pval = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c > 0 else 1.0
                rng = np.random.RandomState(1)
                bi = rng.randint(0, len(y), size=(B, len(y)))
                dacc = cf[bi].mean(1) - cd[bi].mean(1)
                f1d = []
                for pp in (pf, pd_):
                    yh = pp.argmax(1); fs = []
                    for k in range(3):
                        tp = ((yh == k) & (y == k))[bi].sum(1); fp = ((yh == k) & (y != k))[bi].sum(1); fn = ((yh != k) & (y == k))[bi].sum(1)
                        pr = tp / np.maximum(tp + fp, 1); rc = tp / np.maximum(tp + fn, 1); fs.append(2 * pr * rc / np.maximum(pr + rc, 1e-12))
                    f1d.append(np.mean(fs, 0))
                df1 = f1d[0] - f1d[1]
                point_df1 = macro_f1_np(y, pf.argmax(1)) - macro_f1_np(y, pd_.argmax(1))
                pair_rows.append(dict(seed=seed, backbone=BB_LABEL[bb], default_correct_flat_wrong=b, default_wrong_flat_correct=c,
                                      mcnemar_exact_p=pval,
                                      delta_acc_flat_minus_default=ci_str(cf.mean() - cd.mean(), *np.percentile(dacc, [2.5, 97.5])),
                                      delta_macroF1_flat_minus_default=ci_str(point_df1, *np.percentile(df1, [2.5, 97.5]))))
    save_table(pd.DataFrame(ci_rows), "T04_classification_seed0_bootstrapCI")
    pair_df = pd.DataFrame(pair_rows)
    if len(pair_df):
        pair_df["mcnemar_p_holm"] = np.nan
        for seed, g in pair_df.groupby("seed"):
            pair_df.loc[g.index, "mcnemar_p_holm"] = holm(g.mcnemar_exact_p.values)
    save_table(pair_df, "T05_default_vs_flatten_paired_tests")
    return allm


ALLM = classification_tables()


@safe
def fig_classification():
    t = TABLES["T03_classification_mean_sd_over_seeds"]
    t = t[t.subset == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False, layout="constrained")
    for ax, k, lab in zip(axes, ["acc", "macro_f1", "macro_auc"], ["Accuracy", "Macro-F1", "Macro-AUC"]):
        y0 = np.arange(len(BACKBONES))
        for j, hd in enumerate(("default", "flat")):
            s = t[t["head"] == hd].set_index("backbone").reindex([BB_LABEL[b] for b in BACKBONES])
            ax.errorbar(s[f"{k}_mean"], y0 + (j - 0.5) * 0.28, xerr=s[f"{k}_sd"].fillna(0), fmt="o", ms=4.5, color=HEAD_COL[hd],
                        elinewidth=1.2, capsize=0, label=HEAD_LABEL[hd])
        ax.set_yticks(y0); ax.set_yticklabels([BB_LABEL[b] for b in BACKBONES] if ax is axes[0] else [])
        ax.set_xlabel(lab); ax.invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper right", ncol=2)
    fig.suptitle("Test performance, mean ± SD over seeds", x=0.01, ha="left", fontsize=9)
    savefig(fig, "F01_classification_by_backbone")


fig_classification()


@safe
def fig_roc_cm():
    rows = []
    for bb in BACKBONES:
        rid = make_spec(bb, "flat", 0, "main")["run_id"]
        if rid in done_res:
            rows.append((done_res[rid]["val"]["macro_f1"], bb))
    if not rows:
        return
    bb = max(rows)[1]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), layout="constrained")
    for j, hd in enumerate(("default", "flat")):
        rid = make_spec(bb, hd, 0, "main")["run_id"]
        if rid not in done_res:
            continue
        lg = load_logits(rid)
        y, p = LAB_NP[lg["test_idx"]], softmax_np(lg["test"].astype(np.float64))
        cm = confusion_matrix(y, p.argmax(1), labels=[0, 1, 2])
        ax = axes[1 + j]
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                frac = cm[a, b] / max(1, cm[a].sum())
                ax.text(b, a, f"{cm[a, b]}", ha="center", va="center", fontsize=7.5, color="white" if frac > 0.5 else "#0b0b0b")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CLS_SHORT, rotation=20); ax.set_yticklabels(CLS_SHORT if j == 0 else [])
        ax.set_xlabel("Predicted"); ax.set_title(f"{BB_LABEL[bb]}, {HEAD_LABEL[hd]}", fontsize=8)
        if hd == "flat":
            for c in range(3):
                fpr, tpr, _ = roc_curve((y == c).astype(int), p[:, c])
                axes[0].plot(fpr, tpr, color=PAL[c], lw=1.5, label=f"{CLS_SHORT[c]} (AUC {roc_auc_score((y == c).astype(int), p[:, c]):.3f})")
    axes[0].plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8)
    axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate"); axes[0].legend(loc="lower right", fontsize=6, handlelength=1.2)
    axes[0].set_title(f"ROC, {BB_LABEL[bb]} Flatten head", fontsize=8)
    axes[1].set_ylabel("True class", labelpad=2)
    savefig(fig, "F02_roc_confusion_best")


fig_roc_cm()


@safe
def fig_learning_curves():
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(8.2, 1.9), sharey=True, layout="constrained")
    for ax, bb in zip(axes, BACKBONES):
        for hd in ("default", "flat"):
            f = DIRS["runs"] / make_spec(bb, hd, 0, "main")["run_id"] / "history.csv"
            if f.exists():
                h = pd.read_csv(f)
                ax.plot(h.epoch, h.train_loss, color=HEAD_COL[hd], lw=1.0, ls=(0, (2, 1.5)))
                ax.plot(h.epoch, h.val_loss, color=HEAD_COL[hd], lw=1.6, label=HEAD_LABEL[hd])
        ax.set_title(BB_LABEL[bb], fontsize=8); ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy"); axes[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Learning curves (solid: validation, dashed: training; seed 0)", x=0.01, ha="left", fontsize=9)
    savefig(fig, "S01_learning_curves")


fig_learning_curves()


@safe
def shortcut_tables():
    rows = []
    for rid, r in done_res.items():
        if r["seed"] != 0 or r["group"] not in ("main", "lung_masked", "head_ablation"):
            continue
        lg = try_logits(rid)
        if lg is None:
            continue
        y = LAB_NP[lg["test_idx"]]
        variants = [("as_trained", "test")] + [(k.replace("test_", ""), k) for k in lg if k.startswith("test_") and k != "test_idx"]
        d = dict(run_id=rid, backbone=BB_LABEL[r["backbone"]], head=r["head"], training_input=r["lung_mode"] or "full image")
        for name, key in variants:
            m = cls_metrics(lg[key], y)
            d[f"acc_{name}"], d[f"macroF1_{name}"] = m["acc"], m["macro_f1"]
        rows.append(d)
    t = save_table(pd.DataFrame(rows), "T06_shortcut_tests")
    tm = t[t.training_input == "full image"]
    tm = tm[tm["head"].isin(["default", "flat"])]
    fig, ax = plt.subplots(figsize=(6.6, 2.6), layout="constrained")
    labels = [f"{b}\n{'Default' if h == 'default' else 'Flatten'}" for b, h in zip(tm.backbone, tm["head"])]
    xx = np.arange(len(tm))
    for j, (col, lab, c) in enumerate([("acc_as_trained", "Full image", PAL[0]), ("acc_lung_only", "Lungs only", PAL[1]), ("acc_no_lung", "Lungs removed", PAL[2])]):
        if col in tm:
            ax.bar(xx + (j - 1) * 0.26, tm[col], width=0.24, color=c, label=lab)
    ax.axhline(1 / 3, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(len(tm) - 0.4, 1 / 3 + 0.01, "chance", fontsize=6.5, color=INK2, ha="right")
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=6.5); ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7)
    savefig(fig, "F03_shortcut_tests")


shortcut_tables()


@safe
def balance_tables():
    rows = []
    for rid, r in done_res.items():
        if r["backbone"] != "resnet50" or r["head"] != "flat" or r["seed"] != 0 or r["lung_mode"]:
            continue
        if r["group"] not in ("main", "balance_natural", "balance_induced"):
            continue
        counts = np.array(r["train_counts"], float); w = np.array(r["class_weights"])
        prior = counts / counts.sum()
        bal = r["balance"]
        eff = {"none": prior, "weights": prior * w, "sampler": np.ones(3), "both": w}[bal]
        eff = eff / eff.sum()
        samp = np.array(r["sampled_epoch1"], float); samp = samp / samp.sum()
        lg = try_logits(rid)
        if lg is None:
            continue
        m = cls_metrics(lg["test"], LAB_NP[lg["test_idx"]])
        rows.append(dict(scenario=r["imbalance"], balancing=bal, train_counts=str(r["train_counts"]),
                         loss_weights_applied=bal in ("weights", "both"), weighted_sampler_applied=bal in ("sampler", "both"),
                         inverse_frequency_weights=str(np.round(w, 4).tolist()), effective_class_emphasis=str(np.round(eff, 3).tolist()),
                         sampled_distribution_epoch1=str(np.round(samp, 3).tolist()), acc=m["acc"], bal_acc=m["bal_acc"],
                         macro_f1=m["macro_f1"], macro_auc=m["macro_auc"], recall_covid=m["recall_covid"], ece=m["ece"]))
    t = pd.DataFrame(rows)
    if len(t):
        t["order"] = t.balancing.map({"none": 0, "weights": 1, "sampler": 2, "both": 3})
        t = t.sort_values(["scenario", "order"]).drop(columns="order")
    save_table(t, "T07_class_imbalance_ablation")


balance_tables()


@safe
def ensemble_classification():
    rows = []
    for hd in ("flat", "default"):
        for s in CFG["seeds"]:
            rids = [make_spec(bb, hd, s, "main")["run_id"] for bb in BACKBONES]
            if not all(r in done_res for r in rids):
                continue
            lgs = [try_logits(r) for r in rids]
            if any(l is None for l in lgs) or not all(np.array_equal(l["test_idx"], lgs[0]["test_idx"]) for l in lgs):
                continue
            y = LAB_NP[lgs[0]["test_idx"]]
            yv = LAB_NP[lgs[0]["val_idx"]]
            vf1 = [cls_metrics(l["val"], yv)["macro_f1"] for l in lgs]
            top3 = list(np.argsort(vf1)[::-1][:3])
            for name, members in (("all5", list(range(5))), ("top3_by_val", top3)):
                ens = np.mean([lgs[i]["test"] for i in members], 0)
                m = cls_metrics(ens, y)
                rows.append(dict(head=hd, seed=s, ensemble=name, members=", ".join(BB_LABEL[BACKBONES[i]] for i in members), **m))
    save_table(pd.DataFrame(rows), "T08_ensemble_classification")


ensemble_classification()

# %% [markdown]
# ## 8. Evaluation set for explanations and the XAI run
# Images: de-duplicated test images correctly classified by all seed-0 main models (the predicted class is explained), stratified
# per class; COVID-19 images are drawn from those with an expert infection mask. Separate images are reserved for each
# qualitative figure, so no figure repeats another.

# %%
XAI = DIRS["xai"]
MAIN0 = {(bb, hd): make_spec(bb, hd, 0, "main")["run_id"] for bb in BACKBONES for hd in ("default", "flat")}


def select_xai_set():
    f = XAI / "xai_set.csv"
    if f.exists():
        return pd.read_csv(f)
    pool = TEST_DEDUP_IDX.copy()
    ok = np.ones(len(pool), bool)
    for rid in MAIN0.values():
        if rid in done_res:
            lg = try_logits(rid)
            if lg is None:
                continue
            pos = {r: i for i, r in enumerate(lg["test_idx"])}
            pred = lg["test"].argmax(1)
            ok &= np.array([pos.get(r, -1) >= 0 and pred[pos[r]] == LAB_NP[r] for r in pool])
    rng = np.random.RandomState(2026)
    rows = []
    n_c, n_ms, n_ex = CFG["n_xai_per_class"], CFG["n_ms_per_class"], CFG["n_examples"]
    for c in range(3):
        cand = pool[ok & (LAB_NP[pool] == c)]
        if c == 0 and DF.has_inf.values[cand].sum() >= n_c:
            cand = cand[DF.has_inf.values[cand]]
        if len(cand) < n_c:
            log(f"WARNING: only {len(cand)} correctly classified candidates for class {c}; filling from the de-duplicated test set")
            extra = rng.permutation(pool[(LAB_NP[pool] == c) & ~np.isin(pool, cand)])
            if c == 0:
                extra = np.concatenate([extra[DF.has_inf.values[extra]], extra[~DF.has_inf.values[extra]]])
            cand = np.concatenate([cand, extra[: n_c - len(cand)]])
        pick = rng.permutation(cand)[:n_c]
        ex_pool = np.setdiff1d(cand, pick)
        if len(ex_pool) >= 3 * n_ex:
            exs = rng.permutation(ex_pool)[: 3 * n_ex]
        else:
            rest = np.setdiff1d(pool[LAB_NP[pool] == c], np.concatenate([pick, ex_pool]))
            exs = np.concatenate([ex_pool, rng.permutation(rest)])[: 3 * n_ex]
        for j, r in enumerate(pick):
            rows.append(dict(row=int(r), label=c, role="eval", maxsens=j < n_ms))
        for j, r in enumerate(exs):
            rows.append(dict(row=int(r), label=c, role=["ex_methods", "ex_backbones", "ex_heads"][j // n_ex], maxsens=False))
    t = pd.DataFrame(rows)
    t["fname"] = DF.fname.values[t.row.values]
    t["has_inf"] = DF.has_inf.values[t.row.values]
    t.to_csv(f, index=False)
    return t


XSET = select_xai_set()
EVAL_ROWS = XSET[XSET.role == "eval"].row.values
MS_ROWS = XSET[(XSET.role == "eval") & XSET.maxsens].row.values
EX_ROWS = XSET[XSET.role != "eval"].row.values
log(f"XAI set: {len(EVAL_ROWS)} evaluation images ({int(XSET[XSET.role == 'eval'].has_inf.sum())} with infection masks), "
    f"{len(MS_ROWS)} for Max-Sensitivity/randomization, {len(EX_ROWS)} reserved for figures")
RISE_MASKS = make_rise_masks(CFG["rise_masks"], CFG["rise_s"], CFG["rise_p"])


def load_member(rid):
    r = done_res[rid]
    ck = torch.load(r["ckpt"], map_location="cpu", weights_only=True)
    m, info = build_model(r["backbone"], r["head"])
    m.load_state_dict(ck["state_dict"])
    return m.to(DEVICE).eval().to(memory_format=torch.channels_last), info


def xai_entries():
    E = []
    for hd in ("flat", "default"):
        for bb in BACKBONES:
            rid = MAIN0[(bb, hd)]
            E.append(dict(name=rid, runs=[rid], tier="A", methods=ALL_METHODS, lung_mode=None))
    flat_ok = [MAIN0[(bb, "flat")] for bb in BACKBONES if MAIN0[(bb, "flat")] in done_res]
    if len(flat_ok) == len(BACKBONES):
        vf1 = [done_res[r]["val"]["macro_f1"] for r in flat_ok]
        top3 = [flat_ok[i] for i in np.argsort(vf1)[::-1][:3]]
        ens_m = ["gradcam", "hirescam", "ig", "rise", "random"]
        E.append(dict(name="ENSEMBLE_flat_top3", runs=top3, tier="E", methods=ens_m, lung_mode=None))
        E.append(dict(name="ENSEMBLE_flat_all5", runs=flat_ok, tier="E", methods=ens_m, lung_mode=None))
    for rid, r in done_res.items():
        if r["seed"] == 0 and (r["group"] == "head_ablation" or r["group"] == "lung_masked"):
            E.append(dict(name=rid, runs=[rid], tier="B", methods=["gradcam", "hirescam", "random"], lung_mode=r["lung_mode"]))
    return [e for e in E if all(r in done_res for r in e["runs"])]


def run_xai(entry):
    fcsv = XAI / f"{entry['name']}__per_image.csv"
    if fcsv.exists():
        return
    t0 = time.time()
    members, infos = zip(*[load_member(r) for r in entry["runs"]])
    ex = Explainable(list(members), entry["name"], entry["lung_mode"])
    methods = list(entry["methods"])
    bb0 = done_res[entry["runs"][0]]["backbone"]
    ctx = dict(rise_masks=RISE_MASKS, ig_batch=(16 if LOCAL_TEST else (32 if bb0 == "vgg19" or ex.is_ensemble else 64)))
    gen = torch.Generator(device=DEVICE); gen.manual_seed(77)
    rows, curves, examples = [], {}, {}
    all_rows = np.concatenate([EVAL_ROWS, EX_ROWS])
    bs = CFG["xai_batch"]
    for s in range(0, len(all_rows), bs):
        rr = all_rows[s:s + bs]
        is_eval = np.isin(rr, EVAL_ROWS)
        x01, lung = fetch(rr, train=False, lung_mode=entry["lung_mode"], with_lung=True)
        x = to_input(x01)
        with torch.no_grad():
            logits = ex.fwd(x).float()
        target = logits.argmax(1)
        y = LABELS_DEV[torch.as_tensor(rr).to(DEVICE)]
        vis, raw, info = ex.explain(x, target, methods, rr, ctx)
        for k in vis:
            for j, r in enumerate(rr):
                if not is_eval[j]:
                    examples.setdefault(k, {})[int(r)] = vis[k][j].detach().cpu().numpy().astype(np.float16)
        if not is_eval.any():
            continue
        sel = torch.as_tensor(np.where(is_eval)[0], device=DEVICE)
        rr_e = rr[is_eval]
        score_t = logits.gather(1, target.view(-1, 1)).squeeze(1)[sel]
        delta = (info["score"] - info["s0"])[sel] if "score" in info else None
        ig_sum = info["ig_sum"][sel] if "ig_sum" in info else None
        x, x01, lung, target, y = x[sel], x01[sel], lung[sel], target[sel], y[sel]
        vis = {k: v[sel] for k, v in vis.items()}
        raw = {k: v[sel] for k, v in raw.items()}
        inf = fetch_inf(rr_e)
        has_inf = torch.as_tensor(DF.has_inf.values[rr_e], device=DEVICE)
        gn = torch.Generator(device=DEVICE); gn.manual_seed(int(rr_e[0]))
        noise = to_input(torch.rand(len(rr_e), 1, INPUT, INPUT, device=DEVICE, generator=gn))
        blurred = gaussian_blur(x)
        correct_orig = (target == y).float()
        if ig_sum is not None:
            with torch.no_grad():
                f0 = ex.fwd(BASELINE.expand_as(x).contiguous(memory_format=torch.channels_last)).float().gather(1, target.view(-1, 1)).squeeze(1)
            d_ig = score_t - f0
        for k in methods:
            if k not in vis:
                continue
            v = vis[k]
            rec = {f"aopc_g{gr}": None for gr in CFG["aopc_grids"]}
            for gr in CFG["aopc_grids"]:
                c = aopc_curves(ex.fwd, x, v, target, gr, noise, gen)
                rec[f"aopc_g{gr}"] = (c[:, :1] - c).mean(1)
                curves.setdefault((k, f"aopc_g{gr}"), []).append(c.cpu().numpy())
            dc, ic = delins_curves(ex.fwd, x, v, target, blurred, gen)
            curves.setdefault((k, "deletion"), []).append(dc.cpu().numpy())
            curves.setdefault((k, "insertion"), []).append(ic.cpu().numpy())
            ha = ((x01 * 2 - 1) * (1 + v.unsqueeze(1))).clamp(-1, 1)
            with torch.no_grad():
                ha_pred = ex.fwd(to_input((ha + 1) / 2)).float().argmax(1)
            pl = plausibility(v, lung, inf, has_inf)
            comp = torch.full((len(rr_e),), float("nan"), device=DEVICE)
            if k in raw and delta is not None and k != "scorecam":
                comp = (raw[k].flatten(1).sum(1) - delta).abs() / (delta.abs() + 1e-8)
            if k == "ig" and ig_sum is not None:
                comp = (ig_sum - d_ig).abs() / (d_ig.abs() + 1e-8)
            zero_map = (v.flatten(1).amax(1) <= 0).float()
            for j, r in enumerate(rr_e):
                d = dict(model=entry["name"], tier=entry["tier"], method=k, row=int(r), label=int(y[j]), target=int(target[j]),
                         correct=float(correct_orig[j]), deletion_auc=float(norm_auc(dc[j:j + 1])[0]), insertion_auc=float(norm_auc(ic[j:j + 1])[0]),
                         haas_correct=float(ha_pred[j] == y[j]), completeness_relerr=float(comp[j]), zero_map=float(zero_map[j]))
                for gr in CFG["aopc_grids"]:
                    d[f"aopc_g{gr}"] = float(rec[f"aopc_g{gr}"][j])
                for kk, vv in pl.items():
                    d[kk] = float(vv[j])
                if delta is not None:
                    d["logit_minus_ref"] = float(delta[j])
                    if k in raw and k != "scorecam":
                        d["raw_map_sum"] = float(raw[k][j].sum())
                rows.append(d)
    df_rows = pd.DataFrame(rows)
    # Max-Sensitivity (relative, Quantus default normalisation; absolute, as Lamprou et al.)
    ms_methods = [m for m in methods if m not in ("scorecam", "rise", "random")]
    ms_rows = []
    if len(MS_ROWS) and ms_methods:
        gms = torch.Generator(device=DEVICE); gms.manual_seed(99)
        for s in range(0, len(MS_ROWS), bs):
            rr = MS_ROWS[s:s + bs]
            x01, _ = fetch(rr, train=False, lung_mode=entry["lung_mode"])
            with torch.no_grad():
                target = ex.fwd(to_input(x01)).float().argmax(1)
            base, _, _ = ex.explain(to_input(x01), target, ms_methods, rr, ctx, with_raw=False)
            for rad in CFG["ms_radii"]:
                best_abs = {m: torch.zeros(len(rr), device=DEVICE) for m in ms_methods}
                best_rel = {m: torch.zeros(len(rr), device=DEVICE) for m in ms_methods}
                for _ in range(CFG["ms_samples"]):
                    xp = (x01 + (torch.rand(x01.shape, device=DEVICE, generator=gms) * 2 - 1) * rad).clamp(0, 1)
                    vp, _, _ = ex.explain(to_input(xp), target, ms_methods, rr, ctx, with_raw=False)
                    for m in ms_methods:
                        dif = (vp[m] - base[m]).flatten(1).norm(dim=1)
                        best_abs[m] = torch.maximum(best_abs[m], dif)
                        best_rel[m] = torch.maximum(best_rel[m], dif / (base[m].flatten(1).norm(dim=1) + 1e-12))
                for m in ms_methods:
                    for j, r in enumerate(rr):
                        ms_rows.append(dict(model=entry["name"], method=m, row=int(r), radius=rad, maxsens_rel=float(best_rel[m][j]), maxsens_abs=float(best_abs[m][j])))
    # Model-parameter randomization (Adebayo et al. 2018): head, then head + last stage
    rnd_rows = []
    if entry["tier"] == "A" and len(MS_ROWS):
        info0 = infos[0]
        rexs = []
        for level, paths in (("head", ["head"]), ("head+last_stage", ["head"] + info0["last_stage"])):
            rm = copy.deepcopy(members[0])
            for k_, p_ in enumerate(paths):
                reinit_(rm.get_submodule(p_), 4321 + k_)
            rexs.append((level, Explainable([rm.eval()], "rand_" + level)))
        rmeth = [m for m in methods if m not in ("random", "scorecam", "rise")]     # gradient-based methods (cost)
        for s in range(0, len(MS_ROWS), bs):
            rr = MS_ROWS[s:s + bs]
            x01, _ = fetch(rr, train=False, lung_mode=entry["lung_mode"])
            x = to_input(x01)
            with torch.no_grad():
                target = ex.fwd(x).float().argmax(1)
            b0, _, _ = ex.explain(x, target, rmeth, rr, ctx, with_raw=False)
            for level, rex in rexs:
                b1, _, _ = rex.explain(x, target, rmeth, rr, ctx, with_raw=False)
                for m in rmeth:
                    for j, rho in enumerate(spearman_rows(b0[m], b1[m])):
                        rnd_rows.append(dict(model=entry["name"], method=m, level=level, row=int(rr[j]), spearman=rho))
        del rexs
    pd.DataFrame(ms_rows).to_csv(XAI / f"{entry['name']}__maxsens.csv", index=False)
    pd.DataFrame(rnd_rows).to_csv(XAI / f"{entry['name']}__randomization.csv", index=False)
    np.savez_compressed(XAI / f"{entry['name']}__curves.npz", **{f"{k}|{m}": np.concatenate(v).mean(0) for (k, m), v in curves.items()})
    np.savez_compressed(XAI / f"{entry['name']}__examples.npz", **{f"{k}|{r}": a for k, d in examples.items() for r, a in d.items()})
    df_rows.to_csv(fcsv, index=False)
    write_json(XAI / f"{entry['name']}__meta.json", dict(entry, seconds=time.time() - t0))
    log(f"XAI done for {entry['name']} ({(time.time() - t0) / 60:.1f} min)")
    del members, ex
    gc.collect(); torch.cuda.empty_cache()


XAI_FAILED = {}
for e in xai_entries():
    try:
        run_xai(e)
    except Exception as err:
        XAI_FAILED[e["name"]] = repr(err)
        log(f"XAI FAILED {e['name']}: {err!r}\n{traceback.format_exc()}")
        gc.collect(); torch.cuda.empty_cache()
write_json(DIRS["logs"] / "failed_xai.json", XAI_FAILED)

# %% [markdown]
# ## 9. Explanation results: tables with 95 % CIs, paired Wilcoxon tests (Holm), figures

# %%
def model_label(name):
    if name.startswith("ENSEMBLE"):
        return "Ensemble (" + name.split("_", 2)[-1] + ")"
    p = name.split("__")
    lab = f"{BB_LABEL.get(p[0], p[0])} · {HEAD_LABEL.get(p[1], p[1])}"
    if len(p) > 5 and p[5] != "full":
        lab += " · lung-masked"
    return lab


XAI_TABLES = {}


@safe
def xai_tables():
    files = sorted(XAI.glob("*__per_image.csv"))
    if not files:
        log("No XAI results found"); return
    D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    MS = pd.concat([pd.read_csv(f) for f in XAI.glob("*__maxsens.csv") if f.stat().st_size > 5], ignore_index=True) if any(f.stat().st_size > 5 for f in XAI.glob("*__maxsens.csv")) else pd.DataFrame()
    RN = pd.concat([pd.read_csv(f) for f in XAI.glob("*__randomization.csv") if f.stat().st_size > 5], ignore_index=True) if any(f.stat().st_size > 5 for f in XAI.glob("*__randomization.csv")) else pd.DataFrame()
    XAI_TABLES.update(D=D, MS=MS, RN=RN)
    B = CFG["boot"]
    metrics = [f"aopc_g{g}" for g in CFG["aopc_grids"]] + ["deletion_auc", "insertion_auc", "lung_mass", "inf_mass", "inf_rank_acc", "pointing", "pointing_tol"]
    rows = []
    for (mdl, meth), g in D.groupby(["model", "method"]):
        d = dict(model=model_label(mdl), model_id=mdl, method=METHOD_LABEL[meth], n_images=len(g))
        for k in metrics:
            m, lo, hi = boot_mean_ci(g[k], B)
            d[k] = m; d[f"{k}_lo"] = lo; d[f"{k}_hi"] = hi; d[f"{k}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        d["haas"] = g.haas_correct.mean() / max(g.correct.mean(), 1e-12)
        d["completeness_relerr_median"] = g.completeness_relerr.median()
        d["completeness_relerr_max"] = g.completeness_relerr.max()
        d["zero_maps"] = int(g.zero_map.sum())
        d["lung_area_chance"] = g.lung_area.mean()
        if len(MS):
            for rad, gm in MS[(MS.model == mdl) & (MS.method == meth)].groupby("radius"):
                m, lo, hi = boot_mean_ci(gm.maxsens_rel, B)
                d[f"maxsens_rel_r{rad}"] = m; d[f"maxsens_rel_r{rad}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
                d[f"maxsens_abs_r{rad}"] = gm.maxsens_abs.mean()
        if len(RN):
            for lev, gr in RN[(RN.model == mdl) & (RN.method == meth)].groupby("level"):
                d[f"randomization_spearman_{lev}"] = gr.spearman.mean()
        rows.append(d)
    S = pd.DataFrame(rows)
    save_table(S, "T09_xai_summary_all")
    # paired tests: HiResCAM vs every other method, per model and metric; Holm within each (model, metric)
    trows = []
    test_metrics = [f"aopc_g{g}" for g in CFG["aopc_grids"]] + ["deletion_auc", "insertion_auc", "lung_mass", "inf_mass", "pointing_tol"]
    for mdl, g in D.groupby("model"):
        piv = {k: g.pivot_table(index="row", columns="method", values=k) for k in test_metrics}
        for k in test_metrics:
            P = piv[k]
            if "hirescam" not in P:
                continue
            block = []
            for m in P.columns:
                if m == "hirescam":
                    continue
                both = P[["hirescam", m]].dropna()
                p, rb, n = wilcoxon_paired(both["hirescam"], both[m])
                block.append(dict(model=model_label(mdl), metric=k, comparison=f"HiResCAM vs {METHOD_LABEL[m]}", n=n,
                                  mean_hirescam=both["hirescam"].mean(), mean_other=both[m].mean(), p=p, rank_biserial=rb))
            ph = holm([b["p"] for b in block])
            for b, q in zip(block, ph):
                b["p_holm"] = q
            trows += block
        if len(MS) and "hirescam" in MS[MS.model == mdl].method.unique():
            for rad, gm in MS[MS.model == mdl].groupby("radius"):
                P = gm.pivot_table(index="row", columns="method", values="maxsens_rel")
                block = []
                for m in P.columns:
                    if m == "hirescam":
                        continue
                    both = P[["hirescam", m]].dropna()
                    p, rb, n = wilcoxon_paired(both["hirescam"], both[m])
                    block.append(dict(model=model_label(mdl), metric=f"maxsens_rel_r{rad}", comparison=f"HiResCAM vs {METHOD_LABEL[m]}", n=n,
                                      mean_hirescam=both["hirescam"].mean(), mean_other=both[m].mean(), p=p, rank_biserial=rb))
                for b, q in zip(block, holm([b["p"] for b in block])):
                    b["p_holm"] = q
                trows += block
    T10 = pd.DataFrame(trows)
    if len(T10):
        T10["better_is"] = np.where(T10.metric.str.startswith("deletion") | T10.metric.str.startswith("maxsens"), "lower", "higher")
        T10["hirescam_better"] = np.where(T10.better_is == "lower", T10.mean_hirescam < T10.mean_other, T10.mean_hirescam > T10.mean_other)
        T10["significant_holm_0.05"] = T10.p_holm < 0.05
    save_table(T10, "T10_xai_paired_tests")
    # compact reviewer-facing tables
    point = lambda c: not c.endswith(("_ci", "_lo", "_hi"))
    keep = ["model", "method", "n_images"] + [c for c in S.columns if c.startswith("aopc_g") and point(c)] + \
           ["deletion_auc", "insertion_auc", "haas"] + [c for c in S.columns if c.startswith("maxsens_rel_r") and point(c)] + \
           ["completeness_relerr_median"]
    save_table(S[[c for c in keep if c in S.columns]], "T11_xai_faithfulness_robustness")
    keep2 = ["model", "method", "lung_mass", "lung_mass_ci", "lung_area_chance", "inf_mass", "inf_mass_ci", "inf_rank_acc", "pointing", "pointing_tol", "pointing_tol_ci"]
    save_table(S[[c for c in keep2 if c in S.columns]], "T12_xai_plausibility")
    keep3 = ["model", "method"] + [c for c in S.columns if c.startswith("randomization_spearman")]
    save_table(S[[c for c in keep3 if c in S.columns]], "T13_randomization_sanity")
    # head ablation summary (Grad-CAM vs HiResCAM across head variants) joined with classification
    abl = S[S.method.isin(["Grad-CAM", "HiResCAM"])].copy()
    abl["backbone"] = abl.model_id.str.split("__").str[0]
    abl["head"] = abl.model_id.str.split("__").str[1]
    abl = abl[~abl.model_id.str.startswith("ENSEMBLE") & ~abl.model_id.str.contains("lung_only")]
    if ALLM is not None:
        cl = ALLM[(ALLM.subset == "test") & (ALLM.seed == 0) & (ALLM.balance == "none") & (ALLM.imbalance == "natural") & (ALLM.lung_mode == "full")]
        abl = abl.merge(cl[["run_id", "acc", "macro_f1", "macro_auc"]], left_on="model_id", right_on="run_id", how="left")
    cols = ["backbone", "head", "method", "acc", "macro_f1", "macro_auc", "completeness_relerr_median"] + \
           [c for c in abl.columns if (c.startswith("aopc_g") or c.startswith("maxsens_rel_r")) and point(c)] + ["deletion_auc", "insertion_auc", "haas", "lung_mass"]
    abl = abl[[c for c in cols if c in abl.columns]].sort_values(["backbone", "head", "method"])
    save_table(abl, "T14_head_ablation_xai")
    return S


SUMMARY = xai_tables()


@safe
def fig_completeness():
    D = XAI_TABLES.get("D")
    if D is None or "raw_map_sum" not in D:
        return
    sub = D[D.method.isin(["gradcam", "hirescam"]) & D.tier.isin(["A"])].dropna(subset=["raw_map_sum"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), layout="constrained")
    pick = MAIN0.get(("resnet50", "flat"))
    s = sub[sub.model == pick]
    for meth, c in (("gradcam", PAL[0]), ("hirescam", PAL[1])):
        q = s[s.method == meth]
        axes[0].scatter(q.logit_minus_ref, q.raw_map_sum, s=9, color=c, alpha=0.75, lw=0, label=METHOD_LABEL[meth])
    if len(s):
        lim = [s.logit_minus_ref.min(), s.logit_minus_ref.max()]
        axes[0].plot(lim, lim, color=INK2, lw=0.8)
    axes[0].set_xlabel("Class logit − head(0)"); axes[0].set_ylabel("Sum of raw explanation map"); axes[0].legend(fontsize=7)
    axes[0].set_title("ResNet-50, Flatten head", fontsize=8)
    names = [MAIN0[(bb, hd)] for bb in BACKBONES for hd in ("default", "flat")]
    ypos, labels = [], []
    for i, nm in enumerate(names):
        for meth, c, off in (("gradcam", PAL[0], -0.17), ("hirescam", PAL[1], 0.17)):
            q = sub[(sub.model == nm) & (sub.method == meth)].completeness_relerr
            if len(q):
                axes[1].scatter(np.clip(q.median(), 1e-9, None), i + off, color=c, s=14)
        ypos.append(i); labels.append(model_label(nm).replace("Default head", "Default").replace("Flatten head", "Flatten"))
    axes[1].set_xscale("log"); axes[1].set_yticks(ypos); axes[1].set_yticklabels(labels, fontsize=6.5); axes[1].invert_yaxis()
    axes[1].set_xlabel("Median relative completeness error (log)")

    savefig(fig, "F04_completeness")


fig_completeness()


@safe
def fig_method_dots(metric, xlabel, fname, chance=None):
    S = SUMMARY
    if S is None or metric not in S:
        return
    s = S[S.model_id.isin([MAIN0[(bb, hd)] for bb in BACKBONES for hd in ("default", "flat")])]
    meths = [METHOD_LABEL[m] for m in ALL_METHODS]
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(8.6, 2.9), sharey=True, layout="constrained")
    for ax, bb in zip(axes, BACKBONES):
        for j, hd in enumerate(("default", "flat")):
            q = s[s.model_id == MAIN0[(bb, hd)]].set_index("method").reindex(meths)
            v = q[metric].to_numpy(float)
            err = None
            if f"{metric}_lo" in q:
                err = np.nan_to_num(np.clip(np.vstack([v - q[f"{metric}_lo"].to_numpy(float), q[f"{metric}_hi"].to_numpy(float) - v]), 0, None))
            yy = np.arange(len(meths)) + (j - 0.5) * 0.3
            ok = np.isfinite(v)
            if ok.any():
                ax.errorbar(v[ok], yy[ok], xerr=None if err is None else err[:, ok], fmt="o", ms=3.8, color=HEAD_COL[hd],
                            elinewidth=1, capsize=0, label=HEAD_LABEL[hd])
            if chance is not None and chance in q and np.isfinite(q[chance].to_numpy(float)).any():
                ax.axvline(float(np.nanmean(q[chance].to_numpy(float))), color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.set_title(BB_LABEL[bb], fontsize=8)
    fig.supxlabel(xlabel, fontsize=8)
    axes[0].set_yticks(np.arange(len(meths))); axes[0].set_yticklabels(meths); axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper right", ncol=2, fontsize=7)
    savefig(fig, fname)


fig_method_dots(f"aopc_g{max(CFG['aopc_grids'])}", f"AOPC, {max(CFG['aopc_grids'])}×{max(CFG['aopc_grids'])} regions (higher = better)", "F05_aopc_methods")
fig_method_dots("deletion_auc", "Deletion AUC (lower = better)", "F06a_deletion_methods")
fig_method_dots("insertion_auc", "Insertion AUC (higher = better)", "F06b_insertion_methods")
fig_method_dots("lung_mass", "Attribution mass inside lungs (dashed: lung area)", "F08a_lung_mass", chance="lung_area_chance")
fig_method_dots("pointing_tol", "Pointing game on infection masks", "F08b_pointing_game")
fig_method_dots("inf_mass", "Attribution mass inside infection", "F08c_infection_mass")


@safe
def fig_morf_curves():
    pick = MAIN0.get(("resnet50", "flat"))
    f = XAI / f"{pick}__curves.npz"
    if not f.exists():
        return
    z = np.load(f)
    gr = max(CFG["aopc_grids"])
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), layout="constrained")
    for ax, key, title in zip(axes, [f"aopc_g{gr}", "deletion", "insertion"], [f"MoRF, {gr}×{gr} regions", "Deletion", "Insertion"]):
        for m in ALL_METHODS:
            k = f"{m}|{key}"
            if k not in z.files:
                continue
            c = z[k]
            xx = np.linspace(0, 1, len(c))
            hl = m in ("hirescam", "gradcam", "random")
            col = {"hirescam": PAL[1], "gradcam": PAL[0], "random": "#898781"}.get(m, "#c3c2b7")
            ax.plot(xx, c, color=col, lw=1.8 if hl else 0.9, label=METHOD_LABEL[m] if hl else None, zorder=3 if hl else 1)
        ax.set_title(title, fontsize=8); ax.set_xlabel("Fraction perturbed")
    axes[0].set_ylabel("Probability of explained class"); axes[0].legend(fontsize=6.5)
    fig.suptitle("ResNet-50 · Flatten head (grey: other methods)", x=0.01, ha="left", fontsize=8.5)
    savefig(fig, "F07_perturbation_curves")


fig_morf_curves()


@safe
def fig_maxsens():
    MS = XAI_TABLES.get("MS")
    if MS is None or not len(MS):
        return
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(8.6, 2.2), sharey=True, layout="constrained")
    show = {"hirescam": PAL[1], "gradcam": PAL[0], "ig": PAL[2], "layercam": PAL[3]}
    for ax, bb in zip(axes, BACKBONES):
        for hd, ls in (("flat", "-"), ("default", (0, (2, 1.5)))):
            q = MS[MS.model == MAIN0[(bb, hd)]]
            for m, c in show.items():
                g = q[q.method == m].groupby("radius").maxsens_rel.mean()
                if len(g):
                    ax.plot(g.index, g.values, color=c, ls=ls, lw=1.4, marker="o", ms=3, label=f"{METHOD_LABEL[m]}" if (hd == "flat" and bb == BACKBONES[0]) else None)
        ax.set_title(BB_LABEL[bb], fontsize=8); ax.set_xlabel("Noise radius r")
    axes[0].set_ylabel("Max-Sensitivity (relative)"); axes[0].legend(fontsize=6.5)
    fig.suptitle("Max-Sensitivity (solid: Flatten head, dashed: default head; lower = more robust)", x=0.01, ha="left", fontsize=8.5)
    savefig(fig, "F09_max_sensitivity")


fig_maxsens()


@safe
def fig_randomization():
    S = SUMMARY
    if S is None:
        return
    cols = [c for c in S.columns if c.startswith("randomization_spearman")]
    if not cols:
        return
    s = S[S.model_id.isin([MAIN0[(bb, "flat")] for bb in BACKBONES])]
    meths = [METHOD_LABEL[m] for m in ALL_METHODS if METHOD_LABEL[m] in set(s.dropna(subset=cols, how="all").method)]
    if not meths:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(6.8, 2.6), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        for i, bb in enumerate(BACKBONES):
            q = s[s.model_id == MAIN0[(bb, "flat")]].set_index("method").reindex(meths)
            ax.scatter(q[c], np.arange(len(meths)) + (i - 2) * 0.12, s=10, color=PAL[i], label=BB_LABEL[bb])
        ax.axvline(0, color="#c3c2b7", lw=0.8)
        ax.set_title("Randomized: " + c.replace("randomization_spearman_", ""), fontsize=8); ax.set_xlabel("Spearman ρ with original map")
    axes[0].set_yticks(range(len(meths))); axes[0].set_yticklabels(meths); axes[0].invert_yaxis()
    axes[-1].legend(fontsize=6.5, loc="lower right")
    savefig(fig, "F10_randomization_sanity")


fig_randomization()


@safe
def fig_qualitative():
    def overlay(ax, row, vis, title=None, inf=False):
        img = IMG[row]
        x01, _ = fetch(np.array([row]), train=False)
        ax.imshow(x01[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        if vis is not None:
            ax.imshow(vis.astype(np.float32), cmap="inferno", alpha=0.5, vmin=0, vmax=1)
        if inf and DF.has_inf.values[row]:
            m = fetch_inf(np.array([row]))[0, 0].cpu().numpy()
            ax.contour(m, levels=[0.5], colors=["#1baf7a"], linewidths=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if title:
            ax.set_title(title, fontsize=7)

    def ex_of(name):
        f = XAI / f"{name}__examples.npz"
        return np.load(f) if f.exists() else None

    xs = XSET
    # (1) methods, one model
    pick = MAIN0.get(("resnet50", "flat"))
    z = ex_of(pick)
    if z is not None:
        rows = xs[xs.role == "ex_methods"].groupby("label").head(1).row.values
        meths = [m for m in ALL_METHODS if any(k.startswith(m + "|") for k in z.files)]
        fig, axes = plt.subplots(len(rows), len(meths) + 1, figsize=(1.05 * (len(meths) + 1), 1.15 * len(rows)), layout="constrained")
        axes = np.atleast_2d(axes)
        for i, r in enumerate(rows):
            overlay(axes[i, 0], r, None, CLASSES[LAB_NP[r]] if i >= 0 else None, inf=True)
            for j, m in enumerate(meths):
                k = f"{m}|{r}"
                overlay(axes[i, j + 1], r, z[k] if k in z.files else None, METHOD_LABEL[m].replace("Integrated Gradients", "Integr. Grad.") if i == 0 else None, inf=True)
        savefig(fig, "F11_qualitative_methods_resnet50_flat")
    # (2) HiResCAM across backbones (Flatten head) — different images
    rows = xs[xs.role == "ex_backbones"].groupby("label").head(1).row.values
    fig, axes = plt.subplots(len(rows), len(BACKBONES) + 1, figsize=(1.15 * (len(BACKBONES) + 1), 1.2 * len(rows)), layout="constrained")
    axes = np.atleast_2d(axes)
    for j, bb in enumerate(BACKBONES):
        z = ex_of(MAIN0[(bb, "flat")])
        for i, r in enumerate(rows):
            if j == 0:
                overlay(axes[i, 0], r, None, CLASSES[LAB_NP[r]], inf=True)
            k = f"hirescam|{r}"
            overlay(axes[i, j + 1], r, z[k] if (z is not None and k in z.files) else None, BB_LABEL[bb] if i == 0 else None, inf=True)
    savefig(fig, "F12_qualitative_hirescam_backbones")
    # (3) Grad-CAM vs HiResCAM, default vs Flatten head — different images again
    rows = xs[xs.role == "ex_heads"].groupby("label").head(1).row.values
    cols = [(hd, m) for hd in ("default", "flat") for m in ("gradcam", "hirescam")]
    fig, axes = plt.subplots(len(rows), len(cols) + 1, figsize=(1.2 * (len(cols) + 1), 1.2 * len(rows)), layout="constrained")
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        overlay(axes[i, 0], r, None, CLASSES[LAB_NP[r]], inf=True)
    for j, (hd, m) in enumerate(cols):
        z = ex_of(MAIN0[("resnet50", hd)])
        for i, r in enumerate(rows):
            k = f"{m}|{r}"
            overlay(axes[i, j + 1], r, z[k] if (z is not None and k in z.files) else None,
                    f"{'Default' if hd == 'default' else 'Flatten'}: {METHOD_LABEL[m]}" if i == 0 else None, inf=True)
    savefig(fig, "F13_qualitative_heads_resnet50")


fig_qualitative()

# %% [markdown]
# ## 10. Final summary, Excel workbook and zip archive (all on Google Drive)

# %%
@safe
def final_summary():
    lines = [f"# Results summary ({RUN_MODE}) — {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
             f"GPU: {GPU_NAME} · torch {torch.__version__} · output folder: `{OUT}`", "",
             f"Training runs completed: {len(done_res)} / {len(RUNS)}; failed: {len(FAILED)}; XAI failures: {len(XAI_FAILED)}", ""]
    if FAILED or XAI_FAILED:
        lines += ["## Failures", "", "```", json.dumps(dict(training=FAILED, xai=XAI_FAILED), indent=1), "```", ""]
    for name in ["T01_dataset", "T01b_duplicate_audit", "T03_classification_mean_sd_over_seeds", "T05_default_vs_flatten_paired_tests",
                 "T06_shortcut_tests", "T07_class_imbalance_ablation", "T08_ensemble_classification", "T11_xai_faithfulness_robustness",
                 "T12_xai_plausibility", "T13_randomization_sanity", "T14_head_ablation_xai"]:
        if name in TABLES:
            lines += [f"## {name}", "", TABLES[name].to_csv(index=False), ""]
    (OUT / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    with pd.ExcelWriter(OUT / "ALL_TABLES.xlsx") as xw:
        used = set()
        for name, t in TABLES.items():
            sh = name[:31]
            while sh in used:
                sh = sh[:29] + str(len(used))
            used.add(sh)
            t.to_excel(xw, sheet_name=sh, index=False)
    zpath = OUT / f"results_{RUN_MODE}_tables_figures.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("tables", "figures", "audit", "logs"):
            for p in DIRS[sub].rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(OUT))
        for p in (OUT / "RESULTS_SUMMARY.md", OUT / "ALL_TABLES.xlsx"):
            if p.exists():
                z.write(p, p.relative_to(OUT))
        for p in XAI.glob("*.csv"):
            z.write(p, p.relative_to(OUT))
        for p in DIRS["runs"].rglob("*"):
            if p.is_file() and p.suffix in (".json", ".csv"):
                z.write(p, p.relative_to(OUT))
    log(f"ALL DONE. Summary: {OUT / 'RESULTS_SUMMARY.md'} | Excel: {OUT / 'ALL_TABLES.xlsx'} | Zip: {zpath}")


final_summary()
if DRIVE_OK:
    try:
        from google.colab import drive
        drive.flush_and_unmount()
        log("Google Drive flushed.")
    except Exception as e:
        log("drive flush skipped:", repr(e))
