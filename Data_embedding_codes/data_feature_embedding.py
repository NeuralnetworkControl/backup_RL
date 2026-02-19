"""
Run this script in the folder that contains vcbench_final_public.csv.
It will:
 - detect founder_uuid (or founder_id) column,
 - build per-category embeddings (edu/role/exec/industry/orgscale/text),
 - save .npy matrices and a founder_index.csv that maps founder_uuid -> embedding_index.
"""

import json
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys

CSV_PATH = "merged.csv"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-d
OUTPUT_PREFIX = "founder"            # prefix for saved files

# column names in your CSV (change if needed)
EDU_COL = "educations_json"
JOBS_COL = "jobs_json"
TEXT_COL = "anonymised_prose"
IND_COL = "industry"
IPOS_COL = "ipos"
ACQ_COL = "acquisitions"

# ---------- helper functions ----------
def safe_load_json(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, dict)):
        return cell
    s = str(cell).strip()
    if s == "":
        return []
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return []

def safe_text(x):
    return "" if pd.isna(x) else str(x)

def duration_to_years(d):
    if d is None: return 0.0
    s = str(d).strip()
    if s == "": return 0.0
    if re.match(r"^<\s*\d+", s):
        m = re.search(r"\d+", s)
        return float(m.group()) * 0.75 if m else 1.0
    if re.match(r"^\d+\s*-\s*\d+", s):
        a,b = re.findall(r"\d+", s)[:2]
        return (float(a) + float(b)) / 2.0
    if re.match(r"^\d+\+$", s):
        m = re.search(r"\d+", s)
        return float(m.group()) + 1.0
    try:
        return float(s)
    except:
        return 0.0

def pool_mean(embs, emb_dim):
    if len(embs) == 0:
        return np.zeros(emb_dim)
    return np.mean(np.vstack(embs), axis=0)

# ---------- load CSV and detect id column ----------
df = pd.read_csv(CSV_PATH, low_memory=False)
N = len(df)
print(f"Loaded {CSV_PATH}, rows={N}")

# detect founder id column
FOUND_COL_CANDIDATES = ["founder_uuid", "founder_id", "uuid", "id"]
found_id_col = None
for c in FOUND_COL_CANDIDATES:
    if c in df.columns:
        found_id_col = c
        break

if found_id_col is None:
    # fallback: create stable uuid from row index (but better to have a real id)
    print("Warning: no founder_uuid/founder_id found. Creating synthetic founder_uuid from row index.")
    df["founder_uuid"] = df.index.astype(str)
    found_id_col = "founder_uuid"

# check duplicates
dups = df[found_id_col].duplicated().sum()
if dups > 0:
    print(f"Error: found {dups} duplicate founder ids in column {found_id_col}. Please ensure unique ids.")
    sys.exit(1)

# ---------- prepare model ----------
model = SentenceTransformer(EMB_MODEL_NAME)
emb_dim = model.get_sentence_embedding_dimension()
print("Using model:", EMB_MODEL_NAME, "embedding dim:", emb_dim)

# init storage
edu_vecs = np.zeros((N, emb_dim))
role_vecs = np.zeros((N, emb_dim))
exec_vecs = np.zeros((N, emb_dim))
industry_vecs = np.zeros((N, emb_dim))
orgscale_vecs = np.zeros((N, emb_dim))
text_vecs = np.zeros((N, emb_dim))
depth_feats = []   # [total_years, num_jobs, num_long_tenure]
edu_meta = []      # per-row education meta
exit_feats = []    # [has_ipo, num_ipos, num_acq]

EXEC_ROLES = set(["ceo", "cto", "founder", "co-founder", "cfo", "cofounder", "chief", "chair"])
LONG_TENURE_YEARS = 2.0

# ---------- iterate rows ----------
for i, row in tqdm(df.iterrows(), total=N, desc="Processing rows"):
    # education
    eds = safe_load_json(row.get(EDU_COL, "[]"))
    edu_texts = []
    max_degree_level = 0
    qs_rank = None
    for ed in eds:
        if not isinstance(ed, dict):
            edu_texts.append(str(ed))
            continue
        deg = ed.get("degree", "") or ed.get("degree_name", "")
        field = ed.get("field", "") or ed.get("major", "")
        qs = ed.get("qs_ranking") or ed.get("qs-ranking") or ed.get("qs_rank") or ed.get("qsRanking")
        dlow = str(deg).lower()
        if "ph" in dlow or "doctor" in dlow: level = 3
        elif "master" in dlow or "msc" in dlow: level = 2
        elif "b" in dlow or "ba" in dlow or "bs" in dlow: level = 1
        else: level = 0
        max_degree_level = max(max_degree_level, level)
        if qs is not None and qs != "":
            try:
                qs_rank = int(qs)
            except:
                pass
        text = " ".join([str(x) for x in [deg, field, ("QS"+str(qs) if qs else "")] if x])
        edu_texts.append(text)
    if edu_texts:
        edu_embs = model.encode(edu_texts, show_progress_bar=False)
        edu_vecs[i] = pool_mean(edu_embs, emb_dim)
    else:
        edu_vecs[i] = np.zeros(emb_dim)
    edu_meta.append({"max_degree_level": max_degree_level, "qs_rank": qs_rank})

    # jobs
    jobs = safe_load_json(row.get(JOBS_COL, "[]"))
    role_texts = []
    exec_texts = []
    total_years = 0.0
    num_jobs = 0
    num_long = 0
    industries_seen = []
    company_size_texts = []
    for j in jobs:
        if not isinstance(j, dict):
            role_texts.append(str(j))
            continue
        role = j.get("role", "") or j.get("title", "")
        comp_size = j.get("company_size", "")
        industry = j.get("industry", "")
        duration = j.get("duration", "")
        r_txt = " ".join([str(x) for x in [role, industry, comp_size] if x])
        role_texts.append(r_txt)
        industries_seen.append(str(industry))
        if comp_size:
            company_size_texts.append(str(comp_size))
        yrs = duration_to_years(duration)
        total_years += yrs
        num_jobs += 1
        if yrs >= LONG_TENURE_YEARS:
            num_long += 1
        lowrole = str(role).lower()
        if any(tok in lowrole for tok in EXEC_ROLES):
            exec_texts.append(r_txt)

    # role vec
    if role_texts:
        role_embs = model.encode(role_texts, show_progress_bar=False)
        role_vecs[i] = pool_mean(role_embs, emb_dim)
    else:
        role_vecs[i] = np.zeros(emb_dim)

    # exec vec
    if exec_texts:
        exec_embs = model.encode(exec_texts, show_progress_bar=False)
        exec_vecs[i] = pool_mean(exec_embs, emb_dim)
    else:
        exec_vecs[i] = np.zeros(emb_dim)

    # industry vec
    inds = []
    if isinstance(row.get(IND_COL, ""), str) and row.get(IND_COL, "").strip():
        inds.append(row.get(IND_COL))
    inds.extend([x for x in industries_seen if x])
    if inds:
        ind_embs = model.encode(list(set(inds)), show_progress_bar=False)
        industry_vecs[i] = pool_mean(ind_embs, emb_dim)
    else:
        industry_vecs[i] = np.zeros(emb_dim)

    # org scale
    if company_size_texts:
        cs_embs = model.encode(list(set(company_size_texts)), show_progress_bar=False)
        orgscale_vecs[i] = pool_mean(cs_embs, emb_dim) if len(cs_embs)>0 else np.zeros(emb_dim)
    else:
        orgscale_vecs[i] = np.zeros(emb_dim)

    depth_feats.append([total_years, num_jobs, num_long])

    # exit features
    num_ipos = 0
    num_acq = 0
    ipos_cell = row.get(IPOS_COL)
    acq_cell = row.get(ACQ_COL)
    try:
        ipos_list = safe_load_json(ipos_cell)
        if isinstance(ipos_list, list):
            num_ipos = len(ipos_list)
        elif isinstance(ipos_list, dict):
            num_ipos = len(ipos_list.keys())
        else:
            num_ipos = int(ipos_list) if str(ipos_list).isdigit() else 0
    except:
        num_ipos = 0
    try:
        acq_list = safe_load_json(acq_cell)
        if isinstance(acq_list, list):
            num_acq = len(acq_list)
        elif isinstance(acq_list, dict):
            num_acq = len(acq_list.keys())
        else:
            num_acq = int(acq_list) if str(acq_list).isdigit() else 0
    except:
        num_acq = 0
    has_ipo = 1 if num_ipos > 0 else 0
    exit_feats.append([has_ipo, num_ipos, num_acq])

    # anonymised prose
    prose = safe_text(row.get(TEXT_COL, ""))
    text_vecs[i] = model.encode([prose], show_progress_bar=False)[0] if prose.strip() else np.zeros(emb_dim)

# ---------- finalize numeric features ----------
depth_arr = np.array(depth_feats)
exit_arr = np.array(exit_feats)

scaler = StandardScaler()
if N > 0:
    depth_arr_scaled = scaler.fit_transform(depth_arr)
else:
    depth_arr_scaled = depth_arr

# ---------- save embeddings and metadata ----------
# SLOTS = {
#   "edu":        "founder_edu_state.npy",
#   "role":       "founder_role_vecs.npy",
#   "exec":       "founder_exec_vecs.npy",
#   "industry":   "founder_industry_vecs.npy",
#   "orgscale":   "founder_orgscale_vecs.npy",
#   "depth":      "founder_depth_feats.npy",
#   "exit":       "founder_exit_feats.npy",
#   "summary":    "founder_text_vecs.npy"
# }

np.save(f"{OUTPUT_PREFIX}_edu_vecs.npy", edu_vecs)
np.save(f"{OUTPUT_PREFIX}_role_vecs.npy", role_vecs)
np.save(f"{OUTPUT_PREFIX}_exec_vecs.npy", exec_vecs)
np.save(f"{OUTPUT_PREFIX}_industry_vecs.npy", industry_vecs)
np.save(f"{OUTPUT_PREFIX}_orgscale_vecs.npy", orgscale_vecs)
np.save(f"{OUTPUT_PREFIX}_text_vecs.npy", text_vecs)

np.save(f"{OUTPUT_PREFIX}_depth_feats.npy", depth_arr_scaled)
np.save(f"{OUTPUT_PREFIX}_exit_feats.npy", exit_arr)

edu_meta_df = pd.DataFrame(edu_meta)
edu_meta_df.to_csv(f"{OUTPUT_PREFIX}_edu_meta.csv", index=False)

# ---------- save index mapping: embedding_index <-> founder_uuid ----------
index_df = pd.DataFrame({
    "embedding_index": np.arange(N, dtype=int),
    "founder_uuid": df[found_id_col].values
})
index_df.to_csv(f"{OUTPUT_PREFIX}_index.csv", index=False)

print("Saved files with prefix:", OUTPUT_PREFIX)
print(f" - index mapping: {OUTPUT_PREFIX}_index.csv  (columns: embedding_index, founder_uuid)")
print(" - embedding .npy files: edu/role/exec/industry/orgscale/text")
print(" - numeric arrays: depth_feats.npy, exit_feats.npy")
