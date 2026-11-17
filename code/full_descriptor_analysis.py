"""
Full descriptor analysis for the photochemical [2+2] cycloaddition dataset.

What it does
  1. Cleans ML_merged_cleaned.xlsx (decodes Excel-corrupted dr/er ratios, parses
     wavelength/temperature/yield, harmonizes solvents, assigns catalyst classes).
  2. Parses every SMILES with RDKit and reports parse failures (data-quality check).
  3. Builds four nested feature sets:
       A  conditions only
       B  A + RDKit 2D descriptors (Substrate 1, Substrate 2, product)
       C  B + Morgan fingerprints (radius 2) + reaction difference fingerprint
       D  C + Mordred descriptors (optional, --mordred)
  4. Yield regression: mean baseline, Ridge, Random Forest, XGBoost,
     evaluated by repeated random 5-fold CV AND study-grouped 5-fold CV
     (records from one primary study never split across train and test).
  5. Selectivity classification (major-isomer fraction >= 0.90):
     repeated stratified 5-fold CV AND leave-one-source-out CV.
  6. SHAP analysis of the XGBoost yield model on the best feature set.

Setup (laptop or Google Colab):
  pip install rdkit mordredcommunity xgboost shap scikit-learn pandas numpy scipy matplotlib openpyxl

Run:
  python full_descriptor_analysis.py --data ML_merged_cleaned.xlsx --out results
  python full_descriptor_analysis.py --data ML_merged_cleaned.xlsx --out results --mordred   # adds set D (slower)
  python full_descriptor_analysis.py --quick   # fewer repeats, for a fast test

Send back the whole results/ folder; every number and figure for the paper is in it.
"""
import argparse, json, os, re, sys, time, warnings, datetime
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score, balanced_accuracy_score, matthews_corrcoef)

try:
    from rdkit import Chem, RDLogger, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False
try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

SEED = 0


def signed_log(X):
    """Compress heavy-tailed descriptors (e.g. Ipc, BertzCT) so linear models do not extrapolate wildly."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


SLOG = FunctionTransformer(signed_log, validate=False)
SPECIES = {"s1": "Substrate_1_SMILES", "s2": "Substrate_2_SMILES", "p": "Product_SMILES"}


# ----------------------------------------------------------------------------- cleaning
def _ratio_from_timedelta(v):
    """Excel stored 'a:b' ratios as durations of a hours b minutes. Recover (a, b)."""
    if isinstance(v, (datetime.timedelta, pd.Timedelta)):
        minutes = int(round(v.total_seconds() / 60))
        return minutes // 60, minutes % 60
    if isinstance(v, datetime.time):
        return v.hour, v.minute
    if isinstance(v, str):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", v)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.match(r"(\d+)\s*days?,\s*(\d+):(\d+):\d+", v)
        if m:
            days, h, mi = map(int, m.groups())
            return days * 24 + h, mi
    return np.nan, np.nan


def _wavelength(s):
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(s))]
    if not nums:
        return np.nan
    return float(np.mean(nums)) if len(nums) > 1 else nums[0]


def _temperature(x):
    if str(x).strip().lower() in ("r.t.", "rt", "r.t", "room temperature"):
        return 25.0
    return pd.to_numeric(x, errors="coerce")


SOLVENT_MAP = {"CC#N": "MeCN", "N#CC": "MeCN", "ClCCl": "DCM", "CCOCC": "Et2O",
               "CC1=CC=CC=C1": "toluene", "Cc1ccccc1": "toluene", "C1=CC=CC=C1": "benzene",
               "c1ccccc1": "benzene", "CC(C)=O": "acetone", "CCCCCC": "hexane", "C1CCCCC1": "hexane"}


def _catalyst_class(s):
    if pd.isna(s):
        return "none"
    if "Ru" in s:
        return "Ru"
    if "Cu" in s:
        return "Cu"
    if "Al" in s or ("B" in s and "N+" in s):
        return "LewisAcid"
    if "[O+]" in s or "o+" in s:
        return "pyrylium"
    return "other_org"


def _find(fname, near):
    for base in (os.path.dirname(os.path.abspath(near)), '.', 'data', os.path.join('..', 'data')):
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{fname} not found; it ships in the data/ folder of the deposit")


def clean(path):
    d = pd.read_excel(path)
    # documented corrections of Excel drag-fill artifacts (see data/corrections.csv)
    C = pd.read_csv(_find('corrections.csv', path), dtype=str)
    for _, c in C.iterrows():
        mask = d.Reaction_ID == int(c.Reaction_ID)
        if c.Action == 'corrected':
            d.loc[mask, c.Field] = c.Corrected
        elif c.Action == 'set to missing':
            d.loc[mask, c.Field] = np.nan
    # traced primary-study provenance: Study_ID is the grouping unit for validation
    P = pd.read_csv(_find('provenance.csv', path))
    d = d.merge(P[['Reaction_ID', 'Study_ID', 'Review_scheme', 'Primary_DOIs', 'Confidence']], on='Reaction_ID', how='left')
    for col, tag in [("Diastereomeric_ratio", "dr"), ("Enantiomeric_ratio", "er")]:
        ab = d[col].map(_ratio_from_timedelta)
        d[tag + "_a"] = [x[0] for x in ab]
        d[tag + "_b"] = [x[1] for x in ab]
        d[tag + "_major"] = d[tag + "_a"] / (d[tag + "_a"] + d[tag + "_b"])
    d["yield"] = pd.to_numeric(d["Yield"], errors="coerce")
    d.loc[d["yield"] > 1.0, "yield"] = d.loc[d["yield"] > 1.0, "yield"] / 100.0
    d["wl"] = d["Light_nm"].map(_wavelength)
    d["temp"] = d["Temp_°C"].map(_temperature)
    d["intra"] = d["Substrate_2_SMILES"].isna().astype(int)
    d["cat"] = d["Catalyst_SMILES"].map(_catalyst_class)
    d["solv"] = d["Solvent"].map(lambda s: "unk" if pd.isna(s) else SOLVENT_MAP.get(s, "other"))
    d["cat_mol"] = pd.to_numeric(d["Cat.(mol%)"], errors="coerce")
    d["cat_mol_missing"] = d["cat_mol"].isna().astype(int)
    d["has_add"] = d["Reagent_Second"].notna().astype(int)
    return d


# ----------------------------------------------------------------------------- features
def condition_features(d):
    X = d[["wl", "temp", "cat_mol", "cat_mol_missing", "has_add", "intra"]].copy()
    X = pd.concat([X, pd.get_dummies(d[["cat", "solv"]], prefix=["cat", "solv"]).astype(int)], axis=1)
    return X


def to_mol(smi):
    """Parse with full sanitization; fall back to partial sanitization for organometallics."""
    if not isinstance(smi, str) or not smi.strip():
        return None, "missing"
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return m, "ok"
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None, "unparseable"
    try:
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                         Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                         catchErrors=True)
        return m, "partial"
    except Exception:
        return None, "sanitize_failed"


def largest_fragment(m):
    """Drop counter-ions (e.g. triflate, BF4-) so descriptors describe the reacting species."""
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda f: f.GetNumHeavyAtoms()) if frags else m


def rdkit_features(d, n_bits=1024):
    names = [n for n, _ in Descriptors.descList]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    desc_blocks, fp_blocks, parse_log = [], [], []
    count_fps = {}
    for tag, col in SPECIES.items():
        rows, fps, cfps = [], [], []
        for rid, smi in zip(d["Reaction_ID"], d[col]):
            m, status = to_mol(smi)
            parse_log.append({"Reaction_ID": rid, "species": tag, "status": status})
            if m is None:
                rows.append({n: np.nan for n in names})
                fps.append(np.zeros(n_bits, dtype=np.uint8))
                cfps.append(np.zeros(n_bits))
                continue
            m = largest_fragment(m)
            vals = {}
            for n, f in Descriptors.descList:
                try:
                    vals[n] = f(m)
                except Exception:
                    vals[n] = np.nan
            rows.append(vals)
            arr = np.zeros(n_bits, dtype=np.uint8)
            try:
                DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
                carr = gen.GetCountFingerprintAsNumPy(m).astype(float)
            except Exception:
                carr = np.zeros(n_bits)
            fps.append(arr)
            cfps.append(carr)
        desc_blocks.append(pd.DataFrame(rows).add_prefix(f"{tag}_"))
        if tag != "p":
            fp_blocks.append(pd.DataFrame(np.vstack(fps), columns=[f"{tag}_fp{i}" for i in range(n_bits)]))
        count_fps[tag] = np.vstack(cfps)
    # reaction difference fingerprint: bits created/destroyed going from reactants to product
    diff = count_fps["p"] - count_fps["s1"] - count_fps["s2"]
    fp_blocks.append(pd.DataFrame(diff, columns=[f"rxn_fp{i}" for i in range(n_bits)]))
    desc = pd.concat(desc_blocks, axis=1)
    desc = desc.replace([np.inf, -np.inf], np.nan)
    fp = pd.concat(fp_blocks, axis=1)
    return desc, fp, pd.DataFrame(parse_log)


def mordred_features(d):
    from mordred import Calculator, descriptors as mdesc
    calc = Calculator(mdesc, ignore_3D=True)
    blocks = []
    for tag, col in SPECIES.items():
        mols, usable = [], []
        for smi in d[col]:
            m, status = to_mol(smi)
            ok = False
            if m is not None and status == "ok":
                frag = largest_fragment(m)
                try:
                    Chem.SanitizeMol(frag)          # Mordred needs fully valid molecules
                    Chem.MolToSmiles(Chem.RemoveHs(frag))
                    ok = True
                except Exception:
                    ok = False
            mols.append(frag if ok else Chem.MolFromSmiles("C"))
            usable.append(ok)
        df = calc.pandas(mols, nproc=1, quiet=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.loc[~np.array(usable), :] = np.nan   # missing, unparseable or partially sanitized species
        print(f"  Mordred {tag}: {sum(usable)} of {len(usable)} molecules usable")
        blocks.append(df.add_prefix(f"{tag}_mrd_"))
    return pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan)


def prune(X, max_nan=0.5, corr=0.95, var_eps=1e-8):
    """Remove mostly-missing, constant and highly correlated columns (unsupervised: uses no target)."""
    X = X.loc[:, X.isna().mean() <= max_nan]
    X = X.loc[:, X.std(numeric_only=True).fillna(0) > var_eps]
    binary = [c for c in X.columns if set(pd.unique(X[c].dropna())) <= {0, 1}]
    cont = [c for c in X.columns if c not in binary]
    if len(cont) > 1:
        cm = X[cont].corr().abs()
        upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if (upper[c] > corr).any()]
        X = X.drop(columns=drop)
    # binary fingerprint bits: keep bits set in at least 3 and at most n-3 reactions
    for c in [c for c in X.columns if c in binary]:
        s = X[c].sum()
        if s < 3 or s > len(X) - 3:
            X = X.drop(columns=c)
    return X


# ----------------------------------------------------------------------------- validation
def grouped_folds(groups, k=5, seed=0):
    """Randomized greedy assignment of whole sources to k folds with balanced sizes."""
    rng = np.random.RandomState(seed)
    sizes = pd.Series(groups).value_counts()
    order = list(sizes.index)
    big, rest = order[:k], order[k:]
    rng.shuffle(big)
    rng.shuffle(rest)
    folds, load = [[] for _ in range(k)], [0] * k
    for g in big + rest:
        i = int(np.argmin(load))
        folds[i].append(g)
        load[i] += sizes[g]
    return [np.where(np.isin(groups, f))[0] for f in folds]


def regression_models(n_features):
    m = {
        "Mean baseline": DummyRegressor(),
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), SLOG, StandardScaler(),
                               RidgeCV(alphas=np.logspace(-1, 4, 30))),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                                             max_features=0.3, random_state=SEED, n_jobs=-1)),
    }
    if HAVE_XGB:
        m["XGBoost"] = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                        colsample_bytree=0.5, min_child_weight=3, reg_lambda=2.0,
                                        random_state=SEED, n_jobs=-1, verbosity=0)
    else:
        m["HistGradientBoosting (XGBoost unavailable)"] = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=SEED)
    return m


def eval_regression(model, X, y, groups, scheme, repeats):
    out = []
    oof_last = None
    for r in range(repeats):
        if scheme == "random":
            splits = [te for _, te in KFold(5, shuffle=True, random_state=r).split(X)]
        else:
            splits = grouped_folds(groups, 5, seed=r)
        pred = np.zeros(len(y))
        for te in splits:
            tr = np.setdiff1d(np.arange(len(y)), te)
            model.fit(X.iloc[tr], y[tr])
            pred[te] = model.predict(X.iloc[te])
        out.append([r2_score(y, pred), mean_absolute_error(y, pred) * 100,
                    mean_squared_error(y, pred) ** 0.5 * 100])
        oof_last = pred
    a = np.array(out)
    return {"R2_mean": a[:, 0].mean(), "R2_sd": a[:, 0].std(), "MAE_pct": a[:, 1].mean(),
            "RMSE_pct": a[:, 2].mean()}, oof_last


def eval_classification(model, X, y, groups, repeats):
    res = {}
    aucs, bacs, mccs = [], [], []
    for r in range(repeats):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            model.fit(X.iloc[tr], y[tr])
            p[te] = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
        yh = (p >= 0.5).astype(int)
        bacs.append(balanced_accuracy_score(y, yh))
        mccs.append(matthews_corrcoef(y, yh))
    res["random"] = {"AUC_mean": np.mean(aucs), "AUC_sd": np.std(aucs),
                     "BalAcc": np.mean(bacs), "MCC": np.mean(mccs)}
    p = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(set(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        model.fit(X.iloc[tr], y[tr])
        p[te] = model.predict_proba(X.iloc[te])[:, 1]
    yh = (p >= 0.5).astype(int)
    rng = np.random.RandomState(SEED)
    boot = []
    for _ in range(2000):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i])) == 2:
            boot.append(roc_auc_score(y[i], p[i]))
    res["LOSO"] = {"AUC_mean": roc_auc_score(y, p), "AUC_CI_low": np.percentile(boot, 2.5),
                   "AUC_CI_high": np.percentile(boot, 97.5),
                   "BalAcc": balanced_accuracy_score(y, yh), "MCC": matthews_corrcoef(y, yh)}
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ML_merged_cleaned.xlsx")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mordred", action="store_true", help="also compute Mordred descriptors (feature set D)")
    ap.add_argument("--quick", action="store_true", help="2 repeats instead of 5 (smoke test)")
    ap.add_argument("--allow-no-rdkit", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    repeats = 2 if a.quick else 5
    t0 = time.time()
    if not HAVE_RDKIT and not a.allow_no_rdkit:
        sys.exit("RDKit is not installed. Run: pip install rdkit mordredcommunity xgboost shap")

    env = {"python": sys.version.split()[0], "rdkit": None, "xgboost": None, "shap": None, "mordred": None}
    if HAVE_RDKIT:
        import rdkit; env["rdkit"] = getattr(rdkit, "__version__", "unknown")
    if HAVE_XGB: env["xgboost"] = getattr(xgb, "__version__", "unknown")
    if HAVE_SHAP: env["shap"] = getattr(shap, "__version__", "unknown")
    import sklearn; env["sklearn"] = sklearn.__version__

    d = clean(a.data)
    d.to_csv(os.path.join(a.out, "cleaned_records.csv"), index=False)
    cond = condition_features(d)

    feature_sets = {"A: conditions": cond}
    if HAVE_RDKIT:
        desc, fp, parse_log = rdkit_features(d)
        parse_log.to_csv(os.path.join(a.out, "smiles_parse_log.csv"), index=False)
        summary = parse_log.groupby(["species", "status"]).size().unstack(fill_value=0)
        summary.to_csv(os.path.join(a.out, "smiles_parse_summary.csv"))
        print("SMILES parsing:\n", summary)
        feature_sets["B: + RDKit descriptors"] = pd.concat([cond, desc], axis=1)
        feature_sets["C: + Morgan fingerprints"] = pd.concat([cond, desc, fp], axis=1)
        if a.mordred:
            try:
                import mordred
                try:
                    from importlib.metadata import version as _v
                    env["mordred"] = _v("mordredcommunity")
                except Exception:
                    env["mordred"] = getattr(mordred, "__version__", "unknown")
                mrd = mordred_features(d)
                feature_sets["D: + Mordred"] = pd.concat([cond, desc, fp, mrd], axis=1)
            except Exception as e:
                import traceback
                print("Mordred skipped:", repr(e))
                traceback.print_exc()
                open(os.path.join(a.out, "mordred_error.txt"), "w").write(traceback.format_exc())
    json.dump(env, open(os.path.join(a.out, "environment.json"), "w"), indent=1)

    # ---------------- yield regression
    ymask = d["yield"].notna().values
    y = d.loc[ymask, "yield"].values
    g = d.loc[ymask, "Study_ID"].values
    rows, oof_store, pruned = [], {}, {}
    for fs_name, Xfull in feature_sets.items():
        X = prune(Xfull.loc[ymask].reset_index(drop=True))
        pruned[fs_name] = X
        print(f"\n[{fs_name}] {X.shape[1]} features after pruning")
        for m_name, model in regression_models(X.shape[1]).items():
            for scheme in ["random", "grouped"]:
                met, oof = eval_regression(model, X, y, g, scheme, repeats)
                rows.append({"feature_set": fs_name, "model": m_name, "validation": scheme,
                             "n_features": X.shape[1], **met})
                oof_store[(fs_name, m_name, scheme)] = oof
                print(f"  {m_name:28s} {scheme:8s} R2={met['R2_mean']:.3f}±{met['R2_sd']:.3f}  MAE={met['MAE_pct']:.1f}")
    reg = pd.DataFrame(rows)
    reg.to_csv(os.path.join(a.out, "yield_regression_results.csv"), index=False)

    # best non-baseline model under GROUPED validation decides the reported model
    cand = reg[(reg.validation == "grouped") & (reg.model != "Mean baseline")]
    best = cand.sort_values("R2_mean", ascending=False).iloc[0]
    best_key = (best.feature_set, best.model, "grouped")
    np.savetxt(os.path.join(a.out, "best_grouped_oof_predictions.csv"),
               np.c_[y, oof_store[best_key]], delimiter=",", header="observed,predicted", comments="")

    # ---------------- figures
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    models_order = [m for m in reg.model.unique() if m != "Mean baseline"]
    fs_order = list(feature_sets)
    w = 0.8 / len(models_order)
    for i, m in enumerate(models_order):
        for j, scheme in enumerate(["random", "grouped"]):
            sub = reg[(reg.model == m) & (reg.validation == scheme)].set_index("feature_set").loc[fs_order]
            xs = np.arange(len(fs_order)) + i * w - 0.4 + w / 2
            ax.bar(xs if scheme == "random" else xs, sub.R2_mean.values, w * 0.9,
                   yerr=sub.R2_sd.values, capsize=2,
                   color=plt.cm.tab10(i), alpha=1.0 if scheme == "random" else 0.45,
                   label=f"{m} ({scheme})", zorder=2 if scheme == "grouped" else 1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(fs_order)))
    ax.set_xticklabels(fs_order, fontsize=8)
    ax.set_ylabel("Cross-validated R²")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.set_title("Yield: random (solid) vs study-grouped (faded) validation", fontsize=9, loc="left")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig_validation_gap.png"), dpi=300); plt.close()

    o = oof_store[best_key]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(y * 100, o * 100, s=12, alpha=0.6)
    ax.plot([0, 100], [0, 100], "k--", lw=0.8)
    ax.set_xlabel("Reported yield (%)"); ax.set_ylabel("Predicted yield (%)")
    ax.set_title(f"{best.model}, {best.feature_set}\nstudy-grouped CV, R² = {r2_score(y, o):.2f}", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig_parity_grouped.png"), dpi=300); plt.close()

    # ---------------- SHAP on gradient-boosted model, richest feature set
    shap_info = {}
    if HAVE_SHAP and HAVE_XGB:
        fs_shap = fs_order[-1] if len(fs_order) > 1 else fs_order[0]
        Xs = pruned[fs_shap]
        model = regression_models(Xs.shape[1])["XGBoost"].fit(Xs, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
        imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
        imp.to_csv(os.path.join(a.out, "shap_mean_abs.csv"), header=["mean_abs_shap"])
        shap.summary_plot(sv, Xs, max_display=15, show=False, plot_size=(7, 5))
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig_shap_summary.png"), dpi=300); plt.close()
        shap_info = {"feature_set": fs_shap, "top15": imp.head(15).round(5).to_dict()}
        print("\nTop SHAP features:\n", imp.head(15))

    # ---------------- selectivity classification
    smask = d["dr_major"].notna().values
    ys = (d.loc[smask, "dr_major"] >= 0.90).astype(int).values
    gs = d.loc[smask, "Study_ID"].values
    clf_rows = []
    for fs_name, Xfull in feature_sets.items():
        X = prune(Xfull.loc[smask].reset_index(drop=True))
        clfs = {
            "Logistic regression": make_pipeline(SimpleImputer(strategy="median"), SLOG, StandardScaler(),
                                                 LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)),
            "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                           RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                                  class_weight="balanced", random_state=SEED, n_jobs=-1)),
        }
        if HAVE_XGB:
            clfs["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                colsample_bytree=0.5, random_state=SEED, n_jobs=-1, verbosity=0,
                                                scale_pos_weight=(ys == 0).sum() / max((ys == 1).sum(), 1))
        for c_name, c in clfs.items():
            r = eval_classification(c, X, ys, gs, max(repeats * 2, 4))
            for scheme, met in r.items():
                clf_rows.append({"feature_set": fs_name, "model": c_name, "validation": scheme,
                                 "n_features": X.shape[1], **met})
                print(f"  sel {fs_name:25s} {c_name:20s} {scheme:6s} AUC={met['AUC_mean']:.3f}")
    pd.DataFrame(clf_rows).to_csv(os.path.join(a.out, "selectivity_classification_results.csv"), index=False)

    summary = {"n_records": int(len(d)), "n_yield": int(ymask.sum()), "n_selectivity": int(smask.sum()),
               "n_selectivity_positive": int(ys.sum()), "repeats": repeats,
               "best_grouped_model": {"feature_set": best.feature_set, "model": best.model,
                                      "R2": float(best.R2_mean), "MAE_pct": float(best.MAE_pct)},
               "shap": shap_info, "environment": env, "runtime_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(a.out, "results_summary.json"), "w"), indent=1, default=float)
    print(f"\nDone in {summary['runtime_s']} s. Results in {a.out}/")


if __name__ == "__main__":
    main()
