#!/usr/bin/env python3
"""M3 functionally-grounded evaluation (no human subjects).

Avalia tres propriedades quantitativas da explicacao no hold-out De Vito
(30 FPs), sem qualquer dado de opiniao (substitui o survey human-grounded):

  Metrica 2 - Estabilidade: LIME reexecutado K vezes por FP com sementes
              distintas; Jaccard medio par-a-par dos conjuntos top-k e
              correlacao de Spearman dos pesos. O trace simbolico e
              deterministico (Jaccard 1.0, variancia 0).
  Metrica 3 - Ancoragem semantica: fracao da |massa de atribuicao| que recai
              sobre trechos funcionalmente relevantes (os source_text citados
              pelo trace), para LIME-todos-os-tokens e LIME-sem-stopwords
              (stopwords excluidas de numerador e denominador). O trace e 1.0
              por construcao.

Metrica 1 (fidelidade/correcao do trace) reaproveita results/movement-level/.

Saida: results/m3-quantitative/{per_fp.csv, summary.md}. Semente fixa.
"""
from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
OUT = BASE / "results" / "m3-quantitative"
OUT.mkdir(parents=True, exist_ok=True)

DS = BASE / "replications" / "cosmet-devito" / "datasets"
SYSTEMS = {
    "ALBERGATE": DS / "MIS" / "ALBERGATE",
    "K01719": DS / "IoT_Telemedicine" / "K01719",
    "K01720": DS / "IoT_Telemedicine" / "K01720",
    "K01726": DS / "IoT_Telemedicine" / "K01726",
    "U-CURE": DS / "ML" / "U-CURE",
    "AutomaticLineSwitching": DS / "Real-time" / "AutomaticLineSwitching",
    "RiceCooker": DS / "Real-time" / "RiceCooker",
}

TRAIN_DIR = BASE / "results/hybrid-xai/119fp-E28a-qwen3-8b-clean-train-20260525T130245"
K_RUNS = 8           # reexecucoes do LIME por FP (estabilidade)
TOPK = 8             # tamanho do conjunto top-k para Jaccard
NUM_FEATURES = 25    # features retornadas por execucao do LIME
NUM_SAMPLES = 400    # perturbacoes por execucao (igual aos materiais)
SEED = 42


_BERT: dict = {}


def embed(texts: list[str], max_length: int = 512) -> np.ndarray:
    """Embeddings BERT [CLS], com modelo carregado UMA vez (cacheado).

    Identico ao braco BERT do E30f/materiais (mesma revision pinada), mas sem
    recarregar o modelo a cada chamada (a versao de bert_hybrid_improvements_v2
    recarrega, o que domina o custo sob centenas de chamadas do LIME)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not _BERT:
        name = "google-bert/bert-base-uncased"
        rev = "86b5e0934494bd15c9632b12f734a8a67f723594"
        _BERT["tok"] = AutoTokenizer.from_pretrained(name, revision=rev)
        _BERT["model"] = AutoModel.from_pretrained(name, revision=rev).eval()
    tok, model = _BERT["tok"], _BERT["model"]
    out = []
    bs = 16
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            o = model(**enc)
        out.append(o.last_hidden_state[:, 0, :].numpy())
    return np.vstack(out)


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "ner_ontology_pipeline",
        str(BASE / "models" / "hybrid-xai" / "ner_srl" / "ner_ontology_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_ontology_pipeline"] = mod
    spec.loader.exec_module(mod)
    ner = BASE / "models" / "cosmic-ner" / "best"
    return mod.NerOntologyPipeline(ner_model_path=str(ner) if ner.exists() else None)


def tokenize(s: str) -> list[str]:
    """Mesma granularidade do LIME (sequencias de caracteres de palavra)."""
    return [t.lower() for t in re.findall(r"[A-Za-z]+", s or "")]


def holdout_ids() -> list[str]:
    ids = []
    with (BASE / "datasets" / "holdout" / "devito_holdout.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ids.append(r["id"])
    return ids


def fp_text(man_id: str) -> str | None:
    m = re.match(r"COSMET-(.+)-(FP\d+)$", man_id)
    if not m:
        return None
    system, fpname = m.group(1), m.group(2)
    root = SYSTEMS.get(system)
    if root is None:
        return None
    path = root / "FP_prompts" / f"{fpname}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def mean_pairwise_jaccard(sets: list[set]) -> float:
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            vals.append(jaccard(sets[i], sets[j]))
    return float(np.mean(vals)) if vals else 1.0


def mean_pairwise_spearman(weight_dicts: list[dict]) -> float:
    from scipy.stats import spearmanr
    vals = []
    for i in range(len(weight_dicts)):
        for j in range(i + 1, len(weight_dicts)):
            keys = set(weight_dicts[i]) | set(weight_dicts[j])
            if len(keys) < 3:
                continue
            xi = [weight_dicts[i].get(k, 0.0) for k in keys]
            xj = [weight_dicts[j].get(k, 0.0) for k in keys]
            rho, _ = spearmanr(xi, xj)
            if rho == rho:  # not nan
                vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(vals: np.ndarray, n: int = 10000, seed: int = SEED) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    try:
        from shared.seeds import set_global_seeds
        set_global_seeds(SEED)
    except Exception:
        pass

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from lime.lime_text import LimeTextExplainer
    from bert_hybrid_improvements_v2 import load_e28a_data

    stop = set(ENGLISH_STOP_WORDS)

    pipeline = load_pipeline()

    # BERT-Ridge identico ao braco do E30f / dos materiais M3
    train = load_e28a_data(TRAIN_DIR)
    tr_texts = [d["raw_text"] for d in train]
    y_tr = np.array([d["y_true"] for d in train], float)
    emb_tr = embed(tr_texts)
    scaler = StandardScaler().fit(emb_tr)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(emb_tr), y_tr)

    def predict_fn(texts):
        e = embed(list(texts))
        return ridge.predict(scaler.transform(e)).reshape(-1, 1)

    ids = holdout_ids()
    rows = []
    for idx, man_id in enumerate(ids):
        text = fp_text(man_id)
        if text is None:
            print(f"  [warn] sem texto p/ {man_id}", file=sys.stderr)
            continue

        # conjunto funcional F: tokens dos trechos citados pelo trace
        res = pipeline.measure(text, fp_id=man_id)
        F = set()
        n_mov = 0
        for m in res.movements:
            mt = str(m.movement_type).split(".")[-1]
            if mt in ("E", "X", "R", "W"):
                n_mov += 1
                F |= set(tokenize(getattr(m, "source_text", "") or ""))

        # K execucoes do LIME (sementes distintas)
        topk_sets, weight_dicts = [], []
        g_van, g_con = [], []
        for k in range(K_RUNS):
            explainer = LimeTextExplainer(random_state=SEED + k)
            exp = explainer.explain_instance(
                text, predict_fn, labels=(0,),
                num_features=NUM_FEATURES, num_samples=NUM_SAMPLES)
            pairs = [(tok.lower(), w) for tok, w in exp.as_list(label=0)]
            wd = {}
            for tok, w in pairs:
                wd[tok] = wd.get(tok, 0.0) + w
            weight_dicts.append(wd)
            ordered = sorted(pairs, key=lambda p: abs(p[1]), reverse=True)
            topk_sets.append({t for t, _ in ordered[:TOPK]})

            mass_all = sum(abs(w) for _, w in pairs)
            mass_F = sum(abs(w) for t, w in pairs if t in F)
            mass_con = sum(abs(w) for t, w in pairs if t not in stop)
            mass_F_con = sum(abs(w) for t, w in pairs if t in F and t not in stop)
            if mass_all > 0:
                g_van.append(mass_F / mass_all)
            if mass_con > 0:
                g_con.append(mass_F_con / mass_con)

        row = {
            "id": man_id,
            "n_movements": n_mov,
            "n_functional_tokens": len(F),
            "lime_jaccard_topk": mean_pairwise_jaccard(topk_sets),
            "lime_spearman": mean_pairwise_spearman(weight_dicts),
            "lime_groundedness_all": float(np.mean(g_van)) if g_van else float("nan"),
            "lime_groundedness_content": float(np.mean(g_con)) if g_con else float("nan"),
        }
        rows.append(row)
        print(f"[{idx+1}/{len(ids)}] {man_id}: "
              f"Jaccard={row['lime_jaccard_topk']:.2f} "
              f"g_all={row['lime_groundedness_all']:.2f} "
              f"g_content={row['lime_groundedness_content']:.2f}")

    # persistir per-FP
    fields = ["id", "n_movements", "n_functional_tokens", "lime_jaccard_topk",
              "lime_spearman", "lime_groundedness_all", "lime_groundedness_content"]
    with (OUT / "per_fp.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def agg(key):
        v = np.array([r[key] for r in rows], float)
        v = v[~np.isnan(v)]
        lo, hi = bootstrap_ci(v)
        return v.mean(), np.median(v), lo, hi, len(v)

    jm = agg("lime_jaccard_topk")
    sm = agg("lime_spearman")
    ga = agg("lime_groundedness_all")
    gc = agg("lime_groundedness_content")

    md = [
        "# M3 — Avaliacao functionally-grounded (sem sujeitos humanos)",
        "",
        f"Hold-out De Vito, {len(rows)} FPs. LIME: K={K_RUNS} reexecucoes/FP, "
        f"num_features={NUM_FEATURES}, num_samples={NUM_SAMPLES}, top-k={TOPK}. "
        f"IC95 bootstrap percentilico (10k, seed {SEED}).",
        "",
        "## Metrica 2 — Estabilidade da explicacao",
        "",
        "| Explicacao | Jaccard top-k [IC95] | Spearman pesos [IC95] |",
        "|---|---|---|",
        f"| Trace simbolico (deterministico) | 1,000 (variancia 0) | 1,000 (variancia 0) |",
        f"| LIME post-hoc | {jm[0]:.3f} [{jm[2]:.3f}; {jm[3]:.3f}] | "
        f"{sm[0]:.3f} [{sm[2]:.3f}; {sm[3]:.3f}] |",
        "",
        "## Metrica 3 — Ancoragem semantica (massa de atribuicao em trechos funcionais)",
        "",
        "| Explicacao | Ancoragem [IC95] |",
        "|---|---|",
        f"| Trace simbolico (por construcao) | 1,000 |",
        f"| LIME — todos os tokens | {ga[0]:.3f} [{ga[2]:.3f}; {ga[3]:.3f}] |",
        f"| LIME — sem stopwords | {gc[0]:.3f} [{gc[2]:.3f}; {gc[3]:.3f}] |",
        "",
        f"medianas: Jaccard {jm[1]:.3f}; ancoragem todos {ga[1]:.3f}; "
        f"sem stopwords {gc[1]:.3f}.",
        "",
        "Metrica 1 (fidelidade/correcao do trace): ver results/movement-level/ "
        "(F1 por tipo de movimento contra gabarito oficial).",
    ]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
