"""Cutoff Lab: train cancellation models on the public hotel-booking dataset and export data for the web page.

Data:   data/hotels.csv (Antonio, Almeida & Nunes 2019, "Hotel booking demand datasets", Data in Brief 22;
        TidyTuesday mirror). 119,390 bookings, two Portuguese hotels, arrivals Jul 2015 - Aug 2017.
Task:   at booking time, predict whether the booking will be canceled (is_canceled).
Split:  by arrival date (no shuffling): train = arrivals up to 2016-12-31, validation = Jan-Apr 2017,
        test = May-Aug 2017. Cutoffs are chosen on validation and reported on test.
Models: logistic regression (one-hot + standardized numerics, L2) vs histogram gradient boosting.
Output: data.js (score histograms, ROC, metrics, importances) for index.html, and report.json.
Run:    .venv/bin/python train.py        (about 1 minute)
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
BINS = 1000                                           # score histogram resolution for the page (0.001)

# ---------------- leakage: columns we may NOT use, and why ----------------
DROPPED = {
    "reservation_status": "The outcome itself: Canceled / No-Show / Check-Out maps one-to-one onto is_canceled.",
    "reservation_status_date": "The date the status was last set, i.e. the cancellation or check-out date. Known only after the fact.",
    "assigned_room_type": "Assigned at check-in. A room different from the one reserved means the guest showed up (5% cancel vs 42%).",
    "booking_changes": "Counts changes made up to arrival or cancellation, so it includes the future (16% cancel if > 0 vs 41%).",
    "required_car_parking_spaces": "Recorded at arrival: bookings with a parking space have a 0.0% cancellation rate (7,416 of them).",
    "total_of_special_requests": "Requests can be added any time before arrival; guests who are coming add them (22% vs 48% cancel). Dropped to be safe.",
    "country": "Often only confirmed from ID at check-in, and canceled bookings default to Portugal (PRT 57% cancel vs 24%).",
    "company": "94% missing; an ID with thousands of levels.",
    "arrival_date_year": "A calendar year the model would never see again in the future; season is kept via month and week.",
    "days_in_waiting_list": "Kept: known once the booking is confirmed.",
}
DROP = [k for k in DROPPED if not DROPPED[k].startswith("Kept")]
CAT = ["hotel", "arrival_date_month", "meal", "market_segment", "distribution_channel", "reserved_room_type",
       "deposit_type", "customer_type", "agent_top"]
NUM = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
       "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
       "previous_bookings_not_canceled", "adr", "days_in_waiting_list", "has_agent"]


def load():
    d = pd.read_csv(os.path.join(HERE, "data", "hotels.csv"))
    d["arrival"] = pd.to_datetime(d.arrival_date_year.astype(str) + "-" + d.arrival_date_month + "-"
                                  + d.arrival_date_day_of_month.astype(str), format="%Y-%B-%d")
    d["children"] = d["children"].fillna(0)
    d["has_agent"] = d["agent"].notna().astype(int)
    top = d.loc[d.arrival <= "2016-12-31", "agent"].value_counts().head(20).index    # top agents from train only
    d["agent_top"] = np.where(d["agent"].isin(top), "agent_" + d["agent"].fillna(0).astype(int).astype(str), "other")
    return d


def split(d):
    tr = d[d.arrival <= "2016-12-31"]
    va = d[(d.arrival >= "2017-01-01") & (d.arrival <= "2017-04-30")]
    te = d[d.arrival >= "2017-05-01"]
    return tr, va, te


def models():
    pre_lr = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", StandardScaler(), NUM)])
    lr = Pipeline([("pre", pre_lr), ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    pre_gb = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT), ("num", "passthrough", NUM)])
    gb = Pipeline([("pre", pre_gb), ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                                                          l2_regularization=1.0, random_state=SEED))])
    return {"Logistic regression": lr, "Gradient boosting": gb}


def hist(y, p):
    b = np.minimum((p * BINS).astype(int), BINS - 1)
    pos = np.bincount(b[y == 1], minlength=BINS)
    neg = np.bincount(b[y == 0], minlength=BINS)
    return pos.tolist(), neg.tolist()


def main():
    d = load()
    tr, va, te = split(d)
    feats = CAT + NUM
    assert not set(DROP) & set(feats), "a dropped column slipped into the features"
    out = {"meta": {"source": "Antonio, Almeida & Nunes (2019), Hotel booking demand datasets, Data in Brief 22",
                    "n": {"train": len(tr), "validation": len(va), "test": len(te)},
                    "rate": {"train": tr.is_canceled.mean(), "validation": va.is_canceled.mean(), "test": te.is_canceled.mean()},
                    "dates": {"train": [str(tr.arrival.min().date()), str(tr.arrival.max().date())],
                              "validation": [str(va.arrival.min().date()), str(va.arrival.max().date())],
                              "test": [str(te.arrival.min().date()), str(te.arrival.max().date())]},
                    "bins": BINS, "dropped": DROPPED, "features": feats},
           "models": {}}
    for name, m in models().items():
        m.fit(tr[feats], tr.is_canceled)
        res = {}
        for split_name, part in (("validation", va), ("test", te)):
            y, p = part.is_canceled.to_numpy(), m.predict_proba(part[feats])[:, 1]
            pos, neg = hist(y, p)
            res[split_name] = {"pos": pos, "neg": neg, "auc": roc_auc_score(y, p), "logloss": log_loss(y, p),
                               "acc50": accuracy_score(y, p >= 0.5),
                               "check": {str(c): [int(((p >= c) & (y == 1)).sum()), int(((p >= c) & (y == 0)).sum()),
                                                  int(((p < c) & (y == 1)).sum()), int(((p < c) & (y == 0)).sum())]
                                         for c in (0.3, 0.5, 0.7)}}
            print("%-20s %-10s AUC %.4f  log loss %.4f  accuracy@0.5 %.4f" % (name, split_name, res[split_name]["auc"],
                                                                             res[split_name]["logloss"], res[split_name]["acc50"]))
        # permutation importance on validation: drop in AUC when a column is shuffled (association, not causation)
        sample = va.sample(n=min(8000, len(va)), random_state=SEED)
        pi = permutation_importance(m, sample[feats], sample.is_canceled, scoring="roc_auc", n_repeats=5,
                                    random_state=SEED, n_jobs=-1)
        imp = sorted(({"feature": f, "drop_auc": float(mu), "sd": float(sd)} for f, mu, sd in
                      zip(feats, pi.importances_mean, pi.importances_std)), key=lambda r: -r["drop_auc"])
        res["importance"] = imp
        out["models"][name] = res
    # what leakage would have done: same boosting model with the post-booking columns put back
    leaky_cat, leaky_num = CAT + ["assigned_room_type", "country"], NUM + ["booking_changes", "required_car_parking_spaces", "total_of_special_requests"]
    dd = d.copy(); dd["country"] = dd["country"].fillna("unknown")
    ltr, lva, lte = split(dd)
    leaky = Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20), leaky_cat),
                                                 ("num", "passthrough", leaky_num)])),
                      ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=SEED))])
    leaky.fit(ltr[leaky_cat + leaky_num], ltr.is_canceled)
    out["meta"]["leaky_auc"] = {s: float(roc_auc_score(part.is_canceled, leaky.predict_proba(part[leaky_cat + leaky_num])[:, 1]))
                                for s, part in (("validation", lva), ("test", lte))}
    print("LEAKY gradient boosting (for comparison only): AUC validation %.4f, test %.4f" % tuple(out["meta"]["leaky_auc"].values()))
    # a sanity number for the causation note
    out["meta"]["nonrefund_cancel_rate"] = float(d[d.deposit_type == "Non Refund"].is_canceled.mean())
    out["meta"]["nodeposit_cancel_rate"] = float(d[d.deposit_type == "No Deposit"].is_canceled.mean())
    with open(os.path.join(HERE, "data.js"), "w") as f:
        f.write("// generated by train.py; do not edit\nwindow.CUTOFF_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
    summary = {k: {s: {"auc": v[s]["auc"], "logloss": v[s]["logloss"], "acc50": v[s]["acc50"]} for s in ("validation", "test")}
               for k, v in out["models"].items()}
    summary["meta"] = {k: out["meta"][k] for k in ("n", "rate", "dates", "leaky_auc")}
    summary["top_importance"] = {k: v["importance"][:5] for k, v in out["models"].items()}
    with open(os.path.join(HERE, "report.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote data.js (%d KB) and report.json" % (os.path.getsize(os.path.join(HERE, "data.js")) // 1024))


if __name__ == "__main__":
    main()
