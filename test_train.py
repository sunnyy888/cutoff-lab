"""Pipeline checks for train.py. Run: .venv/bin/python test_train.py"""
import train

ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, (": " + detail) if detail else ""))


d = train.load()
tr, va, te = train.split(d)
feats = train.CAT + train.NUM
check("split covers every booking exactly once", len(tr) + len(va) + len(te) == len(d), f"{len(tr)} + {len(va)} + {len(te)} = {len(d)}")
check("train < validation < test by arrival date", tr.arrival.max() < va.arrival.min() and va.arrival.max() < te.arrival.min())
check("no dropped column among the features", not set(train.DROP) & set(feats), ", ".join(train.DROP))
check("outcome not among the features", "is_canceled" not in feats and "reservation_status" not in feats)
top_from_train = set(tr["agent"].value_counts().head(20).index)
used = {int(a.split("_")[1]) for a in d["agent_top"].unique() if a != "other"}
check("top-agent categories come from training data only", used == {int(x) for x in top_from_train}, f"{len(used)} agents")
check("no missing values in features", d[feats].isna().sum().sum() == 0)
check("leaky columns really are leaky (parking > 0 never cancels)", d[d.required_car_parking_spaces > 0].is_canceled.mean() == 0)
print("\n%d/%d PASS" % (sum(ok), len(ok)))
raise SystemExit(0 if all(ok) else 1)
