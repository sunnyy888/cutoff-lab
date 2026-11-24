// Tests for the page's metric code against numbers computed by scikit-learn in train.py. Run: node test.mjs
import { readFileSync } from "node:fs";
const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const Lab = new Function("module", html.match(/<script id="core">([\s\S]*?)<\/script>/)[1] + "; return Lab;")({});
const window = {}; new Function("window", readFileSync(new URL("./data.js", import.meta.url), "utf8"))(window);
const D = window.CUTOFF_DATA;
let pass = 0, fail = 0; const check = (n, ok, d = "") => { ok ? pass++ : fail++; console.log(`${ok ? "PASS" : "FAIL"}  ${n}${d ? ": " + d : ""}`); };

for (const [name, m] of Object.entries(D.models)) for (const split of ["validation", "test"]) {
  const h = m[split];
  // 1. confusion matrix from the 1,000-bin histogram == exact counts from the raw scores (train.py)
  for (const [c, [TP, FP, FN, TN]] of Object.entries(h.check)) {
    const x = Lab.confusion(h, Number(c));
    check(`${name} ${split} @${c}: confusion = sklearn counts`, x.TP === TP && x.FP === FP && x.FN === FN && x.TN === TN, `${x.TP}/${x.FP}/${x.FN}/${x.TN}`);
  }
  // 2. AUC from the histogram ~ roc_auc_score on raw scores (binning to 0.001 loses almost nothing)
  const r = Lab.roc(h);
  check(`${name} ${split}: binned AUC = sklearn AUC`, Math.abs(r.auc - h.auc) < 5e-4, `${r.auc.toFixed(4)} vs ${h.auc.toFixed(4)}`);
  check(`${name} ${split}: accuracy@0.5 = sklearn`, Math.abs(Lab.metrics(h, 0.5, 1, 1).acc - h.acc50) < 1e-12);
}
// 3. metrics definitions on a hand-made histogram (B = 10): scores .05..95, positives in high bins
{
  const h = { pos: [0, 0, 1, 0, 1, 2, 0, 3, 2, 1], neg: [4, 3, 2, 2, 1, 1, 1, 0, 0, 0] };
  const m = Lab.metrics(h, 0.5, 300, 180);   // flagged bins 5..9: TP 8, FP 2; FN 2, TN 12
  check("hand histogram @0.5", m.TP === 8 && m.FP === 2 && m.FN === 2 && m.TN === 12 && m.cost === 300 * 2 + 180 * 2 && Math.abs(m.prec - 0.8) < 1e-12 && Math.abs(m.rec - 0.8) < 1e-12, JSON.stringify([m.TP, m.FP, m.FN, m.TN, m.cost]));
  // AUC by brute-force pair counting with ties (same bin) = 1/2
  let num = 0; const P = 10, N = 14;
  h.pos.forEach((p, i) => h.neg.forEach((q, j) => { num += p * q * (i > j ? 1 : i === j ? 0.5 : 0); }));
  check("trapezoid ROC AUC = pair counting with ties", Math.abs(Lab.roc(h).auc - num / (P * N)) < 1e-12, (num / (P * N)).toFixed(4));
}
// 4. cheapest and most accurate are true optima of the sweep; break-even formula
{
  const h = D.models["Gradient boosting"].validation, b = Lab.best(h, 250, 100);
  check("cheapest cutoff is the sweep minimum", b.sweep.every(m => m.cost >= b.sweep[Math.round(b.cheapest * 100) - 1].cost));
  check("most accurate cutoff is the sweep maximum", b.sweep.every(m => m.acc <= b.sweep[Math.round(b.accurate * 100) - 1].acc));
  check("break-even = cFP / (cFP + cFN)", Math.abs(Lab.breakEven(250, 100) - 250 / 350) < 1e-12);
  check("higher FP cost pushes the cheapest cutoff up", Lab.best(h, 500, 100).cheapest >= Lab.best(h, 100, 500).cheapest, `${Lab.best(h, 500, 100).cheapest} vs ${Lab.best(h, 100, 500).cheapest}`);
}
// 5. data hygiene recorded in data.js
{
  const dropped = Object.entries(D.meta.dropped).filter(([, v]) => !v.startsWith("Kept")).map(([k]) => k);
  check("no dropped column is used as a feature", !dropped.some(c => D.meta.features.includes(c)), dropped.join(", "));
  check("time split is ordered and disjoint", D.meta.dates.train[1] < D.meta.dates.validation[0] && D.meta.dates.validation[1] < D.meta.dates.test[0]);
  check("leaky model scores higher than honest one (the trap is real)", D.meta.leaky_auc.validation > D.models["Gradient boosting"].validation.auc + 0.03);
}
console.log(`\n${pass}/${pass + fail} PASS`); process.exit(fail ? 1 : 0);
