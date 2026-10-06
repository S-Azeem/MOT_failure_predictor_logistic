// Apply the exported model JSON. Mirrors pipeline/failure_model/scoring.py.

const NUMERIC = ["age_years", "n_prior_fails", "fails_last_3", "prev_odometer",
                 "prev_fail_items", "prev_advisories", "days_since_prev"];
const ZERO_IF_NO_HISTORY = new Set(["n_prior_fails", "fails_last_3",
                                    "prev_fail_items", "prev_advisories"]);

export function cleanMake(make, rules) {
  const m = String(make || "").trim().toUpperCase();
  return rules.make_merges[m] || m || "UNKNOWN";
}

export function inScope(make, model, rules) {
  const mk = cleanMake(make, rules);
  const md = String(model || "").trim().toUpperCase();
  if (rules.excluded_makes.includes(mk)) return false;
  if (rules.car_models[mk]) {
    const first = md ? md.split(" ")[0] : "";
    return rules.car_models[mk].includes(first);
  }
  if (mk === "BMW") {
    return !rules.unknown_models.includes(md) && !new RegExp(rules.bmw_bike_regex).test(md);
  }
  return true;
}

export function designRow(features, make, model) {
  const prep = model.preprocessing;
  const noHistory = features.n_prior_fails === null;
  const row = {};
  for (const c of NUMERIC) {
    let v = features[c];
    if (v === null || v === undefined) v = ZERO_IF_NO_HISTORY.has(c) ? 0 : prep.fill_values[c];
    row[c] = (v - prep.means[c]) / prep.sds[c];
  }
  row.no_history = noHistory ? 1 : 0;
  row.odometer_missing = features.prev_odometer === null && !noHistory ? 1 : 0;
  for (const m of prep.makes) row[`make_${m}`] = make === m ? 1 : 0;
  return model.columns.map(c => row[c]);
}

export function score(features, make, model) {
  const x = designRow(features, make, model);
  const z = model.intercept + x.reduce((s, v, i) => s + v * model.coefficients[i], 0);
  const { a, b } = model.calibration;
  const probability = 1 / (1 + Math.exp(-(a * z + b)));
  return { rawLogit: z, probability };
}

export function ageKey(ageYears) {
  const a = Math.floor(ageYears);
  return a >= 25 ? "25+" : String(Math.max(a, 3));
}

export function riskBand(probability, ageYears, model) {
  const [lo, hi] = model.bands.cutoffs_by_age[ageKey(ageYears)];
  return probability < lo ? "Lower risk" : probability < hi ? "Typical" : "Higher risk";
}
