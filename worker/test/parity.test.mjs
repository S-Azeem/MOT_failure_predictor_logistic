// Day 4, part 2: the parity test.
//
// Runs the real cars exported by pipeline/failure_model/export_parity_cases.py
// through the Worker's OWN feature and scoring code, and checks every value
// against what Python computed. Run from the worker folder:
//     npm test
import fs from "fs";
import { fileURLToPath } from "url";
import path from "path";
import { buildFeatures, dayNumber } from "../src/features.js";
import { cleanMake, inScope, riskBand, score } from "../src/model.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const model = JSON.parse(fs.readFileSync(path.join(here, "../../models/model_v1.json")));
const cases = JSON.parse(fs.readFileSync(path.join(here, "parity_cases.json")));

// SQLite and JavaScript can round the age to 2 decimals differently at an exact
// half; allow one hundredth for age, and a matching hair for the probability.
const TOL = { age_years: 0.0101, default: 1e-6, probability: 1e-4 };

let failures = 0;
const byGroup = {};
for (const c of cases) {
  const problems = [];
  const reg = c.vehicle.registration;
  const f = buildFeatures(c.vehicle, dayNumber(c.target_date));

  for (const [name, want] of Object.entries(c.expected.features)) {
    const got = f[name];
    const tol = TOL[name] ?? TOL.default;
    const same = (want === null && got === null) ||
                 (want !== null && got !== null && Math.abs(want - got) <= tol);
    if (!same) problems.push(`${name}: python ${want}, js ${got}`);
  }

  const make = cleanMake(c.vehicle.make, model.vehicle_rules);
  if (make !== c.expected.make) problems.push(`make: python ${c.expected.make}, js ${make}`);
  if (!inScope(c.vehicle.make, c.vehicle.model, model.vehicle_rules)) {
    problems.push("in scope: python yes, js no");
  }

  const { probability } = score(f, make, model);
  if (Math.abs(probability - c.expected.probability) > TOL.probability) {
    problems.push(`probability: python ${c.expected.probability.toFixed(6)}, js ${probability.toFixed(6)}`);
  }
  const band = riskBand(probability, f.age_years, model);
  if (band !== c.expected.band) problems.push(`band: python ${c.expected.band}, js ${band}`);

  byGroup[c.group] ??= { cases: 0, failed: 0 };
  byGroup[c.group].cases++;
  if (problems.length) {
    failures++;
    byGroup[c.group].failed++;
    console.log(`MISMATCH ${reg} (${c.group}, target ${c.target_date})`);
    for (const p of problems) console.log(`    ${p}`);
  }
}

console.log("\nBy group:");
for (const [g, r] of Object.entries(byGroup)) {
  console.log(`  ${g.padEnd(12)} ${r.cases - r.failed}/${r.cases} match`);
}
console.log(`\n${cases.length - failures}/${cases.length} cars match on every feature, make, probability and band`);
process.exit(failures ? 1 : 0);
