// Build the model's features from a DVSA API vehicle record.
//
// This MUST match pipeline/failure_model/build_training_all.py exactly; the
// Day 4 parity test checks it. Rules mirrored here:
//   - GB (DVSA) tests only, PASSED or FAILED results only
//   - tests sorted by date, a fail before a pass at the same time
//   - a test within 60 days of the previous test is a retest and is dropped
//   - history features come from the kept cycles before the prediction
//   - odometer counts only when the reading type is READ; km converted to miles

const RETEST_GAP_DAYS = 60;
const FAIL_TYPES = new Set(["FAIL", "MAJOR", "DANGEROUS", "PRS"]);
const MS_PER_DAY = 86400000;

// "2024-05-20T10:31:00.000Z" -> days since 1970 for that calendar date (UTC)
export function dayNumber(dateText) {
  const d = String(dateText).slice(0, 10);
  return Date.UTC(+d.slice(0, 4), +d.slice(5, 7) - 1, +d.slice(8, 10)) / MS_PER_DAY;
}

function miles(test) {
  if (String(test.odometerResultType || "").toUpperCase() !== "READ") return null;
  const v = parseFloat(test.odometerValue);
  if (Number.isNaN(v)) return null;
  return String(test.odometerUnit || "").toLowerCase().startsWith("k") ? v * 0.621371 : v;
}

// Keep GB pass/fail tests, sorted, with retests removed
export function testCycles(vehicle) {
  const tests = (vehicle.motTests || [])
    .filter(t => !t.dataSource || String(t.dataSource).toUpperCase() === "DVSA")
    .filter(t => ["PASSED", "FAILED"].includes(String(t.testResult).toUpperCase()))
    .map(t => ({
      date: String(t.completedDate),
      day: dayNumber(t.completedDate),
      failed: String(t.testResult).toUpperCase() === "FAILED" ? 1 : 0,
      miles: miles(t),
      failItems: (t.defects || []).filter(d => FAIL_TYPES.has(String(d.type).toUpperCase())).length,
      advisories: (t.defects || []).filter(d => String(d.type).toUpperCase() === "ADVISORY").length,
      expiry: t.expiryDate ? dayNumber(t.expiryDate) : null,
    }))
    // date order; at the same moment a fail comes first
    .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : b.failed - a.failed));

  const cycles = [];
  let prevDay = null;                       // the previous test of ANY kind
  for (const t of tests) {
    if (prevDay === null || t.day - prevDay > RETEST_GAP_DAYS) cycles.push(t);
    prevDay = t.day;
  }
  return cycles;
}

// The date the prediction is for: the MOT due date, or today if it has lapsed.
// Uses ALL passed tests (retests included): a retest pass sets the new expiry.
// A car with no MOT yet is due on its third anniversary of first use.
export function predictionDay(vehicle, todayDay) {
  const expiries = (vehicle.motTests || [])
    .filter(t => String(t.testResult).toUpperCase() === "PASSED" && t.expiryDate)
    .map(t => dayNumber(t.expiryDate));
  let due;
  if (expiries.length) {
    due = Math.max(...expiries);
  } else {
    const start = startDay(vehicle);
    if (start === null) return todayDay;
    const d = new Date(start * MS_PER_DAY);
    due = Date.UTC(d.getUTCFullYear() + 3, d.getUTCMonth(), d.getUTCDate()) / MS_PER_DAY;
  }
  return Math.max(due, todayDay);
}

export function startDay(vehicle) {
  const s = vehicle.firstUsedDate || vehicle.registrationDate;
  return s ? dayNumber(s) : null;
}

// History features as of `predDay`, from cycles strictly before it
export function buildFeatures(vehicle, predDay) {
  const all = testCycles(vehicle);
  const cycles = all.filter(c => c.day < predDay);
  const start = startDay(vehicle);
  const n = cycles.length;
  const last = n ? cycles[n - 1] : null;
  const sum = arr => arr.reduce((s, c) => s + c.failed, 0);
  return {
    age_years: start === null ? null : Math.round(((predDay - start) / 365.25) * 100) / 100,
    n_prior_fails: n ? sum(cycles) : null,
    fails_last_3: n ? sum(cycles.slice(-3)) : null,
    prev_odometer: last ? last.miles : null,
    prev_fail_items: last ? last.failItems : null,
    prev_advisories: last ? last.advisories : null,
    days_since_prev: last ? predDay - last.day : null,
  };
}
