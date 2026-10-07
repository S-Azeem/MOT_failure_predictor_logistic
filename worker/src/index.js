// Roadworthy MOT failure model: Cloudflare Worker
//
//   GET /score?reg=AB12CDE
//
// 1. check the cache   2. fetch MOT history from DVSA   3. build the features
// 4. apply the model   5. choose the risk band
// Secrets (set with `npx wrangler secret put NAME`, never in code):
//   MOT_CLIENT_ID, MOT_CLIENT_SECRET, MOT_API_KEY, MOT_TOKEN_URL
// Setting (wrangler.toml [vars]): ALLOWED_ORIGINS, comma-separated

import model from "../../models/model_v1.json";
import { buildFeatures, dayNumber, predictionDay } from "./features.js";
import { cleanMake, inScope, riskBand, score } from "./model.js";

const API_BASE = "https://history.mot.api.gov.uk";
const SCOPE = "https://tapi.dvsa.gov.uk/.default";
const CACHE_SECONDS = 86400;
const MS_PER_DAY = 86400000;

let cachedToken = null;   // reused across requests while this Worker instance lives

async function getToken(env) {
  if (cachedToken && cachedToken.expires > Date.now() + 60000) return cachedToken.value;
  const resp = await fetch(env.MOT_TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "client_credentials",
      client_id: env.MOT_CLIENT_ID,
      client_secret: env.MOT_CLIENT_SECRET,
      scope: SCOPE,
    }),
  });
  if (!resp.ok) throw new Error(`token request failed: ${resp.status}`);
  const data = await resp.json();
  cachedToken = { value: data.access_token, expires: Date.now() + (data.expires_in || 3600) * 1000 };
  return cachedToken.value;
}

async function fetchVehicle(reg, env) {
  const url = `${API_BASE}/v1/trade/vehicles/registration/${encodeURIComponent(reg)}`;
  for (let attempt = 0; attempt < 3; attempt++) {
    const resp = await fetch(url, {
      headers: { Authorization: `Bearer ${await getToken(env)}`, "X-API-Key": env.MOT_API_KEY },
    });
    if (resp.status === 404) return null;
    if (resp.status === 429 || resp.status >= 500) {          // back off and retry
      await new Promise(r => setTimeout(r, 500 * 2 ** attempt));
      continue;
    }
    if (!resp.ok) throw new Error(`DVSA API error: ${resp.status}`);
    return resp.json();
  }
  throw new Error("DVSA API busy");
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = (env.ALLOWED_ORIGINS || "").split(",").map(s => s.trim());
  return {
    "Access-Control-Allow-Origin": allowed.includes(origin) ? origin : allowed[0] || "",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    Vary: "Origin",
  };
}

function json(body, status, headers) {
  return new Response(JSON.stringify(body), {
    status, headers: { "Content-Type": "application/json", ...headers },
  });
}

export async function scoreVehicle(vehicle, todayDay) {
  const rules = model.vehicle_rules;
  const make = cleanMake(vehicle.make, rules);
  const base = { registration: vehicle.registration, make: vehicle.make, model: vehicle.model };
  if (!inScope(vehicle.make, vehicle.model, rules)) {
    return { ...base, supported: false,
             message: "Motorcycles and scooters aren't covered: this model is for cars and light vans." };
  }
  const predDay = predictionDay(vehicle, todayDay);
  const features = buildFeatures(vehicle, predDay);
  if (features.age_years === null) {
    return { ...base, supported: false, message: "No first-registration date on record." };
  }
  const { probability } = score(features, make, model);
  return {
    ...base,
    supported: true,
    due_date: new Date(predDay * MS_PER_DAY).toISOString().slice(0, 10),
    band: riskBand(probability, features.age_years, model),
    probability: Math.round(probability * 1000) / 1000,
    first_mot: features.n_prior_fails === null,
    features,
    model_version: model.version,
  };
}

export default {
  async fetch(request, env, ctx) {
    const cors = corsHeaders(request, env);
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });

    const url = new URL(request.url);
    if (url.pathname !== "/score") return json({ error: "Not found" }, 404, cors);
    const reg = (url.searchParams.get("reg") || "").replace(/\s+/g, "").toUpperCase();
    if (!/^[A-Z0-9]{2,8}$/.test(reg)) {
      return json({ error: "That doesn't look like a UK registration." }, 400, cors);
    }

    // 1. cache: one day per plate (the Cache API is inactive on *.workers.dev URLs)
    const cache = caches.default;
    const cacheKey = new Request(`https://cache.roadworthy.internal/score/${reg}`);
    const hit = await cache.match(cacheKey);
    if (hit) return new Response(hit.body, { status: hit.status, headers: { ...Object.fromEntries(hit.headers), ...cors } });

    try {
      const vehicle = await fetchVehicle(reg, env);                         // 2.
      if (!vehicle) return json({ error: "No vehicle found with that registration." }, 404, cors);
      const todayDay = dayNumber(new Date().toISOString());
      const result = await scoreVehicle(vehicle, todayDay);                 // 3-5.
      const resp = json(result, 200, { "Cache-Control": `max-age=${CACHE_SECONDS}` });
      ctx.waitUntil(cache.put(cacheKey, resp.clone()));
      return new Response(resp.body, { status: 200, headers: { ...Object.fromEntries(resp.headers), ...cors } });
    } catch (err) { console.error("score failed:", reg, err.stack || err);
      return json({ error: "The MOT service is unavailable right now. Please try again shortly." }, 503, cors);
    }
  },
};
