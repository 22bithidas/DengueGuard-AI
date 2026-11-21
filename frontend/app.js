const byId = id => document.getElementById(id);
const state = { meta: null, result: null };
const pretty = name => name.replace(/^\w+__/, "").replaceAll("_", " ").replace(/\b\w/g, x => x.toUpperCase());

function showError(message) {
  const box = byId("error"); box.textContent = message || ""; box.hidden = !message;
}
function setOptions(select, values, selected) {
  select.replaceChildren(...values.map(value => {
    const option = document.createElement("option"); option.value = value; option.textContent = value;
    option.selected = value === selected; return option;
  }));
}
async function api(url, init) {
  const response = await fetch(url, init); const result = await response.json();
  if (!response.ok) throw new Error(result.error || result.message || `Request failed (${response.status})`);
  return result;
}
async function loadOptions(city, ward) {
  const query = new URLSearchParams(); if (city) query.set("city", city); if (ward) query.set("ward_id", ward);
  const meta = await api(`/api/options?${query.toString()}`); state.meta = meta;
  setOptions(byId("city"), meta.cities, city || meta.default_city);
  setOptions(byId("ward"), meta.wards, ward || meta.default_ward);
  setOptions(byId("date"), [...meta.dates].reverse(), meta.default_date);
  byId("latestDate").textContent = meta.latest_date;
  byId("footerDate").textContent = meta.latest_date;
  byId("wardCount").textContent = `${meta.wards.length} WARDS`;
  byId("datasetStatus").textContent = `${Number(meta.dataset_rows).toLocaleString()} rows · ${meta.ward_count} wards`;
  byId("modelStatus").textContent = "LOCAL MODELS READY";
  return meta;
}
function drawChart(history) {
  if (!history?.length) { byId("chart").textContent = "No case history available for this ward."; return; }
  const width = 680, height = 132, padding = 10, cases = history.map(d => d.cases), max = Math.max(...cases, 1);
  const points = cases.map((v,i) => `${padding+i*(width-2*padding)/Math.max(1,cases.length-1)},${height-padding-v/max*(height-2*padding)}`).join(" ");
  const area = `${padding},${height-padding} ${points} ${width-padding},${height-padding}`;
  byId("chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Observed weekly dengue cases"><defs><linearGradient id="fade" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#41d2b0" stop-opacity=".25"/><stop offset="1" stop-color="#41d2b0" stop-opacity="0"/></linearGradient></defs><polygon points="${area}" fill="url(#fade)"/><polyline points="${points}" fill="none" stroke="#41d2b0" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  byId("chartFirst").textContent = history[0].date.slice(0,7); byId("chartLast").textContent = history.at(-1).date.slice(0,7);
}
function drawDrivers(items) {
  byId("driverList").innerHTML = items.length ? items.map(x => `<div class="driver"><span class="driver-name">${pretty(x.feature)}</span><span class="driver-percent">${x.importance_pct.toFixed(1)}%</span><div class="driver-bar"><i style="width:${Math.min(100,x.importance_pct*3)}%"></i></div></div>`).join("") : "<p class='tier-note'>Feature importance is not exposed by the selected model.</p>";
}
function drawWatchlist(result) {
  byId("rank").textContent = result.selected_rank ? `#${result.selected_rank} / ${state.meta.wards.length}` : "—";
  byId("watchText").textContent = `Top wards by ${result.horizon_weeks}-week predicted cases · ${result.city}`;
  byId("watchRows").innerHTML = result.watchlist.map(row => `<tr class="${row.ward_id===result.ward_id?"current":""}"><td class="ward">${row.ward_id}</td><td><span class="badge ${row.risk_tier.toLowerCase()}">${row.risk_tier}</span></td><td class="cases">${row.predicted_cases.toFixed(2)}</td><td>${row.date}</td></tr>`).join("");
}
function render(result) {
  state.result = result; showError("");
  byId("cityName").textContent = `/ ${result.city.toUpperCase()}`;
  byId("prediction").textContent = result.predicted_cases.toFixed(2);
  byId("leadLabel").textContent = `· ${result.horizon_weeks} WEEKS AHEAD`;
  byId("riskLabel").className = `risk-label ${result.risk_tier.toLowerCase()}`;
  byId("riskLabel").querySelector("b").textContent = result.risk_tier;
  byId("context").textContent = `${result.ward_id} · ${result.city} · ${result.date}`;
  byId("historyWard").textContent = result.ward_id;
  byId("modelName").textContent = result.model_name;
  byId("mae").textContent = result.test_metrics.mae.toFixed(2);
  byId("r2").textContent = result.test_metrics.r2.toFixed(3);
  byId("testMae").textContent = result.test_metrics.mae.toFixed(2);
  byId("testRmse").textContent = result.test_metrics.rmse.toFixed(2);
  byId("testR2").textContent = result.test_metrics.r2.toFixed(3);
  byId("valMae").textContent = result.validation_metrics.mae.toFixed(2);
  for (const key of ["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]) byId("scenario").elements[key].value = result.weather[key].toFixed(1);
  drawChart(result.history); drawDrivers(result.drivers); drawWatchlist(result);
}
async function predict(weather = {}) {
  const payload = { city: byId("city").value, ward_id: byId("ward").value, date: byId("date").value, horizon_weeks: Number(byId("horizon").value), weather };
  render(await api("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)}));
}
byId("city").addEventListener("change", async () => { try { await loadOptions(byId("city").value); await predict(); } catch(e) { showError(e.message); } });
byId("ward").addEventListener("change", async () => { try { await loadOptions(byId("city").value, byId("ward").value); await predict(); } catch(e) { showError(e.message); } });
byId("selectors").addEventListener("submit", async event => { event.preventDefault(); try { await predict(); } catch(e) { showError(e.message); } });
byId("toggleScenario").addEventListener("click", () => { byId("scenario").hidden = !byId("scenario").hidden; });
byId("scenario").addEventListener("submit", async event => { event.preventDefault(); const weather = Object.fromEntries(new FormData(event.currentTarget)); try { await predict(weather); } catch(e) { showError(e.message); } });
(async () => { try { await loadOptions(); await predict(); } catch(e) { byId("modelStatus").textContent = "SETUP REQUIRED"; byId("datasetStatus").textContent = "Check README setup"; showError(`${e.message}. Install requirements and run python backend\\train_models.py first.`); } })();
