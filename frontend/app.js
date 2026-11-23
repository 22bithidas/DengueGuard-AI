const byId = id => document.getElementById(id);
const state = { snapshot: null };
const pretty = name => name.replace(/^\w+__/, "").replaceAll("_", " ").replace(/\b\w/g, x => x.toUpperCase());

function showError(message) {
  const box = byId("error");
  box.textContent = message || "";
  box.hidden = !message;
}

function setOptions(select, values, selected) {
  select.replaceChildren(...values.map(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selected;
    return option;
  }));
}

function selectedRows() {
  return state.snapshot.cities[byId("city").value]?.wards[byId("ward").value] || [];
}

function updateCityOptions() {
  const snapshot = state.snapshot;
  const cities = Object.keys(snapshot.cities).sort();
  const currentCity = cities.includes(byId("city").value) ? byId("city").value : cities[0];
  setOptions(byId("city"), cities, currentCity);
  const wards = Object.keys(snapshot.cities[currentCity].wards).sort();
  const currentWard = wards.includes(byId("ward").value) ? byId("ward").value : wards[0];
  setOptions(byId("ward"), wards, currentWard);
  byId("wardCount").textContent = `${wards.length} WARDS`;
  updateDateOptions();
}

function updateDateOptions() {
  const rows = selectedRows();
  const dates = rows.map(row => row.date);
  const currentDate = dates.includes(byId("date").value) ? byId("date").value : dates.at(-1);
  setOptions(byId("date"), [...dates].reverse(), currentDate);
  updateDashboard();
}

function drawChart(history) {
  if (!history?.length) {
    byId("chart").textContent = "No case history available for this ward.";
    return;
  }
  const width = 680, height = 132, padding = 10;
  const cases = history.map(d => d.cases), max = Math.max(...cases, 1);
  const points = cases.map((value, i) => `${padding + i * (width - 2 * padding) / Math.max(1, cases.length - 1)},${height - padding - value / max * (height - 2 * padding)}`).join(" ");
  const area = `${padding},${height - padding} ${points} ${width - padding},${height - padding}`;
  byId("chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Observed weekly dengue cases"><defs><linearGradient id="fade" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#41d2b0" stop-opacity=".25"/><stop offset="1" stop-color="#41d2b0" stop-opacity="0"/></linearGradient></defs><polygon points="${area}" fill="url(#fade)"/><polyline points="${points}" fill="none" stroke="#41d2b0" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  byId("chartFirst").textContent = history[0].date.slice(0, 7);
  byId("chartLast").textContent = history.at(-1).date.slice(0, 7);
}

function drawDrivers(items) {
  byId("driverList").innerHTML = items.length
    ? items.map(item => `<div class="driver"><span class="driver-name">${pretty(item.feature)}</span><span class="driver-percent">${item.importance_pct.toFixed(1)}%</span><div class="driver-bar"><i style="width:${Math.min(100, item.importance_pct * 3)}%"></i></div></div>`).join("")
    : "<p class='tier-note'>Feature importance is not exposed by the selected model.</p>";
}

function drawWatchlist(result) {
  byId("rank").textContent = result.selected_rank ? `#${result.selected_rank} / ${result.ward_count}` : "—";
  byId("watchText").textContent = `Top wards by ${result.horizon_weeks}-week predicted cases · ${result.city}`;
  byId("watchRows").innerHTML = result.watchlist.map(row => `<tr class="${row.ward_id === result.ward_id ? "current" : ""}"><td class="ward">${row.ward_id}</td><td><span class="badge ${row.risk_tier.toLowerCase()}">${row.risk_tier}</span></td><td class="cases">${row.predicted_cases.toFixed(2)}</td></tr>`).join("");
}

function makeResult() {
  const snapshot = state.snapshot;
  const city = byId("city").value;
  const ward = byId("ward").value;
  const date = byId("date").value;
  const horizon = Number(byId("horizon").value);
  const rows = selectedRows();
  const row = rows.find(item => item.date === date) || rows.at(-1);
  const history = rows.filter(item => item.date <= row.date).slice(-26);
  const suffix = String(horizon);
  const modelKey = `${horizon}_week`;
  return {
    city,
    ward_id: ward,
    date: row.date,
    horizon_weeks: horizon,
    model_name: snapshot.selected_models[modelKey],
    predicted_cases: row[`forecast_${suffix}`],
    risk_tier: row[`tier_${suffix}`],
    weather: row.weather,
    drivers: snapshot.drivers[suffix],
    history: history.map(item => ({ date: item.date, cases: item.cases })),
    watchlist: snapshot.watchlists[city][suffix],
    selected_rank: snapshot.ranks[city][suffix][ward],
    ward_count: Object.keys(snapshot.cities[city].wards).length,
    test_metrics: snapshot.test_metrics[modelKey],
    validation_metrics: snapshot.validation_metrics[modelKey],
  };
}

function render(result) {
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
  drawChart(result.history);
  drawDrivers(result.drivers);
  drawWatchlist(result);
}

function updateDashboard() {
  if (!state.snapshot || !byId("date").value) return;
  try {
    render(makeResult());
    showError("");
  } catch (error) {
    showError(`Could not show the saved forecast: ${error.message}`);
  }
}

async function initialize() {
  try {
    const response = await fetch("/dashboard-data.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Dashboard data file returned HTTP ${response.status}`);
    state.snapshot = await response.json();
    byId("modelStatus").textContent = "STATIC MODEL DATA READY";
    byId("datasetStatus").textContent = `${state.snapshot.dataset_rows.toLocaleString()} rows · ${state.snapshot.ward_count} wards`;
    byId("latestDate").textContent = state.snapshot.latest_date;
    byId("footerDate").textContent = state.snapshot.latest_date;
    updateCityOptions();
  } catch (error) {
    byId("modelStatus").textContent = "DATA FILE MISSING";
    byId("datasetStatus").textContent = "Static build required";
    showError(`Dashboard snapshot did not load: ${error.message}. Follow the Static dashboard deployment steps in README.md.`);
  }
}

byId("city").addEventListener("change", updateCityOptions);
byId("ward").addEventListener("change", updateDateOptions);
byId("date").addEventListener("change", updateDashboard);
byId("horizon").addEventListener("change", updateDashboard);
byId("selectors").addEventListener("submit", event => { event.preventDefault(); updateDashboard(); });

initialize();
