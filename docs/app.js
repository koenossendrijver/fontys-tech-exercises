/* The Manifest - behaviour: theme, scroll reveals, the 891-dot unit chart,
   Chart.js summaries, and the Logistic Regression that runs in the browser. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const root = document.documentElement;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cssVar = (name) => getComputedStyle(root).getPropertyValue(name).trim();

  /* ------------------------------------------------------------ theme */
  const toggle = $("#theme-toggle");
  const savedTheme = localStorage.getItem("manifest-theme");
  if (savedTheme === "light" || savedTheme === "dark") root.dataset.theme = savedTheme;
  const syncToggle = () => {
    const light = root.dataset.theme === "light";
    toggle.setAttribute("aria-pressed", String(light));
    toggle.setAttribute("aria-label", light ? "Switch to dark theme" : "Switch to light theme");
  };
  syncToggle();
  toggle.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem("manifest-theme", root.dataset.theme);
    syncToggle();
    if (state.agg) renderCharts();
  });

  /* ------------------------------------------------------------ reveals */
  const revealObs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      revealObs.unobserve(e.target);
    }
  }, { threshold: 0.12 });
  $$(".reveal").forEach((el) => revealObs.observe(el));

  const fmt = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const countObs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      countObs.unobserve(e.target);
      const el = e.target;
      const target = parseFloat(el.dataset.count);
      const decimals = parseInt(el.dataset.decimals || "0", 10);
      const suffix = el.dataset.suffix || "";
      if (reduced) { el.textContent = fmt(target, decimals) + suffix; continue; }
      const t0 = performance.now();
      const tick = (t) => {
        const p = Math.min(1, (t - t0) / 1500);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = fmt(target * eased, decimals) + suffix;
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  }, { threshold: 0.5 });
  $$("[data-count]").forEach((el) => countObs.observe(el));

  const navLinks = $$(".chapters a");
  const sectionObs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      navLinks.forEach((a) => a.classList.toggle("current", a.dataset.chapter === e.target.id));
    }
  }, { rootMargin: "-35% 0px -60% 0px" });
  $$("main section[id]").forEach((s) => sectionObs.observe(s));

  /* ------------------------------------------------------------ data */
  const state = { agg: null, units: null, model: null, names: null, charts: [] };
  const load = (path) => fetch(path).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); });

  Promise.all([load("data/aggregates.json"), load("data/units.json"), load("data/model.json"), load("data/names.json")])
    .then(([agg, units, model, names]) => {
      Object.assign(state, { agg, units, model, names });
      initTicker();
      initUnitChart();
      renderCharts();
      initPredictor();
      initCourseLinks();
    })
    .catch((err) => {
      console.error(err);
      $("#model-note").textContent = "The data files could not be loaded. Serve this folder over HTTP (for example with `make site`).";
      initCourseLinks();
    });

  /* ------------------------------------------------------------ ticker */
  function initTicker() {
    const track = $("#ticker");
    const CLASS = ["1st", "2nd", "3rd"];
    const items = state.names.map(([name, pclass, survived]) =>
      `<span><b>${escapeHtml(name)}</b> ${CLASS[pclass - 1]} class <i class="${survived ? "" : "d"}">${survived ? "survived" : "died"}</i></span>`);
    track.innerHTML = items.join("") + items.join("");
  }
  function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  /* ------------------------------------------------------------ unit chart */
  function initUnitChart() {
    const NS = "http://www.w3.org/2000/svg";
    const svg = $("#unit-chart");
    const W = 900, ROWS = 27, TOP = 96, MAX_SPACING = 25;
    const U = state.units;               // [survived, pclass, sex(0 f/1 m), age|-1, fare, family, port, titleIdx]
    const gLabels = document.createElementNS(NS, "g");
    const gDots = document.createElementNS(NS, "g");
    svg.append(gLabels, gDots);
    const dots = U.map((u, i) => {
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("r", 9);
      if (u[0]) c.setAttribute("class", "survived");
      if (u[3] < 0) c.dataset.noage = "1";
      c.style.setProperty("--i", i);
      gDots.appendChild(c);
      return c;
    });

    const grouped = (defs) => {
      const groups = defs.map((d) => ({ ...d, members: [] }));
      U.forEach((u, i) => { const g = groups.find((gr) => gr.filter(u)); if (g) g.members.push(i); });
      groups.forEach((g) => g.members.sort((a, b) => U[b][0] - U[a][0]));     // survivors first
      const cols = groups.map((g) => Math.max(1, Math.ceil(g.members.length / ROWS)));
      const gap = 1.6;
      const totalCols = cols.reduce((a, b) => a + b, 0) + gap * (groups.length - 1);
      const spacing = Math.min(MAX_SPACING, (W - 30) / totalCols);
      const pos = new Array(U.length);
      let x = (W - spacing * totalCols) / 2 + spacing / 2;
      const out = [];
      groups.forEach((g, gi) => {
        g.members.forEach((idx, k) => { pos[idx] = [x + Math.floor(k / ROWS) * spacing, TOP + (k % ROWS) * spacing]; });
        const n = g.members.length;
        const s = g.members.filter((i) => U[i][0]).length;
        out.push({ label: g.label, x: x - spacing / 2, w: cols[gi] * spacing, n, rate: n ? s / n : 0 });
        x += (cols[gi] + gap) * spacing;
      });
      return { pos, groups: out, spacing };
    };

    const LAYOUTS = {
      manifest: () => {
        const spacing = MAX_SPACING, cols = 33;
        const x0 = (W - cols * spacing) / 2 + spacing / 2;
        return { pos: U.map((_, i) => [x0 + (i % cols) * spacing, TOP + Math.floor(i / cols) * spacing]), groups: [], spacing };
      },
      outcome: () => grouped([
        { label: "Survived", filter: (u) => u[0] === 1 },
        { label: "Died", filter: (u) => u[0] === 0 },
      ]),
      class: () => grouped([1, 2, 3].map((c) => ({ label: ["1st class", "2nd class", "3rd class"][c - 1], filter: (u) => u[1] === c }))),
      sex: () => grouped([
        { label: "Women", filter: (u) => u[2] === 0 },
        { label: "Men", filter: (u) => u[2] === 1 },
      ]),
      classsex: () => grouped([[1, 0, "1st, women"], [1, 1, "1st, men"], [2, 0, "2nd, women"], [2, 1, "2nd, men"], [3, 0, "3rd, women"], [3, 1, "3rd, men"]]
        .map(([c, s, label]) => ({ label, filter: (u) => u[1] === c && u[2] === s }))),
      age: () => grouped([["0-12", 0, 13], ["13-19", 13, 20], ["20-29", 20, 30], ["30-39", 30, 40], ["40-49", 40, 50], ["50-59", 50, 60], ["60+", 60, 200]]
        .map(([label, lo, hi]) => ({ label, filter: (u) => u[3] >= lo && u[3] < hi }))
        .concat([{ label: "No age", filter: (u) => u[3] < 0 }])),
    };
    const CAPTIONS = {
      manifest: "All 891 passengers in manifest order. Teal survived, ember died.",
      outcome: "Sorted by outcome: 342 survived, 549 died.",
      class: "Grouped by ticket class, survivors at the top of each column.",
      sex: "Grouped by sex: 74% of women survived, 19% of men.",
      classsex: "Class and sex together: from 97% (first-class women) to 13% (third-class men).",
      age: "Grouped by age band. Hollow dots: the 177 passengers with no recorded age.",
    };

    const text = (x, y, cls, content) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("text-anchor", "middle"); t.setAttribute("class", cls);
      t.textContent = content;
      return t;
    };

    function apply(name) {
      const L = LAYOUTS[name]();
      const r = (L.spacing * 0.36).toFixed(1);
      const dy = ((MAX_SPACING - L.spacing) * (ROWS - 1)) / 2;        // keep shorter layouts centered in the frame
      dots.forEach((c, i) => { const [x, y] = L.pos[i]; c.style.transform = `translate(${x}px, ${y + dy}px)`; c.setAttribute("r", r); });
      svg.classList.toggle("show-noage", name === "age");
      const narrow = L.groups.some((g) => g.w < 100);
      gLabels.replaceChildren(...L.groups.flatMap((g, i) => {
        const cx = g.x + g.w / 2;
        const lift = narrow && i % 2 ? 38 : 0;                       // alternate label rows when groups are narrow
        const pct = `${Math.round(g.rate * 100)}%`;
        return [
          text(cx, TOP + dy - 50 - lift, "glabel", g.label),
          text(cx, TOP + dy - 31 - lift, "gsub", narrow ? `${g.n} · ${pct}` : `${g.n} passengers · ${pct} survived`),
        ];
      }));
      $("#unit-caption").textContent = CAPTIONS[name];
    }

    const steps = $$(".step");
    const stepObs = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        steps.forEach((s) => s.classList.toggle("is-active", s === e.target));
        apply(e.target.dataset.layout);
      }
    }, { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach((s) => stepObs.observe(s));
    apply("manifest");
  }

  /* ------------------------------------------------------------ charts */
  function chartTheme() {
    return {
      ink: cssVar("--ink"), ink2: cssVar("--ink-2"), muted: cssVar("--muted"), line: cssVar("--line"),
      survived: cssVar("--survived"), died: cssVar("--died"), panel2: cssVar("--panel-2"), accent: cssVar("--accent"),
    };
  }

  function rateBar(id, groups, { unit = "passengers" } = {}) {
    const t = chartTheme();
    const labels = Object.keys(groups);
    const rates = labels.map((k) => Math.round(groups[k].rate * 1000) / 10);
    const ctx = $(`#${id}`);
    const chart = new Chart(ctx, {
      type: "bar",
      data: { labels, datasets: [{ data: rates, backgroundColor: t.survived, borderRadius: 4, barPercentage: 0.72, categoryPercentage: 0.8 }] },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: reduced ? false : { duration: 700 },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: t.panel2, titleColor: t.ink, bodyColor: t.ink2, borderColor: t.line, borderWidth: 1, padding: 10, displayColors: false,
            callbacks: { label: (c) => { const g = groups[c.label]; return [`${c.parsed.x}% survived`, `${g.survived} of ${g.n} ${unit}`]; } },
          },
        },
        scales: {
          x: { min: 0, max: 100, grid: { color: t.line }, border: { display: false }, ticks: { color: t.muted, callback: (v) => `${v}%`, font: { family: "JetBrains Mono", size: 11 } } },
          y: { grid: { display: false }, border: { display: false }, ticks: { color: t.ink2, font: { family: "Instrument Sans", size: 12 } } },
        },
      },
    });
    state.charts.push(chart);
  }

  const MODELS = [
    ["Logistic Regression", 0.819, 0.020], ["Gradient Boosting", 0.816, 0.030], ["K-Nearest Neighbors", 0.806, 0.026],
    ["Random Forest", 0.797, 0.054], ["Decision Tree", 0.751, 0.022], ["Dummy: always died", 0.617, 0.003],
  ];

  function modelsChart() {
    const t = chartTheme();
    const floorLine = {
      id: "floorLine",
      afterDatasetsDraw(chart) {
        const { ctx, scales: { x, y } } = chart;
        const px = x.getPixelForValue(0.617);
        ctx.save();
        ctx.strokeStyle = t.muted; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, y.top); ctx.lineTo(px, y.bottom); ctx.stroke();
        ctx.fillStyle = t.muted; ctx.font = "11px JetBrains Mono"; ctx.textAlign = "left";
        ctx.fillText("floor 0.617", px + 6, y.bottom - 4);
        ctx.restore();
      },
    };
    const valueLabels = {
      id: "valueLabels",
      afterDatasetsDraw(chart) {
        const { ctx } = chart;
        ctx.save(); ctx.fillStyle = t.ink; ctx.font = "600 12px JetBrains Mono"; ctx.textBaseline = "middle";
        chart.getDatasetMeta(0).data.forEach((bar, i) => { ctx.fillText(`${MODELS[i][1].toFixed(3)} ± ${MODELS[i][2].toFixed(3)}`, bar.x + 8, bar.y); });
        ctx.restore();
      },
    };
    const chart = new Chart($("#chart-models"), {
      type: "bar",
      data: { labels: MODELS.map((m) => m[0]), datasets: [{ data: MODELS.map((m) => m[1]), backgroundColor: MODELS.map((m, i) => (i === 0 ? t.survived : i === MODELS.length - 1 ? `${t.muted}66` : t.muted)), borderRadius: 4, barPercentage: 0.7 }] },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: reduced ? false : { duration: 700 },
        layout: { padding: { right: 110 } },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { min: 0.5, max: 0.9, grid: { color: t.line }, border: { display: false }, ticks: { color: t.muted, font: { family: "JetBrains Mono", size: 11 } }, title: { display: true, text: "mean 5-fold CV accuracy (712 training rows)", color: t.muted, font: { size: 11 } } },
          y: { grid: { display: false }, border: { display: false }, ticks: { color: t.ink2, font: { family: "Instrument Sans", size: 12 } } },
        },
      },
      plugins: [floorLine, valueLabels],
    });
    state.charts.push(chart);
  }

  function renderCharts() {
    if (typeof Chart === "undefined") { setTimeout(renderCharts, 150); return; }
    state.charts.forEach((c) => c.destroy());
    state.charts = [];
    Chart.defaults.font.family = "Instrument Sans";
    const a = state.agg;
    rateBar("chart-class", a.by_class);
    rateBar("chart-title", a.by_title);
    rateBar("chart-age", a.by_age_band);
    rateBar("chart-fare", a.by_fare_band);
    modelsChart();
  }

  /* ------------------------------------------------------------ predictor */
  const PRESETS = {
    owen: { Pclass: "3", Sex: "male", Title: "Mr", Age: 22, Fare: 7.25, SibSp: 1, Parch: 0, Embarked: "S", pid: 1 },
    florence: { Pclass: "1", Sex: "female", Title: "Mrs", Age: 38, Fare: 71.2833, SibSp: 1, Parch: 0, Embarked: "C", pid: 2 },
    frankie: { Pclass: "3", Sex: "male", Title: "Master", Age: 9, Fare: 20.525, SibSp: 0, Parch: 2, Embarked: "S", pid: 166 },
  };

  function featureVector(inp) {
    const M = state.model;
    const raw = { Age: inp.Age, Fare: inp.Fare, SibSp: inp.SibSp, Parch: inp.Parch, FamilySize: inp.SibSp + inp.Parch + 1 };
    const x = {};
    M.numeric.forEach((n) => { const v = Number.isFinite(raw[n]) ? raw[n] : M.medians[n]; x[n] = (v - M.means[n]) / M.scales[n]; });
    M.categorical.forEach((c) => { const val = String(inp[c] ?? M.cat_modes[c]); M.categories[c].forEach((cat) => { x[`${c}_${cat}`] = cat === val ? 1 : 0; }); });
    return M.feature_names.map((n) => x[n]);
  }
  function predict(inp) {
    const M = state.model;
    const vec = featureVector(inp);
    let z = M.intercept;
    const contrib = M.feature_names.map((name, i) => { const pull = M.coef[i] * vec[i]; z += pull; return { name, value: vec[i], weight: M.coef[i], pull }; });
    return { p: 1 / (1 + Math.exp(-z)), z, contrib };
  }

  function initPredictor() {
    const form = $("#controls");
    let titleTouched = false;
    const read = () => {
      const fd = new FormData(form);
      return { Pclass: fd.get("Pclass"), Sex: fd.get("Sex"), Title: fd.get("Title"), Embarked: fd.get("Embarked"),
        Age: parseFloat(fd.get("Age")), Fare: parseFloat(fd.get("Fare")), SibSp: parseInt(fd.get("SibSp"), 10), Parch: parseInt(fd.get("Parch"), 10) };
    };
    const setRadio = (name, value) => { const el = form.querySelector(`input[name="${name}"][value="${value}"]`); if (el) el.checked = true; };
    const suggestTitle = (inp) => (inp.Sex === "male" ? (inp.Age < 13 ? "Master" : "Mr") : (inp.Age < 18 ? "Miss" : "Mrs"));

    const update = () => {
      const inp = read();
      ["Age", "Fare", "SibSp", "Parch"].forEach((k) => { $(`#out-${k}`).textContent = k === "Fare" ? inp.Fare.toFixed(2) : String(inp[k]); });
      const { p, contrib } = predict(inp);
      const pct = p * 100;
      $("#prob").textContent = `${pct.toFixed(1)}%`;
      $("#verdict").textContent = p >= 0.5 ? "probability of survival: the model says survived" : "probability of survival: the model says died";
      const fill = $("#gauge-fill");
      fill.style.strokeDasharray = `${(p * 314.16).toFixed(1)} 314.16`;
      fill.classList.toggle("survived", p >= 0.5);
      const tbody = $("#contrib tbody");
      const maxPull = Math.max(0.01, ...contrib.map((c) => Math.abs(c.pull)));
      tbody.innerHTML = [{ name: "intercept", value: 1, weight: state.model.intercept, pull: state.model.intercept }]
        .concat(contrib.slice().sort((a, b) => Math.abs(b.pull) - Math.abs(a.pull)))
        .map((c) => {
          const w = Math.abs(c.pull) / maxPull * 50;
          const pos = c.pull >= 0;
          return `<tr><td>${c.name}</td><td class="num">${Number.isInteger(c.value) ? c.value : c.value.toFixed(2)}</td><td class="num">${c.weight.toFixed(3)}</td>` +
            `<td><div class="bar"><i class="${pos ? "pos" : ""}" style="left:${pos ? 50 : 50 - w}%;width:${w}%"></i></div></td></tr>`;
        }).join("");
      const body = { Pclass: Number(inp.Pclass), Sex: inp.Sex, Age: inp.Age, Fare: inp.Fare, SibSp: inp.SibSp, Parch: inp.Parch, Embarked: inp.Embarked };
      $("#api-json").textContent = `POST /predict\n${JSON.stringify(body, null, 2)}\n\n200 OK\n${JSON.stringify({ prediction: p >= 0.5 ? 1 : 0, probability: Math.round(p * 1000) / 1000 }, null, 2)}`;
    };

    form.addEventListener("input", (e) => {
      if (e.target.name === "Title") titleTouched = true;
      if (!titleTouched && (e.target.name === "Sex" || e.target.name === "Age")) setRadio("Title", suggestTitle(read()));
      update();
    });
    $$(".presets button").forEach((b) => b.addEventListener("click", () => {
      const pr = PRESETS[b.dataset.preset];
      ["Pclass", "Sex", "Title", "Embarked"].forEach((k) => setRadio(k, pr[k]));
      ["Age", "Fare", "SibSp", "Parch"].forEach((k) => { form.elements[k].value = pr[k]; });
      titleTouched = true;
      update();
    }));

    const M = state.model;
    $("#model-note").textContent = `Logistic Regression trained on 712 rows with the notebooks' recipe. On the 179 sealed test rows it scored ${M.test_accuracy} accuracy, F1 ${M.test_f1}, ROC AUC ${M.test_auc}. Move a control: the eighteen numbers below change with it.`;
    Object.values(PRESETS).forEach((pr) => { const el = $(`.odds[data-pid="${pr.pid}"]`); if (el) el.textContent = `model odds ${(predict(pr).p * 100).toFixed(1)}%`; });
    update();
  }

  /* ------------------------------------------------------------ course links */
  function initCourseLinks() {
    const REPO = "maglionejm/fontys-tech-exercises";
    $$(".module ol[data-module] li[data-nb]").forEach((li) => {
      const path = `${li.parentElement.dataset.module}/${li.dataset.nb}.ipynb`;
      const links = document.createElement("div");
      links.className = "links";
      links.innerHTML = `<a href="https://colab.research.google.com/github/${REPO}/blob/main/${path}" rel="noopener">Open in Colab</a>` +
        `<a href="https://github.com/${REPO}/blob/main/${path}" rel="noopener">Read on GitHub</a>`;
      li.appendChild(links);
    });
  }
})();
