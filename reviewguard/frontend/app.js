/**
 * ReviewGuard Frontend Application Logic
 * Single Page Application wiring 10 pages to FastAPI endpoints.
 */

// Global State
let currentDataset = "kaggle";
let currentExplorePage = 1;
let exploreDebounceTimer = null;
let trainPollInterval = null;
let activeEvaluationData = null;

// Chart references for clean re-renders
const charts = {
  overviewRadar: null,
  explorerClass: null,
  explorerRating: null,
  chi2: null,
  mi: null,
  evalRoc: null,
  ensembleComparison: null,
  ensembleImportance: null,
  clusterElbow: null,
  clusterPca: null,
};

// ==========================================
// ROUTER & NAVIGATION
// ==========================================

function getActivePageFromHash() {
  const hash = window.location.hash.replace("#", "").trim();
  const validPages = [
    "overview",
    "dataset-explorer",
    "preprocessing-and-features",
    "model-training",
    "model-evaluation",
    "ensemble-learning",
    "reviewer-clustering",
    "pattern-mining",
    "sequence-analysis",
    "live-review-checker",
  ];
  return validPages.includes(hash) ? hash : "overview";
}

function navigateTo(pageId) {
  window.location.hash = pageId;
}

function renderRoute() {
  const urlParams = new URLSearchParams(window.location.search);
  const ds = urlParams.get("dataset");
  if (ds && (ds === "kaggle" || ds === "sample") && currentDataset !== ds) {
    currentDataset = ds;
    const selectElem = document.getElementById("globalDatasetSelect");
    if (selectElem) selectElem.value = ds;
  }

  const activePage = getActivePageFromHash();

  // Hide all sections
  document.querySelectorAll(".page-section").forEach((sec) => {
    sec.classList.add("hidden");
  });

  // Show active section
  const targetSection = document.getElementById(`page-${activePage}`);
  if (targetSection) {
    targetSection.classList.remove("hidden");
  }

  // Update sidebar active styling
  document.querySelectorAll(".nav-item").forEach((item) => {
    if (item.getAttribute("data-page") === activePage) {
      item.classList.add("active-nav");
      item.classList.remove("text-secondary");
    } else {
      item.classList.remove("active-nav");
      item.classList.add("text-secondary");
    }
  });

  // Update top breadcrumb
  const pageTitles = {
    overview: "System Overview",
    "dataset-explorer": "Dataset Explorer",
    "preprocessing-and-features": "Preprocessing & Features",
    "model-training": "Model Training",
    "model-evaluation": "Model Evaluation",
    "ensemble-learning": "Ensemble Learning",
    "reviewer-clustering": "Reviewer Clustering",
    "pattern-mining": "Pattern Mining",
    "sequence-analysis": "Sequence Analysis (HMM)",
    "live-review-checker": "Live Review Checker",
  };
  const bc = document.getElementById("headerBreadcrumb");
  if (bc) bc.innerText = pageTitles[activePage] || "Overview";

  // Trigger page-specific data loader
  loadPageData(activePage);
}

function loadPageData(pageId) {
  switch (pageId) {
    case "overview":
      loadOverviewData();
      break;
    case "dataset-explorer":
      loadExploreData();
      break;
    case "preprocessing-and-features":
      loadFeaturesData();
      break;
    case "model-training":
      checkTrainingStatus();
      break;
    case "model-evaluation":
      loadEvaluationData();
      break;
    case "ensemble-learning":
      loadEnsembleData();
      break;
    case "reviewer-clustering":
      loadClusteringData();
      break;
    case "pattern-mining":
      loadAssociationData();
      break;
    case "sequence-analysis":
      loadHmmData();
      break;
    case "live-review-checker":
      updateLiveStats();
      break;
  }
}

// ==========================================
// GLOBAL DATASET SELECTOR
// ==========================================

function changeGlobalDataset(datasetId) {
  currentDataset = datasetId;
  const selectElem = document.getElementById("globalDatasetSelect");
  if (selectElem && selectElem.value !== datasetId) {
    selectElem.value = datasetId;
  }

  // Highlight selection cards in Dataset Explorer
  const cardKaggle = document.getElementById("card-kaggle");
  const cardSample = document.getElementById("card-sample");
  if (cardKaggle && cardSample) {
    if (datasetId === "kaggle") {
      cardKaggle.classList.add("border-2", "border-primary");
      cardSample.classList.remove("border-2", "border-primary");
    } else {
      cardSample.classList.add("border-2", "border-primary");
      cardKaggle.classList.remove("border-2", "border-primary");
    }
  }

  // Reload data for currently visible page
  const activePage = getActivePageFromHash();
  loadPageData(activePage);
}

function selectDataset(datasetId) {
  changeGlobalDataset(datasetId);
}

function switchDatasetAndReload(datasetId) {
  changeGlobalDataset(datasetId);
}

function handleGlobalSearch(e) {
  if (e.key === "Enter") {
    const val = e.target.value.trim();
    if (val) {
      navigateTo("dataset-explorer");
      const searchBox = document.getElementById("tableSearchInput");
      if (searchBox) {
        searchBox.value = val;
        loadExploreData();
      }
    }
  }
}

// ==========================================
// PAGE 1: OVERVIEW
// ==========================================

async function loadOverviewData() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();

    // Key metrics
    document.getElementById("overviewTotalReviews").innerText = Number(
      data.metrics.total_reviews_analyzed
    ).toLocaleString();
    document.getElementById("overviewFakeCount").innerText = Number(
      data.metrics.fake_reviews_detected
    ).toLocaleString();
    document.getElementById("overviewFakeRatio").innerText = `${data.metrics.fake_ratio_pct}%`;
    document.getElementById("overviewBestF1").innerText = `${data.metrics.best_f1_score}%`;
    document.getElementById("overviewBestModel").innerText = data.metrics.best_model_name;
    document.getElementById("overviewModelsCount").innerText = data.metrics.models_active_count;

    // Leaderboard table
    const tbody = document.getElementById("overviewLeaderboardBody");
    if (tbody) {
      if (data.radar_comparison && data.radar_comparison.length > 0) {
        tbody.innerHTML = data.radar_comparison
          .sort((a, b) => b.f1 - a.f1)
          .map(
            (m, i) => `
            <tr class="hover:bg-surface-container-low transition-colors">
              <td class="py-2.5 px-2 font-medium text-on-surface flex items-center gap-1.5">
                <span class="w-1.5 h-1.5 rounded-full ${i === 0 ? "bg-primary" : "bg-secondary"}"></span>
                <span>${m.model}</span>
              </td>
              <td class="py-2.5 px-2 text-right font-bold text-primary tabular-nums">${(m.f1 * 100).toFixed(1)}%</td>
              <td class="py-2.5 px-2 text-right text-secondary tabular-nums">${(m.accuracy * 100).toFixed(1)}%</td>
              <td class="py-2.5 px-2 text-right text-secondary tabular-nums">${(m.auc * 100).toFixed(1)}%</td>
              <td class="py-2.5 px-2 text-center">
                <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${
                  i === 0
                    ? "bg-primary-fixed text-on-primary-fixed-variant"
                    : "bg-surface-container text-secondary"
                }">${i === 0 ? "Top Model" : "Active"}</span>
              </td>
            </tr>`
          )
          .join("");
      } else {
        tbody.innerHTML = `<tr><td colspan="5" class="py-4 text-center text-secondary">No models trained yet. Visit Model Training.</td></tr>`;
      }
    }

    // Recent detections
    const recBody = document.getElementById("overviewRecentDetections");
    if (recBody) {
      if (data.recent_detections && data.recent_detections.length > 0) {
        recBody.innerHTML = data.recent_detections
          .map((d) => {
            const isFake = d.verdict === "Fake";
            const isSuspicious = d.verdict === "Suspicious";
            const badgeClass = isFake
              ? "bg-error-container text-on-error-container font-bold"
              : isSuspicious
              ? "bg-amber-100 text-amber-800 font-bold"
              : "bg-emerald-100 text-emerald-800 font-bold";
            return `
            <tr class="hover:bg-surface-container-low transition-colors">
              <td class="py-2.5 px-3 text-secondary text-[11px] font-mono">${d.timestamp ? d.timestamp.split("T")[1]?.slice(0, 8) || "Recent" : "Recent"}</td>
              <td class="py-2.5 px-3 text-on-surface truncate max-w-xs">${d.text_snippet}</td>
              <td class="py-2.5 px-3 text-center">
                <span class="px-2 py-0.5 rounded text-[10px] ${badgeClass}">${d.verdict}</span>
              </td>
              <td class="py-2.5 px-3 text-right font-semibold tabular-nums ${isFake ? "text-error" : "text-emerald-700"}">
                ${(d.fake_probability * 100).toFixed(1)}%
              </td>
            </tr>`;
          })
          .join("");
      } else {
        recBody.innerHTML = `<tr><td colspan="4" class="py-4 text-center text-secondary">No inferences recorded yet. Try the Live Review Checker.</td></tr>`;
      }
    }

    // Radar Chart
    renderOverviewRadar(data.radar_comparison);
  } catch (err) {
    console.error("Overview error:", err);
  }
}

function renderOverviewRadar(models) {
  const ctx = document.getElementById("overviewRadarChart");
  if (!ctx || !models || models.length === 0) return;

  if (charts.overviewRadar) charts.overviewRadar.destroy();

  const labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"];
  const colors = [
    { bg: "rgba(79, 70, 229, 0.2)", border: "#4F46E5" }, // Primary Indigo
    { bg: "rgba(16, 185, 129, 0.2)", border: "#10B981" }, // Emerald
    { bg: "rgba(245, 158, 11, 0.2)", border: "#F59E0B" }, // Amber
    { bg: "rgba(239, 68, 68, 0.2)", border: "#EF4444" },  // Red
    { bg: "rgba(139, 92, 246, 0.2)", border: "#8B5CF6" }, // Purple
  ];

  const datasets = models.slice(0, 4).map((m, idx) => {
    const c = colors[idx % colors.length];
    return {
      label: m.model,
      data: [m.accuracy, m.precision, m.recall, m.f1, m.auc].map((v) =>
        Math.round(v * 100)
      ),
      backgroundColor: c.bg,
      borderColor: c.border,
      borderWidth: 2,
      pointRadius: 3,
    };
  });

  charts.overviewRadar = new Chart(ctx, {
    type: "radar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 40,
          max: 100,
          ticks: { stepSize: 20, font: { size: 9, family: "Inter" } },
          pointLabels: { font: { size: 11, family: "Inter", weight: "500" } },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: { font: { size: 10, family: "Inter" }, boxWidth: 10 },
        },
      },
    },
  });
}

// ==========================================
// PAGE 2: DATASET EXPLORER
// ==========================================

function debounceExploreTable() {
  clearTimeout(exploreDebounceTimer);
  exploreDebounceTimer = setTimeout(() => {
    currentExplorePage = 1;
    loadExploreData();
  }, 300);
}

function changeExplorePage(delta) {
  currentExplorePage = Math.max(1, currentExplorePage + delta);
  loadExploreData();
}

async function loadExploreData() {
  try {
    const searchVal = document.getElementById("tableSearchInput")?.value || "";
    const catVal = document.getElementById("tableCategorySelect")?.value || "all";
    const labelVal = document.getElementById("tableLabelSelect")?.value || "all";

    const params = new URLSearchParams({
      dataset: currentDataset,
      page: currentExplorePage,
      page_size: 15,
    });
    if (searchVal) params.append("search", searchVal);
    if (catVal && catVal !== "all") params.append("category", catVal);
    if (labelVal && labelVal !== "all") params.append("label_filter", labelVal);

    const res = await fetch(`/api/dataset/explore?${params.toString()}`);
    const data = await res.json();

    // Stats
    document.getElementById("statMeanWords").innerText = data.word_stats.mean;
    document.getElementById("statMedianWords").innerText = data.word_stats.median;
    document.getElementById("statMaxWords").innerText = Number(data.word_stats.max).toLocaleString();

    // Populate category dropdown if needed
    const catSelect = document.getElementById("tableCategorySelect");
    if (catSelect && catSelect.children.length <= 1 && data.categories) {
      data.categories.forEach((cat) => {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.innerText = cat;
        catSelect.appendChild(opt);
      });
    }

    // Class balance chart
    renderExplorerClassChart(data.class_balance);

    // Rating distribution chart
    renderExplorerRatingChart(data.rating_distribution);

    // Populate table
    document.getElementById("tableRecordCount").innerText = `${Number(data.pagination.total_filtered).toLocaleString()} reviews`;
    document.getElementById("currentPageDisplay").innerText = `Page ${data.pagination.page} of ${data.pagination.total_pages}`;
    document.getElementById("paginationInfo").innerText = `Showing page ${data.pagination.page} (${data.records.length} items)`;

    document.getElementById("btnPrevPage").disabled = data.pagination.page <= 1;
    document.getElementById("btnNextPage").disabled = data.pagination.page >= data.pagination.total_pages;

    const tbody = document.getElementById("exploreTableBody");
    if (tbody) {
      tbody.innerHTML = data.records
        .map((r) => {
          const isFake = r.label_int === 1;
          const stars = "★".repeat(Math.round(r.rating)) + "☆".repeat(5 - Math.round(r.rating));
          const metaBadge = r.user_id
            ? `<span class="text-[10px] bg-surface-container px-1.5 py-0.5 rounded text-secondary">${r.user_id} • Age ${r.account_age_days}d</span>`
            : `<span class="text-[10px] text-zinc-400">No metadata</span>`;

          return `
          <tr class="hover:bg-surface-container-low transition-colors">
            <td class="py-2.5 px-3 font-mono text-secondary">${r.review_id}</td>
            <td class="py-2.5 px-3 font-medium text-on-surface truncate max-w-[120px]">${r.category}</td>
            <td class="py-2.5 px-3 text-amber-500 font-semibold tracking-tighter">${stars} (${r.rating})</td>
            <td class="py-2.5 px-3 text-on-surface text-xs leading-relaxed max-w-sm truncate" title="${r.text}">
              ${r.snippet}
            </td>
            <td class="py-2.5 px-3 text-center">
              <span class="px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                isFake ? "bg-error-container text-on-error-container" : "bg-emerald-100 text-emerald-800"
              }">${r.label}</span>
            </td>
            <td class="py-2.5 px-3 text-center">${metaBadge}</td>
          </tr>`;
        })
        .join("");
    }
  } catch (err) {
    console.error("Explore error:", err);
  }
}

function renderExplorerClassChart(balance) {
  const ctx = document.getElementById("explorerClassChart");
  if (!ctx || !balance) return;

  if (charts.explorerClass) charts.explorerClass.destroy();

  charts.explorerClass = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Genuine (OR)", "Fake (CG)"],
      datasets: [
        {
          data: [balance.genuine_or, balance.fake_cg],
          backgroundColor: ["#10B981", "#EF4444"],
          borderWidth: 2,
          borderColor: "#ffffff",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { font: { size: 10, family: "Inter" } } },
      },
      cutout: "68%",
    },
  });

  const statsEl = document.getElementById("explorerClassStats");
  if (statsEl) {
    statsEl.innerHTML = `
      <div class="text-center"><span class="text-emerald-700 font-bold">${Number(balance.genuine_or).toLocaleString()}</span> <span class="text-secondary text-[10px]">(${balance.genuine_pct}%) Genuine</span></div>
      <div class="text-center"><span class="text-error font-bold">${Number(balance.fake_cg).toLocaleString()}</span> <span class="text-secondary text-[10px]">(${balance.fake_pct}%) Fake</span></div>
    `;
  }
}

function renderExplorerRatingChart(ratingDist) {
  const ctx = document.getElementById("explorerRatingChart");
  if (!ctx || !ratingDist) return;

  if (charts.explorerRating) charts.explorerRating.destroy();

  const labels = ratingDist.map((d) => `${d.rating}★`);
  const values = ratingDist.map((d) => d.count);

  charts.explorerRating = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Reviews Count",
          data: values,
          backgroundColor: "#4F46E5",
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 10, family: "Inter" } } },
        y: { ticks: { font: { size: 9, family: "Inter" } } },
      },
    },
  });
}

async function handleDatasetUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/dataset/upload", {
      method: "POST",
      body: formData,
    });
    const result = await res.json();
    alert(`Upload successful! ${result.filename} (${result.rows} rows).`);
    currentDataset = result.filename;
    loadExploreData();
  } catch (err) {
    alert("Upload failed: " + err.message);
  }
}

// ==========================================
// PAGE 3: PREPROCESSING & FEATURES
// ==========================================

async function loadFeaturesData() {
  try {
    const res = await fetch(`/api/features/importance?dataset=${currentDataset}&top_n=12`);
    const data = await res.json();

    renderChi2Chart(data.chi_square_features);
    renderMiChart(data.mutual_information_features);

    // Initial simulation of imbalance
    testImbalanceHandling();
  } catch (err) {
    console.error("Features error:", err);
  }
}

function renderChi2Chart(features) {
  const ctx = document.getElementById("chi2Chart");
  if (!ctx || !features) return;

  if (charts.chi2) charts.chi2.destroy();

  const labels = features.map((f) => f.feature);
  const scores = features.map((f) => f.score);

  charts.chi2 = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Chi-Square (χ²) Score",
          data: scores,
          backgroundColor: "#4F46E5",
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 9, family: "Inter" } } },
        y: { ticks: { font: { size: 10, family: "Inter" } } },
      },
    },
  });
}

function renderMiChart(features) {
  const ctx = document.getElementById("miChart");
  if (!ctx || !features) return;

  if (charts.mi) charts.mi.destroy();

  const labels = features.map((f) => f.feature);
  const scores = features.map((f) => f.score);

  charts.mi = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Mutual Information",
          data: scores,
          backgroundColor: "#059669",
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 9, family: "Inter" } } },
        y: { ticks: { font: { size: 10, family: "Inter" } } },
      },
    },
  });
}

async function testImbalanceHandling() {
  const method = document.getElementById("imbalanceMethodSelect")?.value || "smote";
  try {
    const res = await fetch("/api/imbalance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_name: currentDataset, method }),
    });
    const data = await res.json();

    document.getElementById("imbBeforeGenuine").innerText = `${data.before.genuine_0} (${(
      (data.before.genuine_0 / data.before.total) *
      100
    ).toFixed(1)}%)`;
    document.getElementById("imbBeforeFake").innerText = `${data.before.fake_1} (${(
      (data.before.fake_1 / data.before.total) *
      100
    ).toFixed(1)}%)`;

    document.getElementById("imbAfterGenuine").innerText = `${data.after.genuine_0} (${(
      (data.after.genuine_0 / data.after.total) *
      100
    ).toFixed(1)}%)`;
    document.getElementById("imbAfterFake").innerText = `${data.after.fake_1} (${(
      (data.after.fake_1 / data.after.total) *
      100
    ).toFixed(1)}%)`;

    const beforeGenPct = (data.before.genuine_0 / data.before.total) * 100;
    const afterGenPct = (data.after.genuine_0 / data.after.total) * 100;

    document.getElementById("barBeforeGenuine").style.width = `${beforeGenPct}%`;
    document.getElementById("barBeforeFake").style.width = `${100 - beforeGenPct}%`;
    document.getElementById("barAfterGenuine").style.width = `${afterGenPct}%`;
    document.getElementById("barAfterFake").style.width = `${100 - afterGenPct}%`;
  } catch (err) {
    console.error("Imbalance error:", err);
  }
}

// ==========================================
// PAGE 4: MODEL TRAINING
// ==========================================

async function startTrainingPipeline() {
  const dataset = document.getElementById("trainDatasetSelect")?.value || currentDataset;
  const cvFolds = parseInt(document.getElementById("trainCvSelect")?.value || "5", 10);
  const imbalance = document.getElementById("trainImbalanceSelect")?.value || "none";
  const samples = parseInt(document.getElementById("trainSampleSlider")?.value || "6000", 10);

  const checkboxes = document.querySelectorAll('input[name="trainModel"]:checked');
  const selectedModels = Array.from(checkboxes).map((cb) => cb.value);

  if (selectedModels.length === 0) {
    alert("Please select at least one classical model to train.");
    return;
  }

  const btn = document.getElementById("btnStartTraining");
  btn.disabled = true;
  btn.innerHTML = `<span class="material-symbols-outlined animate-spin text-[18px]">progress_activity</span> Launching Pipeline...`;

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_name: dataset,
        models: selectedModels,
        cv_folds: cvFolds,
        imbalance_method: imbalance,
        max_samples: samples,
      }),
    });
    const data = await res.json();

    // Start status polling
    pollTrainingStatus();
  } catch (err) {
    alert("Training trigger failed: " + err.message);
    btn.disabled = false;
    btn.innerHTML = `<span class="material-symbols-outlined text-[18px]">play_circle</span> Launch Training Pipeline`;
  }
}

function pollTrainingStatus() {
  clearInterval(trainPollInterval);
  trainPollInterval = setInterval(async () => {
    await checkTrainingStatus();
  }, 1200);
}

async function checkTrainingStatus() {
  try {
    const res = await fetch("/api/train/status");
    const data = await res.json();

    document.getElementById("trainStatusMessage").innerText = data.message;
    document.getElementById("trainProgressPercent").innerText = `${data.progress}%`;
    document.getElementById("trainProgressBar").style.width = `${data.progress}%`;

    const badge = document.getElementById("trainStatusBadge");
    if (data.is_running) {
      badge.innerText = "RUNNING";
      badge.className = "text-xs font-bold px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800 animate-pulse";
    } else if (data.status === "completed") {
      badge.innerText = "COMPLETED";
      badge.className = "text-xs font-bold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800";
      clearInterval(trainPollInterval);

      const btn = document.getElementById("btnStartTraining");
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined text-[18px]">play_circle</span> Launch Training Pipeline`;
      }
    } else if (data.status === "error") {
      badge.innerText = "ERROR";
      badge.className = "text-xs font-bold px-2.5 py-0.5 rounded-full bg-error-container text-on-error-container";
      clearInterval(trainPollInterval);

      const btn = document.getElementById("btnStartTraining");
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined text-[18px]">play_circle</span> Launch Training Pipeline`;
      }
    }

    // Update terminal output
    const terminal = document.getElementById("trainTerminal");
    if (terminal && data.logs) {
      terminal.innerHTML = data.logs
        .map((l) => `<div><span class="text-zinc-500">[${l.time}]</span> ${l.message}</div>`)
        .join("");
      terminal.scrollTop = terminal.scrollHeight;
    }
  } catch (err) {
    console.error("Train status error:", err);
  }
}

// ==========================================
// PAGE 5: MODEL EVALUATION
// ==========================================

async function loadEvaluationData() {
  try {
    const res = await fetch("/api/evaluation");
    const data = await res.json();
    activeEvaluationData = data;

    if (!data.is_trained) {
      document.getElementById("evaluationTableBody").innerHTML = `
        <tr><td colspan="8" class="p-6 text-center text-secondary">
          Models have not been trained yet. Please visit <a href="#model-training" class="text-primary font-semibold underline">Model Training</a> to run the pipeline.
        </td></tr>`;
      return;
    }

    // Populate leaderboard table
    const tbody = document.getElementById("evaluationTableBody");
    tbody.innerHTML = data.ranked_models
      .map((m, idx) => {
        const isBest = idx === 0;
        return `
        <tr class="hover:bg-surface-container-low transition-colors ${isBest ? "bg-primary/5" : ""}">
          <td class="py-3 px-3 font-semibold text-on-surface flex items-center gap-2">
            ${isBest ? '<span class="material-symbols-outlined text-[16px] text-primary">verified</span>' : '<span class="w-2 h-2 rounded-full bg-secondary"></span>'}
            <span>${m.name}</span>
          </td>
          <td class="py-3 px-3 text-right tabular-nums text-secondary">${(m.accuracy * 100).toFixed(1)}%</td>
          <td class="py-3 px-3 text-right tabular-nums text-secondary">${(m.precision * 100).toFixed(1)}%</td>
          <td class="py-3 px-3 text-right tabular-nums text-secondary">${(m.recall * 100).toFixed(1)}%</td>
          <td class="py-3 px-3 text-right tabular-nums font-bold text-primary text-sm">${(m.f1 * 100).toFixed(1)}%</td>
          <td class="py-3 px-3 text-right tabular-nums text-secondary">${(m.auc * 100).toFixed(1)}%</td>
          <td class="py-3 px-3 text-right tabular-nums text-secondary font-mono">${(m.cv_mean * 100).toFixed(1)}% ± ${(m.cv_std * 100).toFixed(1)}%</td>
          <td class="py-3 px-3 text-center">
            <button onclick="selectConfusionMatrixModel('${m.model_key}')" class="px-2 py-1 rounded card-border text-[11px] font-medium text-secondary hover:text-on-surface hover:bg-surface-container transition-colors">
              Inspect Matrix
            </button>
          </td>
        </tr>`;
      })
      .join("");

    // Populate confusion matrix model select
    const cmSelect = document.getElementById("cmModelSelect");
    cmSelect.innerHTML = data.ranked_models
      .map((m) => `<option value="${m.model_key}">${m.name}</option>`)
      .join("");

    renderActiveConfusionMatrix();
    renderEvalRocCurves(data.models);
  } catch (err) {
    console.error("Evaluation error:", err);
  }
}

function selectConfusionMatrixModel(modelKey) {
  const cmSelect = document.getElementById("cmModelSelect");
  if (cmSelect) cmSelect.value = modelKey;
  renderActiveConfusionMatrix();
}

function renderActiveConfusionMatrix() {
  if (!activeEvaluationData || !activeEvaluationData.models) return;
  const cmSelect = document.getElementById("cmModelSelect");
  const modelKey = cmSelect ? cmSelect.value : "svm";

  const modelMetrics = activeEvaluationData.models[modelKey];
  if (!modelMetrics || !modelMetrics.confusion_matrix) return;

  const cm = modelMetrics.confusion_matrix;
  document.getElementById("cmTN").innerText = Number(cm.tn).toLocaleString();
  document.getElementById("cmFP").innerText = Number(cm.fp).toLocaleString();
  document.getElementById("cmFN").innerText = Number(cm.fn).toLocaleString();
  document.getElementById("cmTP").innerText = Number(cm.tp).toLocaleString();

  const spec = cm.tn / (cm.tn + cm.fp || 1);
  const sens = cm.tp / (cm.tp + cm.fn || 1);
  document.getElementById("cmSpecificity").innerText = `${(spec * 100).toFixed(1)}%`;
  document.getElementById("cmSensitivity").innerText = `${(sens * 100).toFixed(1)}%`;
}

function renderEvalRocCurves(models) {
  const ctx = document.getElementById("evalRocChart");
  if (!ctx || !models) return;

  if (charts.evalRoc) charts.evalRoc.destroy();

  const colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"];
  const datasets = Object.keys(models).map((key, i) => {
    const m = models[key];
    const pts = m.roc_curve || [{ x: 0, y: 0 }, { x: 1, y: 1 }];
    return {
      label: `${m.display_name || key} (AUC: ${(m.auc * 100).toFixed(1)}%)`,
      data: pts,
      borderColor: colors[i % colors.length],
      borderWidth: 2,
      fill: false,
      pointRadius: 0,
      tension: 0.2,
    };
  });

  // Reference chance line
  datasets.push({
    label: "Random Chance (AUC: 50.0%)",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: "#9CA3AF",
    borderDash: [4, 4],
    borderWidth: 1.5,
    pointRadius: 0,
    fill: false,
  });

  charts.evalRoc = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { font: { size: 10, family: "Inter" } } },
      },
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: 1,
          title: { display: true, text: "False Positive Rate (1 - Specificity)", font: { size: 10 } },
        },
        y: {
          min: 0,
          max: 1,
          title: { display: true, text: "True Positive Rate (Sensitivity)", font: { size: 10 } },
        },
      },
    },
  });
}

// ==========================================
// PAGE 6: ENSEMBLE LEARNING
// ==========================================

async function loadEnsembleData() {
  try {
    const res = await fetch("/api/ensemble");
    const data = await res.json();

    if (!data.is_trained) return;

    if (data.ensembles.random_forest) {
      document.getElementById("ensRfF1").innerText = `${(data.ensembles.random_forest.f1 * 100).toFixed(1)}%`;
    }
    if (data.ensembles.gradient_boosting) {
      document.getElementById("ensGbF1").innerText = `${(data.ensembles.gradient_boosting.f1 * 100).toFixed(1)}%`;
    }
    if (data.ensembles.soft_voting) {
      document.getElementById("ensVoteF1").innerText = `${(data.ensembles.soft_voting.f1 * 100).toFixed(1)}%`;
    }

    renderEnsembleComparisonChart(data.comparison);
    renderEnsembleImportanceChart(data.feature_importances);
  } catch (err) {
    console.error("Ensemble error:", err);
  }
}

function renderEnsembleComparisonChart(comp) {
  const ctx = document.getElementById("ensembleComparisonChart");
  if (!ctx || !comp) return;

  if (charts.ensembleComparison) charts.ensembleComparison.destroy();

  const labels = comp.map((c) => c.model);
  const values = comp.map((c) => Math.round(c.f1 * 100));
  const bgColors = comp.map((c) => (c.type === "Single Model" ? "#94A3B8" : "#4F46E5"));

  charts.ensembleComparison = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "F1 Score (%)",
          data: values,
          backgroundColor: bgColors,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 50, max: 100, ticks: { font: { size: 9, family: "Inter" } } },
        x: { ticks: { font: { size: 9, family: "Inter" } } },
      },
    },
  });
}

function renderEnsembleImportanceChart(features) {
  const ctx = document.getElementById("ensembleImportanceChart");
  if (!ctx || !features) return;

  if (charts.ensembleImportance) charts.ensembleImportance.destroy();

  const top12 = features.slice(0, 12);
  const labels = top12.map((f) => f.feature);
  const scores = top12.map((f) => f.importance);

  charts.ensembleImportance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Gini Importance",
          data: scores,
          backgroundColor: "#4F46E5",
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 9, family: "Inter" } } },
        y: { ticks: { font: { size: 10, family: "Inter" } } },
      },
    },
  });
}

// ==========================================
// PAGE 7: REVIEWER CLUSTERING
// ==========================================

async function loadClusteringData() {
  const linkage = document.getElementById("clusterLinkageSelect")?.value || "ward";
  try {
    const res = await fetch(`/api/clustering?dataset=${currentDataset}&n_clusters=4&linkage=${linkage}`);
    const data = await res.json();

    const noticeEl = document.getElementById("clusteringMetadataNotice");
    if (data.requires_metadata) {
      if (noticeEl) noticeEl.classList.remove("hidden");
      return;
    } else {
      if (noticeEl) noticeEl.classList.add("hidden");
    }

    renderClusterElbowChart(data.elbow_curve);
    renderClusterPcaChart(data.pca_points);

    // Summary table
    const tbody = document.getElementById("clusterSummaryBody");
    if (tbody && data.cluster_summaries) {
      tbody.innerHTML = data.cluster_summaries
        .map((c) => {
          const badgeClass =
            c.severity === "high"
              ? "bg-error-container text-on-error-container"
              : c.severity === "medium"
              ? "bg-amber-100 text-amber-800"
              : "bg-emerald-100 text-emerald-800";
          return `
          <tr class="hover:bg-surface-container-low transition-colors">
            <td class="py-2.5 px-3 font-bold text-primary">Cluster ${c.cluster_id}</td>
            <td class="py-2.5 px-3 font-medium">${c.size} users <span class="text-secondary text-[11px]">(${c.percentage}%)</span></td>
            <td class="py-2.5 px-3 text-right tabular-nums text-amber-600 font-semibold">${c.avg_rating}★</td>
            <td class="py-2.5 px-3 text-right tabular-nums">${c.reviews_per_day}</td>
            <td class="py-2.5 px-3 text-right tabular-nums">${(c.verified_ratio * 100).toFixed(0)}%</td>
            <td class="py-2.5 px-3 text-right tabular-nums">${c.avg_length} chars</td>
            <td class="py-2.5 px-3 text-center">
              <span class="px-2 py-0.5 rounded-full text-[10px] font-semibold ${badgeClass}">
                ${c.suspicious_tag}
              </span>
            </td>
          </tr>`;
        })
        .join("");
    }

    // MST edges
    const mstContainer = document.getElementById("mstEdgesContainer");
    if (mstContainer && data.mst_edges) {
      mstContainer.innerHTML = data.mst_edges
        .map(
          (e) => `
        <div class="px-2.5 py-1 rounded-lg card-border bg-surface-container-low flex items-center gap-1.5 font-mono text-[11px]">
          <span>Cluster ${e.source} ↔ Cluster ${e.target}</span>
          <span class="text-secondary">(d = ${e.weight})</span>
          ${e.is_anomaly_cut ? '<span class="text-[9px] bg-error text-white font-bold px-1.5 py-0.2 rounded">CUT</span>' : ""}
        </div>`
        )
        .join("");
    }
  } catch (err) {
    console.error("Clustering error:", err);
  }
}

function renderClusterElbowChart(elbow) {
  const ctx = document.getElementById("clusterElbowChart");
  if (!ctx || !elbow) return;

  if (charts.clusterElbow) charts.clusterElbow.destroy();

  const labels = elbow.map((e) => `k=${e.k}`);
  const inertias = elbow.map((e) => e.inertia);

  charts.clusterElbow = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Inertia (Within-Cluster Sum of Squares)",
          data: inertias,
          borderColor: "#4F46E5",
          backgroundColor: "#4F46E5",
          borderWidth: 2,
          pointRadius: 4,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { font: { size: 10, family: "Inter" } } },
        y: { ticks: { font: { size: 9, family: "Inter" } } },
      },
    },
  });
}

function renderClusterPcaChart(points) {
  const ctx = document.getElementById("clusterPcaChart");
  if (!ctx || !points) return;

  if (charts.clusterPca) charts.clusterPca.destroy();

  const colors = ["#4F46E5", "#10B981", "#EF4444", "#F59E0B"];
  const clusterGroups = {};

  points.forEach((p) => {
    if (!clusterGroups[p.cluster]) clusterGroups[p.cluster] = [];
    clusterGroups[p.cluster].push({ x: p.x, y: p.y, user: p.user_id, rating: p.avg_rating });
  });

  const datasets = Object.keys(clusterGroups).map((cId) => ({
    label: `Cluster ${cId}`,
    data: clusterGroups[cId],
    backgroundColor: colors[parseInt(cId, 10) % colors.length],
    pointRadius: 4,
  }));

  charts.clusterPca = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { font: { size: 10, family: "Inter" } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `User: ${ctx.raw.user} (${ctx.raw.rating}★)`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "PCA Component 1", font: { size: 10 } } },
        y: { title: { display: true, text: "PCA Component 2", font: { size: 10 } } },
      },
    },
  });
}

// ==========================================
// PAGE 8: PATTERN MINING
// ==========================================

async function loadAssociationData() {
  const algo = document.getElementById("assocAlgoSelect")?.value || "fpgrowth";
  try {
    const res = await fetch(`/api/association?dataset=${currentDataset}&algorithm=${algo}&min_support=0.05&min_confidence=0.4`);
    const data = await res.json();

    const noticeEl = document.getElementById("assocMetadataNotice");
    if (data.requires_metadata) {
      if (noticeEl) noticeEl.classList.remove("hidden");
      return;
    } else {
      if (noticeEl) noticeEl.classList.add("hidden");
    }

    document.getElementById("assocTotalRules").innerText = data.total_rules || 0;

    const tbody = document.getElementById("assocRulesBody");
    if (tbody && data.rules) {
      tbody.innerHTML = data.rules
        .map((r) => {
          const isFake = r.is_fake_target;
          return `
          <tr class="hover:bg-surface-container-low transition-colors">
            <td class="py-2.5 px-3 font-mono text-xs text-on-surface">
              ${r.antecedents.map((a) => `<span class="bg-surface-container px-1.5 py-0.5 rounded mr-1">${a}</span>`).join("")}
            </td>
            <td class="py-2.5 px-3 font-mono text-xs">
              ${r.consequents.map((c) => `<span class="bg-primary-fixed text-primary px-1.5 py-0.5 rounded font-bold">${c}</span>`).join("")}
            </td>
            <td class="py-2.5 px-3 text-right tabular-nums text-secondary">${(r.support * 100).toFixed(1)}%</td>
            <td class="py-2.5 px-3 text-right tabular-nums font-semibold">${(r.confidence * 100).toFixed(1)}%</td>
            <td class="py-2.5 px-3 text-right tabular-nums font-bold text-primary">${r.lift}x</td>
            <td class="py-2.5 px-3 text-center">
              <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${
                isFake ? "bg-error-container text-on-error-container" : "bg-emerald-100 text-emerald-800"
              }">
                ${isFake ? "Fake Marker" : "Organic Marker"}
              </span>
            </td>
          </tr>`;
        })
        .join("");
    }
  } catch (err) {
    console.error("Association error:", err);
  }
}

// ==========================================
// PAGE 9: SEQUENCE ANALYSIS (HMM)
// ==========================================

async function loadHmmData() {
  try {
    const res = await fetch(`/api/hmm/summary?dataset=${currentDataset}`);
    const data = await res.json();

    const noticeEl = document.getElementById("hmmMetadataNotice");
    if (data.requires_metadata) {
      if (noticeEl) noticeEl.classList.remove("hidden");
      return;
    } else {
      if (noticeEl) noticeEl.classList.add("hidden");
    }

    // Display transition matrix
    if (data.transition_matrix) {
      const tm = data.transition_matrix;
      document.getElementById("hmmTransMatDisplay").innerHTML = `
        P(0→0)=${tm[0][0]}  P(0→1)=${tm[0][1]}<br/>
        P(1→0)=${tm[1][0]}  P(1→1)=${tm[1][1]}
      `;
    }

    // Populate user selector
    const userSelect = document.getElementById("hmmUserSelect");
    if (userSelect && data.user_summaries) {
      userSelect.innerHTML = data.user_summaries
        .map((u) => `<option value="${u.user_id}">${u.user_id} (${u.review_count} reviews, ${u.avg_rating}★, Risk: ${(u.spam_risk_score * 100).toFixed(0)}%)</option>`)
        .join("");

      if (data.user_summaries.length > 0) {
        loadUserHmmTimeline(data.user_summaries[0].user_id);
      }
    }
  } catch (err) {
    console.error("HMM error:", err);
  }
}

async function loadUserHmmTimeline(userId) {
  try {
    const res = await fetch(`/api/hmm/${userId}?dataset=${currentDataset}`);
    const data = await res.json();

    document.getElementById("hmmUserVerdict").innerText = data.classification;
    document.getElementById("hmmUserLikelihood").innerText = data.overall_log_likelihood;
    document.getElementById("hmmUserRisk").innerText = `${(data.spam_risk_score * 100).toFixed(1)}%`;

    const tbody = document.getElementById("hmmTimelineBody");
    if (tbody && data.timeline) {
      tbody.innerHTML = data.timeline
        .map((step) => {
          const isSpam = step.viterbi_state === 1;
          const stars = "★".repeat(Math.round(step.rating));
          return `
          <tr class="hover:bg-surface-container-low transition-colors">
            <td class="py-2.5 px-3 font-mono font-bold text-secondary">#${step.step}</td>
            <td class="py-2.5 px-3 text-secondary text-xs">${step.date}</td>
            <td class="py-2.5 px-3 font-semibold text-amber-500">${stars} (${step.rating})</td>
            <td class="py-2.5 px-3 text-right tabular-nums text-secondary">${step.time_gap_days}d</td>
            <td class="py-2.5 px-3 text-xs text-on-surface">${step.observation_desc}</td>
            <td class="py-2.5 px-3 text-center">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold ${
                isSpam ? "bg-error-container text-on-error-container" : "bg-emerald-100 text-emerald-800"
              }">
                ${step.viterbi_state_name}
              </span>
            </td>
            <td class="py-2.5 px-3 text-secondary text-xs truncate max-w-xs" title="${step.review_snippet}">
              ${step.review_snippet}
            </td>
          </tr>`;
        })
        .join("");
    }
  } catch (err) {
    console.error("User HMM error:", err);
  }
}

// ==========================================
// PAGE 10: LIVE REVIEW CHECKER
// ==========================================

function handleLiveReviewInput() {
  updateLiveStats();
  resetLiveResult();
}

function clearLiveForm() {
  document.getElementById("liveReviewInput").value = "";
  updateLiveStats();
  resetLiveResult();
}

function resetLiveResult() {
  const verdictEl = document.getElementById("liveVerdictText");
  const subEl = document.getElementById("liveVerdictSub");
  const fakeProbEl = document.getElementById("liveFakeProbability");
  const inferenceTimeEl = document.getElementById("liveInferenceTime");
  const fakeBar = document.getElementById("liveProbBarFake");
  const genuineBar = document.getElementById("liveProbBarGenuine");
  const perModelEl = document.getElementById("livePerModelContainer");
  const highlightsEl = document.getElementById("liveHighlightsContainer");

  if (verdictEl) {
    verdictEl.innerText = "—";
    verdictEl.className = "text-3xl font-extrabold tracking-tight text-secondary";
  }
  if (subEl) subEl.innerText = "Enter a review and analyze it to see the result.";
  if (fakeProbEl) fakeProbEl.innerText = "—";
  if (inferenceTimeEl) inferenceTimeEl.innerText = "Not analyzed";
  if (fakeBar) fakeBar.style.width = "0%";
  if (genuineBar) genuineBar.style.width = "0%";
  if (perModelEl) perModelEl.innerHTML = "";
  if (highlightsEl) {
    highlightsEl.innerText = "Enter or load a review and click Analyze to view token-level explainability highlights.";
  }
}

function updateLiveStats() {
  const text = document.getElementById("liveReviewInput")?.value || "";
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const statsEl = document.getElementById("liveCharStats");
  if (statsEl) {
    statsEl.innerText = `${words} words | ${chars} characters`;
  }
}

async function executeLiveInference() {
  const text = document.getElementById("liveReviewInput")?.value || "";
  if (!text.trim()) {
    alert("Please enter or paste review text to analyze.");
    return;
  }

  const rating = parseFloat(document.getElementById("liveRatingSelect")?.value || "5.0");
  const age = parseFloat(document.getElementById("liveAccountAgeInput")?.value || "365");
  const verified = document.getElementById("liveVerifiedSelect")?.value === "true";

  const btn = document.getElementById("btnLiveAnalyze");
  btn.disabled = true;
  btn.innerHTML = `<span class="material-symbols-outlined animate-spin text-[18px]">progress_activity</span> Running Multi-Model Inference...`;

  const startTime = performance.now();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        rating,
        account_age_days: age,
        verified_purchase: verified,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || data.message || `Request failed (${res.status})`);
    }
    const latency = Math.round(performance.now() - startTime);

    document.getElementById("liveInferenceTime").innerText = `Latency ~${latency}ms`;

    // Verdict Banner
    const verdictEl = document.getElementById("liveVerdictText");
    const subEl = document.getElementById("liveVerdictSub");
    const fakeProbEl = document.getElementById("liveFakeProbability");

    verdictEl.innerText = data.verdict;
    fakeProbEl.innerText = `${(data.fake_probability * 100).toFixed(1)}%`;

    if (data.verdict === "Fake") {
      verdictEl.className = "text-3xl font-extrabold tracking-tight text-error";
      subEl.innerText = "High probability artificial / promotional text";
    } else if (data.verdict === "Suspicious") {
      verdictEl.className = "text-3xl font-extrabold tracking-tight text-amber-600";
      subEl.innerText = "Borderline anomaly: mixed signals detected";
    } else {
      verdictEl.className = "text-3xl font-extrabold tracking-tight text-emerald-600";
      subEl.innerText = "Natural organic linguistic structure verified";
    }

    const fakePct = data.fake_probability * 100;
    document.getElementById("liveProbBarFake").style.width = `${fakePct}%`;
    document.getElementById("liveProbBarGenuine").style.width = `${100 - fakePct}%`;

    // Per-Model Breakdown Chips
    const perModelEl = document.getElementById("livePerModelContainer");
    if (perModelEl && data.per_model_predictions) {
      perModelEl.innerHTML = Object.entries(data.per_model_predictions)
        .map(([key, val]) => {
          const isF = val.verdict === "Fake";
          const displayNames = {
            naive_bayes: "Naive Bayes",
            logistic_regression: "Logistic Reg (LASSO)",
            decision_tree: "Decision Tree",
            svm: "Linear SVM",
            knn: "KNN (k=5)",
            random_forest: "Random Forest",
            gradient_boosting: "Gradient Boosting",
            soft_voting: "Soft Voting Consensus",
          };
          return `
          <div class="p-2.5 rounded-lg bg-surface-container-low card-border flex items-center justify-between text-xs">
            <span class="font-medium text-on-surface truncate">${displayNames[key] || key}</span>
            <span class="font-bold px-2 py-0.5 rounded text-[10px] ${
              isF ? "bg-error-container text-on-error-container" : "bg-emerald-100 text-emerald-800"
            }">
              ${val.verdict} (${(val.probability_fake * 100).toFixed(0)}%)
            </span>
          </div>`;
        })
        .join("");
    }

    // Highlights Box
    const hlContainer = document.getElementById("liveHighlightsContainer");
    if (hlContainer) {
      if (data.suspicious_highlights && data.suspicious_highlights.length > 0) {
        hlContainer.innerHTML = `
          <div class="space-y-2">
            <div class="font-semibold text-secondary text-[11px]">Flagged Tokens:</div>
            <div class="flex flex-wrap gap-1.5">
              ${data.suspicious_highlights
                .map(
                  (h) => `
                <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded ${
                  h.severity === "high" ? "bg-error-container text-on-error-container font-bold" : "bg-amber-100 text-amber-800 font-semibold"
                }">
                  <span>${h.word}</span>
                  <span class="text-[9px] opacity-75">(${h.reason})</span>
                </span>`
                )
                .join("")}
            </div>
          </div>
        `;
      } else {
        hlContainer.innerHTML = `<span class="text-emerald-700 font-medium">No anomalous spam markers or extreme exclamations detected.</span>`;
      }
    }
  } catch (err) {
    alert("Inference error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="material-symbols-outlined text-[18px]">troubleshoot</span> Analyze Review Against Ensemble Pipeline`;
  }
}

// ==========================================
// INITIALIZATION
// ==========================================

window.addEventListener("DOMContentLoaded", () => {
  window.addEventListener("hashchange", renderRoute);
  renderRoute();
});
