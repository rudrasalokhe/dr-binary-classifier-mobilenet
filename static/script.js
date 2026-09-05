/* ═══════════════════════════════════════════════
   RetinaScope v3 — Script
   Pipeline step animation, proper state machine,
   timeout handling, severity-based result badges.
   ═══════════════════════════════════════════════ */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const hero         = $("hero");
  const uploadCard   = $("upload-card");
  const dropzone     = $("dropzone");
  const dropIdle     = $("dropzone-idle");
  const dropPreview  = $("dropzone-preview");
  const fileInput    = $("file-input");
  const previewImg   = $("preview-img");
  const removeBtn    = $("remove-btn");
  const analyzeBtn   = $("analyze-btn");
  const processing   = $("processing");
  const results      = $("results");
  const resultImg    = $("result-img");
  const resultBadge  = $("result-badge");
  const resultBadgeT = $("result-badge-text");
  const resultClass  = $("result-class");
  const resultConf   = $("result-conf");
  const chart        = $("chart");
  const resetBtn     = $("reset-btn");

  let file = null;
  let previewDataUrl = null;

  const LABELS = {
    Normal: "Normal",
    Diabetic_Retinopathy: "Diabetic Retinopathy",
    Glaucoma: "Glaucoma",
    AMD: "AMD (Macular Degeneration)",
  };

  // ─── File handling ──────────────────────────
  function setFile(f) {
    if (!f || !f.type.startsWith("image/")) return toast("Please upload a valid image.", true);
    file = f;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewDataUrl = e.target.result;
      previewImg.src = previewDataUrl;
      dropIdle.hidden = true;
      dropPreview.hidden = false;
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(f);
  }

  function clearFile() {
    file = null;
    previewDataUrl = null;
    fileInput.value = "";
    previewImg.src = "";
    dropPreview.hidden = true;
    dropIdle.hidden = false;
    analyzeBtn.disabled = true;
  }

  // ─── Drag & drop ───────────────────────────
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
  );
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#remove-btn") || e.target.closest("label")) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => { if (fileInput.files.length) setFile(fileInput.files[0]); });
  removeBtn.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

  // ─── Pipeline step animation ───────────────
  const STEP_COUNT = 5;
  let stepTimer = null;

  function startPipeline() {
    // Reset all steps
    for (let i = 0; i < STEP_COUNT; i++) {
      $(`step-${i}`).classList.remove("active", "done");
    }
    const progress = $("step-progress");
    if (progress) progress.style.width = "0%";

    let current = 0;

    function tick() {
      if (current > 0) {
        $(`step-${current - 1}`).classList.remove("active");
        $(`step-${current - 1}`).classList.add("done");
      }
      if (current < STEP_COUNT) {
        $(`step-${current}`).classList.add("active");
        if (progress) {
          progress.style.width = `${(current / (STEP_COUNT - 1)) * 100}%`;
        }
        current++;
        stepTimer = setTimeout(tick, 600);
      }
    }
    tick();
  }

  function stopPipeline() {
    clearTimeout(stepTimer);
    // Mark all done
    for (let i = 0; i < STEP_COUNT; i++) {
      $(`step-${i}`).classList.remove("active");
      $(`step-${i}`).classList.add("done");
    }
    const progress = $("step-progress");
    if (progress) progress.style.width = "100%";
  }

  // ─── State machine ─────────────────────────
  function showView(view) {
    // view: "upload" | "processing" | "results"
    hero.hidden        = view !== "upload";
    uploadCard.hidden  = view !== "upload";
    processing.hidden  = view !== "processing";
    results.hidden     = view !== "results";
  }

  // ─── Analyze ───────────────────────────────
  analyzeBtn.addEventListener("click", async () => {
    if (!file) return;

    showView("processing");
    startPipeline();

    const body = new FormData();
    body.append("file", file);

    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 120_000);

    try {
      const res = await fetch("/predict", { method: "POST", body, signal: ctrl.signal });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Server returned an error.");
      stopPipeline();
      // Small delay so user sees pipeline complete
      await sleep(400);
      render(data);
    } catch (err) {
      stopPipeline();
      if (err.name === "AbortError") {
        toast("Timed out — model may be warming up. Try again.", true);
      } else {
        toast(err.message || "Something went wrong.", true);
      }
      showView("upload");
    } finally {
      clearTimeout(timeout);
    }
  });

  // ─── Render ────────────────────────────────
  function render(data) {
    showView("results");

    // Show the uploaded image in results
    resultImg.src = previewDataUrl || "";

    // Severity badge
    const isNormal = data.predicted_class === "Normal";
    resultBadge.className = "result-badge " + (isNormal ? "severity-low" : "severity-high");
    resultBadgeT.textContent = isNormal ? "No pathology detected" : "Pathology detected";

    // Class & confidence
    resultClass.textContent = LABELS[data.predicted_class] || data.predicted_class;
    resultConf.textContent  = `${data.confidence.toFixed(1)}% confidence`;

    // Bar chart
    chart.innerHTML = "";
    const entries = Object.entries(data.all_confidences);
    const maxVal = Math.max(...entries.map(([, v]) => v), 0.1);

    entries.forEach(([cls, pct]) => {
      const row = document.createElement("div");
      row.className = "chart-row" + (cls === data.predicted_class ? " is-top" : "");
      row.innerHTML = `
        <div class="chart-top">
          <span class="chart-name">${LABELS[cls] || cls}</span>
          <span class="chart-val">${pct.toFixed(1)}%</span>
        </div>
        <div class="chart-track">
          <div class="chart-bar" data-cls="${cls}"></div>
        </div>`;
      chart.appendChild(row);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          row.querySelector(".chart-bar").style.width = `${(pct / maxVal) * 100}%`;
        });
      });
    });
  }

  // ─── Reset ─────────────────────────────────
  resetBtn.addEventListener("click", () => {
    clearFile();
    showView("upload");
  });

  // ─── Helpers ───────────────────────────────
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  function toast(msg, isError) {
    document.querySelectorAll(".toast").forEach((t) => t.remove());
    const el = document.createElement("div");
    el.className = "toast" + (isError ? " toast--error" : "");
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }
})();
