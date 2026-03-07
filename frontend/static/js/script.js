(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    streamStatus: $("streamStatus"),
    sseStatus: $("sseStatus"),
    lastDetect: $("lastDetect"),
    alertBox: $("alertBox"),
    alertText: $("alertText"),
    dismissAlert: $("dismissAlert"),
    clearHistory: $("clearHistory"),
    historyList: $("historyList"),
    video: $("video"),
  };

  const eventsUrl = document.body.dataset.eventsUrl || "/events";
  const videoUrl = document.body.dataset.videoUrl || "/video_feed";

  // Ensure <img src> matches backend route
  if (els.video && els.video.getAttribute("src") !== videoUrl) {
    els.video.setAttribute("src", videoUrl);
  }

  const history = [];
  const HISTORY_MAX = 80;
  let alertTimer = null;

  function setStatus(el, text, state = "ok") {
    if (!el) return;
    el.textContent = text;
    el.classList.remove("ok", "warn", "bad");
    el.classList.add(state);
  }

  function showAlert(text) {
    if (!els.alertBox || !els.alertText) return;
    els.alertText.textContent = text || "UYARI: Insan tespit edildi!";
    els.alertBox.classList.remove("hidden");
    if (alertTimer) window.clearTimeout(alertTimer);
    alertTimer = window.setTimeout(() => {
      els.alertBox.classList.add("hidden");
    }, 4500);
  }

  function renderHistory() {
    if (!els.historyList) return;

    if (history.length === 0) {
      els.historyList.innerHTML = `<li class="history-empty">Henüz kayıt yok.</li>`;
      return;
    }

    els.historyList.innerHTML = history
      .map((e) => {
        const ts = e.ts || "-";
        const title = e.type === "person" ? "İNSAN TESPİTİ" : (e.type || "OLAY");
        const msg = e.message || "";
        const count = typeof e.count === "number" ? e.count : null;
        const meta = count !== null ? `<span class="pill">adet: ${count}</span>` : "";
        const cls = e.type === "person" ? "history-item danger" : "history-item";
        return `
          <li class="${cls}">
            <div class="row">
              <div class="left">
                <div class="h-title">${title}</div>
                <div class="h-msg">${escapeHtml(msg)}</div>
              </div>
              <div class="right">
                <div class="h-ts">${ts}</div>
                ${meta}
              </div>
            </div>
          </li>
        `;
      })
      .join("");
  }

  function addHistory(evt) {
    history.unshift(evt);
    if (history.length > HISTORY_MAX) history.pop();
    renderHistory();
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  // Stream health signals
  if (els.video) {
    els.video.addEventListener("load", () => setStatus(els.streamStatus, "AKTİF", "ok"));
    els.video.addEventListener("error", () => setStatus(els.streamStatus, "HATA", "bad"));
  }

  if (els.dismissAlert) {
    els.dismissAlert.addEventListener("click", () => els.alertBox?.classList.add("hidden"));
  }

  if (els.clearHistory) {
    els.clearHistory.addEventListener("click", () => {
      history.length = 0;
      renderHistory();
    });
  }

  // EventSource real-time detection listener
  setStatus(els.sseStatus, "BAĞLANIYOR", "warn");
  const es = new EventSource(eventsUrl);

  es.onopen = () => setStatus(els.sseStatus, "BAĞLI", "ok");
  es.onerror = () => setStatus(els.sseStatus, "SORUN", "bad");

  es.addEventListener("person", (ev) => {
    let data = null;
    try {
      data = JSON.parse(ev.data);
    } catch {
      data = { type: "person", ts: new Date().toISOString(), message: "UYARI: Insan tespit edildi!" };
    }

    els.lastDetect && (els.lastDetect.textContent = data.ts || "-");
    showAlert(data.message);
    addHistory(data);
  });
})();

