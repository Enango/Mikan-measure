"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");

// マーカー検出用の作業キャンバス
const work = document.createElement("canvas");
const wctx = work.getContext("2d", { willReadFrequently: true });

const detector = new AR.Detector(); // 辞書は js-aruco 標準(ARUCO)

let engine = null;
let workerId = null;
let running = false;

// 最小値の記録
let minDiameter = Infinity;
let minData = null;
let validFrames = 0;

// ---------- 設定の保存(公開可能キーのみ。ブラウザ内保存) ----------
for (const id of ["apiKey", "modelName", "modelVersion", "confidence", "markerMm", "markerId"]) {
  const saved = localStorage.getItem("mikan_" + id);
  if (saved !== null) $(id).value = saved;
  $(id).addEventListener("change", () => localStorage.setItem("mikan_" + id, $(id).value));
}

function setStatus(msg) {
  $("status").textContent = msg;
}

// ---------- 入力ソース ----------
$("btnCamera").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    video.removeAttribute("src");
    await video.play();
    setStatus("カメラ起動。「計測開始」を押してください");
  } catch (e) {
    setStatus("カメラを起動できません: " + e.message);
  }
});

$("fileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = false;
  video.onloadedmetadata = () => setStatus("動画を読み込みました。「計測開始」を押してください");
});

// ---------- モデル読み込み ----------
async function loadModel() {
  if (workerId) return;
  const key = $("apiKey").value.trim();
  const name = $("modelName").value.trim();
  const ver = Number($("modelVersion").value);
  if (!key || !name || !ver) throw new Error("APIキー・モデルID・バージョンを入力してください");
  setStatus("モデルを読み込み中…(初回は時間がかかります)");
  engine = new inferencejs.InferenceEngine();
  workerId = await engine.startWorker(name, ver, key);
}

// ---------- 計測の開始/停止 ----------
$("btnStart").addEventListener("click", async () => {
  if (running) return;
  if (!video.srcObject && !video.src) {
    setStatus("先にカメラか動画ファイルを選んでください");
    return;
  }
  try {
    await loadModel();
  } catch (e) {
    setStatus("モデル読み込みエラー: " + e.message);
    return;
  }
  running = true;
  if (video.ended) video.currentTime = 0;
  await video.play();
  setStatus("計測中…");
  loop();
});

$("btnStop").addEventListener("click", stop);

function stop() {
  running = false;
  video.pause();
  setStatus("停止しました");
}

$("btnReset").addEventListener("click", () => {
  minDiameter = Infinity;
  minData = null;
  validFrames = 0;
  updatePanel(null);
  setStatus("最小値をリセットしました");
});

$("btnCopy").addEventListener("click", async () => {
  if (!minData) return;
  const text = minDiameter.toFixed(1);
  try {
    await navigator.clipboard.writeText(text);
    setStatus(`コピーしました: ${text} mm`);
  } catch {
    setStatus(`最小値: ${text} mm(コピーに失敗しました)`);
  }
});

// ---------- メインループ ----------
async function loop() {
  if (!running) return;
  if (video.ended) {
    running = false;
    setStatus("動画の終了。最小値が確定しました");
    return;
  }
  if (video.readyState >= 2 && !video.paused) {
    try {
      await processFrame();
    } catch (e) {
      console.error(e);
      setStatus("処理エラー: " + e.message);
    }
  }
  requestAnimationFrame(loop);
}

// 4辺のうち最も長い辺(px)。斜めでも縮みが最も小さい辺を使う
function markerSidePx(c) {
  let longest = 0;
  for (let i = 0; i < 4; i++) {
    const a = c[i];
    const b = c[(i + 1) % 4];
    longest = Math.max(longest, Math.hypot(a.x - b.x, a.y - b.y));
  }
  return longest;
}

async function processFrame() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;

  if (work.width !== w || work.height !== h) {
    work.width = w;
    work.height = h;
    overlay.width = w;
    overlay.height = h;
  }
  wctx.drawImage(video, 0, 0, w, h);
  octx.clearRect(0, 0, w, h);

  // 1) ARマーカー検出
  const markers = detector.detect(wctx.getImageData(0, 0, w, h));
  const idText = $("markerId").value.trim();
  const marker = idText === "" ? markers[0] : markers.find((m) => m.id === Number(idText));

  if (marker) drawMarker(marker, w);

  // 2) ミカン検出(Roboflow)
  const img = new inferencejs.CVImage(video);
  const preds = await engine.infer(workerId, img);
  const minConf = Number($("confidence").value);
  const oranges = preds.filter((p) => p.confidence >= minConf);
  if (oranges.length === 0) return;
  const best = oranges.reduce((a, b) => (a.confidence > b.confidence ? a : b));

  drawBox(best, w);

  if (!marker) {
    setStatus("マーカーが見つかりません(この フレームは無視)");
    return;
  }

  // 3) サイズ換算
  const markerMm = Number($("markerMm").value);
  const mmPerPx = markerMm / markerSidePx(marker.corners);
  const wMm = best.bbox.width * mmPerPx;
  const hMm = best.bbox.height * mmPerPx;
  const diameter = (wMm + hMm) / 2;

  validFrames++;
  if (diameter < minDiameter) {
    minDiameter = diameter;
    minData = { wMm, hMm, confidence: best.confidence };
  }
  updatePanel(diameter);
  setStatus("計測中…");
}

// ---------- 描画・表示 ----------
function drawMarker(marker, w) {
  const c = marker.corners;
  octx.lineWidth = Math.max(2, w / 400);
  octx.strokeStyle = "#00e676";
  octx.beginPath();
  octx.moveTo(c[0].x, c[0].y);
  for (let i = 1; i < 4; i++) octx.lineTo(c[i].x, c[i].y);
  octx.closePath();
  octx.stroke();
}

function drawBox(p, w) {
  const { x, y, width, height } = p.bbox; // x,y は中心座標
  octx.lineWidth = Math.max(2, w / 400);
  octx.strokeStyle = "#ff9800";
  octx.strokeRect(x - width / 2, y - height / 2, width, height);
  octx.fillStyle = "#ff9800";
  octx.font = `${Math.max(14, w / 50)}px sans-serif`;
  octx.fillText(`${p.class} ${(p.confidence * 100).toFixed(0)}%`, x - width / 2, y - height / 2 - 6);
}

function updatePanel(current) {
  $("curVal").textContent = current == null ? "--" : current.toFixed(1) + " mm";
  $("minVal").textContent = minData ? minDiameter.toFixed(1) + " mm" : "--";
  $("frameCount").textContent = validFrames;
  $("minDetail").textContent = minData
    ? `幅 ${minData.wMm.toFixed(1)} mm / 高さ ${minData.hMm.toFixed(1)} mm`
    : "幅 -- / 高さ --";
}
