const youtubeUrlInput = document.getElementById('youtube-url');
const languageSelect  = document.getElementById('language');
const chunkInput      = document.getElementById('chunk');
const btn             = document.getElementById('transcribe-btn');
const retrySafeBtn    = document.getElementById('retry-safe');
const loadingMsg      = document.getElementById('loading-message');
const transcriptBox   = document.getElementById('transcript-box');
const downloadLink    = document.getElementById('download-link');
const statusBar       = document.getElementById('status');
const metaBox         = document.getElementById('meta');

let lastPayloadForRetry = null;

function setStatus(kind, html) {
  statusBar.className = 'alert';
  if (kind === 'error') statusBar.classList.add('alert--error');
  else if (kind === 'info') statusBar.classList.add('alert--info');
  else if (kind === 'success') statusBar.classList.add('alert--success');
  statusBar.innerHTML = html;
  statusBar.style.display = 'block';
}
function clearStatus() {
  statusBar.style.display = 'none';
  statusBar.innerHTML = '';
  statusBar.className = 'alert';
}

function setLoading(isLoading) {
  btn.disabled = isLoading;
  retrySafeBtn.disabled = isLoading;
  loadingMsg.style.display = isLoading ? 'block' : 'none';
}

function showTranscript(text) {
  transcriptBox.textContent = text;
  transcriptBox.style.display = 'block';
  // download link
  const blob = new Blob([text], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  downloadLink.href = url;
  downloadLink.download = 'transcript.txt';
  downloadLink.textContent = 'Download Transcript';
  downloadLink.style.display = 'block';
}

function showMeta(strategy, warnings) {
  const parts = [];
  if (strategy) {
    parts.push(`<strong>Strategy</strong> — device: ${strategy.device}${strategy.gpu_index !== null ? ' #'+strategy.gpu_index : ''}, compute: ${strategy.compute_type}, chunk: ${strategy.chunk_size}, diar: ${strategy.diar_device}`);
  }
  if (warnings && warnings.length) {
    parts.push(`<strong>Notes</strong> — ${warnings.join(' • ')}`);
  }
  if (parts.length) {
    metaBox.innerHTML = parts.join('<br>');
    metaBox.style.display = 'block';
  } else {
    metaBox.style.display = 'none';
  }
}

async function callTranscribe(body) {
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const ct = res.headers.get('content-type') || '';
  let payload;
  if (ct.includes('application/json')) {
    payload = await res.json();
  } else {
    const text = await res.text();
    payload = { error_code: 'SERVER_ERROR', message: text };
  }

  if (!res.ok) {
    throw { status: res.status, payload };
  }
  return payload;
}

async function doTranscribe({ safeMode = false, overrideChunk = null } = {}) {
  clearStatus();
  transcriptBox.style.display = 'none';
  downloadLink.style.display  = 'none';
  metaBox.style.display       = 'none';

  const youtubeUrl = youtubeUrlInput.value.trim();
  if (!/^https?:\/\//.test(youtubeUrl)) {
    setStatus('error', 'Please enter a valid YouTube URL (http/https).');
    return;
  }
  const language  = languageSelect.value;
  const chunkSize = overrideChunk ?? parseInt(chunkInput.value || '30', 10);

  const body = {
    youtube_url: youtubeUrl,
    language,
    chunk_size: chunkSize,
    safe_mode: safeMode
  };

  lastPayloadForRetry = { ...body }; // store for user-initiated retries

  try {
    setLoading(true);
    setStatus('info', 'Working on it… this can take a while for long videos.');
    const out = await callTranscribe(body);
    clearStatus();
    showTranscript(out.transcript);
    showMeta(out.strategy, out.warnings || []);
    setStatus('success', 'Done.');
    retrySafeBtn.style.display = 'none';
  } catch (err) {
    const { status, payload } = err;
    if (payload && payload.error_code === 'OOM') {
      const sugg = payload.suggestion || {};
      const nextChunk = sugg.suggested_chunk_size || Math.max(10, Math.floor(chunkSize / 2));
      setStatus('error', `GPU memory was tight. ${sugg.message || ''}<br><small>Tip: Safe Mode uses smaller chunks and diarization on CPU.</small>`);
      retrySafeBtn.style.display = 'inline-block';
      retrySafeBtn.onclick = () => doTranscribe({ safeMode: true, overrideChunk: nextChunk });
    } else if (payload && payload.error_code === 'BUSY') {
      const wait = payload.retry_after ?? 2;
      setStatus('error', `Server is busy. Retrying in ${wait}s…`);
      retrySafeBtn.style.display = 'none';
      setTimeout(() => doTranscribe({ safeMode: false }), wait * 1000);
    } else {
      setStatus('error', `Error: ${payload && payload.message ? payload.message : 'Unknown server error'}`);
      retrySafeBtn.style.display = 'inline-block';
      retrySafeBtn.onclick = () => doTranscribe({ safeMode: true });
    }
  } finally {
    setLoading(false);
  }
}

btn.addEventListener('click', () => doTranscribe());
