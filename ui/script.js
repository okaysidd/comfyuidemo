(() => {
  const form = document.getElementById('prompt-form');
  const promptInput = document.getElementById('prompt-input');
  const submitBtn = document.getElementById('submit-btn');
  const statusText = document.getElementById('status-text');
  const result = document.getElementById('result');
  const imgEl = document.getElementById('result-image');
  const promptIdEl = document.getElementById('prompt-id');
  const fileNameEl = document.getElementById('filename');

  let pollTimer = null;

  function setBusy(busy) {
    submitBtn.disabled = busy;
    form.classList.toggle('busy', busy);
  }

  function setStatus(text, type = 'info') {
    statusText.textContent = text;
    statusText.className = `status ${type}`;
  }

  async function callGenerate(posPrompt) {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pos_prompt: posPrompt }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`Failed to start job: ${res.status} ${detail}`);
    }
    return res.json();
  }

  async function fetchStatus(promptId) {
    const res = await fetch(`/status/${encodeURIComponent(promptId)}`);
    if (!res.ok) {
      throw new Error(`Status request failed: ${res.status}`);
    }
    return res.json();
  }

  function showImage(filename) {
    const url = `/image/${encodeURIComponent(filename)}`;
    imgEl.src = url;
    result.classList.remove('hidden');
  }

  function startPolling(promptId) {
    if (pollTimer) clearInterval(pollTimer);
    setStatus('Queued... polling status every 5s');
    pollTimer = setInterval(async () => {
      try {
        const data = await fetchStatus(promptId);
        const status = (data && data.status) || '';
        const outputs = (data && data.outputs) || [];

        if (status === 'completed' || status === 'success') {
          setStatus('Completed', 'success');
          clearInterval(pollTimer);
          if (outputs.length > 0 && outputs[0] && outputs[0].filename) {
            const file = outputs[outputs.length - 1].filename;
            fileNameEl.textContent = file;
            showImage(file);
          } else {
            setStatus('Completed but no outputs recorded. Try again.', 'warn');
            setBusy(false);
          }
          setBusy(false);
        } else if (status === 'not_found') {
          setStatus('Job not found', 'error');
          clearInterval(pollTimer);
          setBusy(false);
        } else {
          setStatus(`Status: ${status || 'unknown'}...`);
        }
      } catch (err) {
        console.error(err);
        setStatus('Error while polling status', 'error');
        clearInterval(pollTimer);
        setBusy(false);
      }
    }, 5000);
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text) return;

    result.classList.add('hidden');
    imgEl.removeAttribute('src');
    promptIdEl.textContent = '';
    fileNameEl.textContent = '';

    try {
      setBusy(true);
      setStatus('Submitting prompt...');
      const gen = await callGenerate(text);
      const promptId = gen && gen.prompt_id;
      if (!promptId) throw new Error('No prompt_id returned');
      promptIdEl.textContent = promptId;
      setStatus('Started. Waiting for progress updates...');
      startPolling(promptId);
    } catch (err) {
      console.error(err);
      setStatus(String(err.message || err), 'error');
      setBusy(false);
    }
  });
})();


