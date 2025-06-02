const sendBtn = document.getElementById('send');
const promptEl = document.getElementById('prompt');
const outputEl = document.getElementById('output');

sendBtn.addEventListener('click', async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  outputEl.textContent = 'Loading...';
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt})
    });
    const data = await res.json();
    outputEl.textContent = data.response || data.error;
  } catch (err) {
    outputEl.textContent = 'Error: ' + err.message;
  }
});