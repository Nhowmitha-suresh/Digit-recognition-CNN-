const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultText = document.getElementById('result');

// Canvas setup
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';
ctx.lineWidth = 15;
ctx.lineCap = 'round';

let drawing = false;

// Draw only while mouse is pressed inside canvas
canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  draw(e);
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener('mouseleave', () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

// Clear canvas
clearBtn.addEventListener('click', () => {
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultText.textContent = 'Predicted Digit: _';
});

// Predict function
predictBtn.addEventListener('click', async () => {
  const image = canvas.toDataURL('image/png');
  resultText.textContent = "⏳ Predicting...";
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image })
    });
    const data = await response.json();
    if (data.prediction !== undefined) {
      resultText.textContent = `Predicted Digit: ${data.prediction}`;
    } else {
      resultText.textContent = "⚠️ Error in prediction!";
    }
  } catch (err) {
    resultText.textContent = "⚠️ Server Error!";
    console.error(err);
  }
});
