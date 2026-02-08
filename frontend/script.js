const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');

const networkCanvas = document.getElementById('networkCanvas');
const netCtx = networkCanvas.getContext('2d');

const outputContainer = document.getElementById('outputContainer');

let isDrawing = false;
let debounceTimer;

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';
ctx.lineWidth = 20;
ctx.lineCap = 'round';

function getMousePos(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
    predictDigit();
}

function draw(e) {
    if (!isDrawing) return;

    const pos = getMousePos(e);

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(predictDigit, 100);
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    netCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
    document.querySelectorAll('.output-item').forEach(el => {
        el.classList.remove('active');
        el.style.backgroundColor = '';
        el.style.color = '';
    });
});

async function predictDigit() {
    const imageData = canvas.toDataURL('image/png');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData }),
        });

        const result = await response.json();
        if (result.error) return;

        updateOutputStrip(result.top_predictions);
        drawNetworkVisualization(result.activations, result.probabilities);

    } catch (error) {
        console.error('Error:', error);
    }
}

function updateOutputStrip(predictions) {
    outputContainer.innerHTML = '';

    predictions.forEach(p => {
        const div = document.createElement('div');
        div.className = 'output-item';
        div.textContent = p.char;

        const prob = p.probability;
        if (prob > 0.05) {
            div.classList.add('active');
            div.style.backgroundColor = `rgba(37, 99, 235, ${prob * 0.9})`;
            div.style.color = 'white';
        }
        outputContainer.appendChild(div);
    });
}

function drawNetworkVisualization(activations, probabilities) {
    const w = networkCanvas.width;
    const h = networkCanvas.height;
    netCtx.clearRect(0, 0, w, h);

    const margin = 20;
    const layer1Y = h - 50;
    const layer2Y = h / 2;
    const outputY = 50;

    drawDynamicConnections(layer2Y, outputY, activations[1], probabilities, 50, 15);

    drawDynamicConnections(layer1Y, layer2Y, activations[0], activations[1], 100, 50);

    drawLayerNodes(layer1Y, activations[0], 256);
    drawLayerNodes(layer2Y, activations[1], 128);
    drawLayerNodes(outputY, probabilities, 47, true);
}

function drawDynamicConnections(y1, y2, sourceData, targetData, numSourceVis, numTargetVis) {
    const sourceIndices = getActiveIndices(sourceData, numSourceVis);
    const targetIndices = getActiveIndices(targetData, numTargetVis);

    netCtx.lineWidth = 1.5;

    targetIndices.forEach(tIdx => {
        const tx = mapIndexToX(tIdx, targetData.length, networkCanvas.width);

        for (let i = 0; i < 5; i++) {
            if (sourceIndices.length === 0) break;
            const randS = sourceIndices[Math.floor(Math.random() * sourceIndices.length)];
            const sx = mapIndexToX(randS, sourceData.length, networkCanvas.width);

            const val = targetData[tIdx];
            const alpha = Math.min(val * 0.8, 0.8);
            if (alpha < 0.05) continue;

            netCtx.strokeStyle = `rgba(37, 99, 235, ${alpha})`;

            if (Math.random() > 0.8) {
                netCtx.strokeStyle = `rgba(219, 39, 119, ${alpha})`;
            }

            netCtx.beginPath();
            netCtx.moveTo(sx, y1);
            netCtx.lineTo(tx, y2);
            netCtx.stroke();
        }
    });
}

function drawLayerNodes(y, data, trueLength, isOutput = false) {
    const w = networkCanvas.width;
    const nodeW = Math.max((w - 40) / trueLength, 2);
    const spacing = 1;

    data.forEach((val, i) => {
        const x = 20 + i * (w - 40) / trueLength;

        const intensity = Math.min(val / 2.0, 1.0);

        if (intensity > 0.1) {
            netCtx.fillStyle = `rgba(30, 41, 59, ${intensity})`;
            if (isOutput) netCtx.fillStyle = `rgba(37, 99, 235, ${intensity})`;

            netCtx.fillRect(x, y - 5, nodeW - spacing, 10);

        } else {
            netCtx.fillStyle = `rgba(0, 0, 0, 0.05)`;
            netCtx.fillRect(x, y - 2, nodeW - spacing, 4);
        }
    });
}

function getActiveIndices(data, limit) {
    const indexed = data.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    return indexed.slice(0, limit).map(o => o.i);
}

function mapIndexToX(i, total, w) {
    return 20 + i * (w - 40) / total + ((w - 40) / total) / 2;
}
