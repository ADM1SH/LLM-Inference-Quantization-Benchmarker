// Configuration constants matching Python output keys
const CONFIGS = {
    q4: 'qwen2.5-0.5b-q4',
    q8: 'qwen2.5-0.5b-q8',
    f16: 'qwen2.5-0.5b-f16'
};

// Global variables to store loaded benchmark results
let benchmarkData = null;
let chartInstances = {};

// On window load, fetch the JSON results and initialize the page
window.addEventListener('DOMContentLoaded', () => {
    loadBenchmarkResults();
});

/**
 * Fetch benchmark results from local directory.
 * If not found, handles error state gracefully.
 */
async function loadBenchmarkResults() {
    try {
        const response = await fetch('results/benchmark_results.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        benchmarkData = await response.json();
        
        // Hide loading state and initialize components
        initializeDashboard();
    } catch (error) {
        console.error('Error fetching benchmark results:', error);
        showErrorState();
    }
}

/**
 * Show visual fallback if results are missing.
 */
function showErrorState() {
    const errorMsg = "Benchmark results not found. Please run `run.sh` or `python benchmark.py` to compile the results first.";
    
    // Inject alert banners
    const overviewSection = document.getElementById('overview');
    if (overviewSection) {
        const alertDiv = document.createElement('div');
        alertDiv.style.background = 'rgba(239, 68, 68, 0.1)';
        alertDiv.style.border = '1px solid #ef4444';
        alertDiv.style.color = '#ef4444';
        alertDiv.style.padding = '1rem 1.5rem';
        alertDiv.style.borderRadius = '8px';
        alertDiv.style.marginTop = '1.5rem';
        alertDiv.style.fontWeight = '500';
        alertDiv.innerHTML = `⚠️ ${errorMsg}`;
        overviewSection.appendChild(alertDiv);
    }
}

/**
 * Populate UI metrics, charts, and selector dropdowns
 */
function initializeDashboard() {
    // 1. Update KPI cards
    updateKPINumbers();

    // 2. Populate prompt comparison selector dropdown
    populatePromptSelector();

    // 3. Render charts
    renderSpeedMemoryChart();
    renderTTFTQualityChart();
    renderCategorySpeedChart();

    // 4. Set up change event listener on selector
    const selector = document.getElementById('prompt-selector');
    selector.addEventListener('change', (e) => {
        updatePromptComparison(parseInt(e.target.value));
    });

    // 5. Initial draw of prompt comparison cards
    if (selector.options.length > 0) {
        updatePromptComparison(parseInt(selector.value));
    }
}

/**
 * Calculates and updates values for summary cards on dashboard
 */
function updateKPINumbers() {
    let bestSpeed = { name: '', val: 0 };
    let bestVram = { name: '', val: Infinity };
    let bestTtft = { name: '', val: Infinity };
    let bestQuality = { name: '', val: 0 };

    // Iterate through configurations to find the peaks
    for (const key in CONFIGS) {
        const mKey = CONFIGS[key];
        const mData = benchmarkData[mKey];
        if (!mData || !mData.summary) continue;

        const sum = mData.summary;

        if (sum.avg_tokens_per_sec > bestSpeed.val) {
            bestSpeed = { name: mData.quantization, val: sum.avg_tokens_per_sec };
        }
        if (sum.avg_peak_vram_mb < bestVram.val) {
            bestVram = { name: mData.quantization, val: sum.avg_peak_vram_mb };
        }
        if (sum.avg_ttft_sec < bestTtft.val) {
            bestTtft = { name: mData.quantization, val: sum.avg_ttft_sec };
        }
        if (sum.avg_judge_score > bestQuality.val) {
            bestQuality = { name: mData.quantization, val: sum.avg_judge_score };
        }
    }

    // Set text on card elements
    document.getElementById('kpi-best-speed').innerHTML = `${bestSpeed.val.toFixed(1)} <span style="font-size: 1rem; color: var(--accent-cyan);">t/s</span>`;
    document.querySelector('.speed-card .kpi-subtitle').innerText = `Winner: ${bestSpeed.name}`;

    document.getElementById('kpi-best-vram').innerHTML = `${bestVram.val.toFixed(0)} <span style="font-size: 1rem; color: var(--accent-magenta);">MB</span>`;
    document.querySelector('.memory-card .kpi-subtitle').innerText = `Winner: ${bestVram.name}`;

    document.getElementById('kpi-best-ttft').innerHTML = `${bestTtft.val.toFixed(3)} <span style="font-size: 1rem; color: var(--accent-blue);">s</span>`;
    document.querySelector('.ttft-card .kpi-subtitle').innerText = `Winner: ${bestTtft.name}`;

    document.getElementById('kpi-best-quality').innerHTML = `${bestQuality.val.toFixed(2)} <span style="font-size: 1rem; color: var(--accent-purple);">/10</span>`;
    document.querySelector('.quality-card .kpi-subtitle').innerText = `Winner: ${bestQuality.name}`;
}

/**
 * Reads list of runs and adds them to dropdown selector
 */
function populatePromptSelector() {
    const selector = document.getElementById('prompt-selector');
    selector.innerHTML = '';

    // Take prompts list from first model runs
    const firstModelKey = Object.values(CONFIGS)[0];
    const firstModelData = benchmarkData[firstModelKey];
    if (!firstModelData || !firstModelData.runs) return;

    firstModelData.runs.forEach(run => {
        const option = document.createElement('option');
        option.value = run.prompt_id;
        option.text = `[Prompt ${run.prompt_id}] ${run.category}`;
        selector.appendChild(option);
    });
}

/**
 * Updates side-by-side prompt detail cards based on dropdown value
 */
function updatePromptComparison(promptId) {
    let promptText = '';

    // Card configurations map
    const cards = {
        'q4': {
            speed: document.getElementById('q4-speed'),
            vram: document.getElementById('q4-vram'),
            ttft: document.getElementById('q4-ttft'),
            score: document.getElementById('q4-score'),
            output: document.getElementById('q4-output'),
            exp: document.getElementById('q4-explanation')
        },
        'q8': {
            speed: document.getElementById('q8-speed'),
            vram: document.getElementById('q8-vram'),
            ttft: document.getElementById('q8-ttft'),
            score: document.getElementById('q8-score'),
            output: document.getElementById('q8-output'),
            exp: document.getElementById('q8-explanation')
        },
        'f16': {
            speed: document.getElementById('f16-speed'),
            vram: document.getElementById('f16-vram'),
            ttft: document.getElementById('f16-ttft'),
            score: document.getElementById('f16-score'),
            output: document.getElementById('f16-output'),
            exp: document.getElementById('f16-explanation')
        }
    };

    // Load data for each card
    for (const key in CONFIGS) {
        const mKey = CONFIGS[key];
        const mData = benchmarkData[mKey];
        if (!mData || !mData.runs) continue;

        const run = mData.runs.find(r => r.prompt_id === promptId);
        if (!run) continue;

        promptText = run.prompt; // Save prompt description text

        const els = cards[key];
        els.speed.innerText = `${run.tokens_per_sec.toFixed(1)} t/s`;
        els.vram.innerText = `${run.peak_vram_mb.toFixed(0)} MB`;
        els.ttft.innerText = `${run.ttft_sec.toFixed(3)} s`;
        els.score.innerText = `${run.judge_score}/10`;
        els.output.innerText = run.response.trim();
        els.exp.innerText = run.judge_explanation;
    }

    document.getElementById('selected-prompt-text').innerText = promptText;
}

/**
 * Chart 1: Dual-axis Speed (Bar) & peak VRAM (Line)
 */
function renderSpeedMemoryChart() {
    const ctx = document.getElementById('speedMemoryChart').getContext('2d');
    
    const labels = [];
    const speedData = [];
    const memoryData = [];

    for (const key in CONFIGS) {
        const mKey = CONFIGS[key];
        const mData = benchmarkData[mKey];
        if (mData && mData.summary) {
            labels.push(mData.quantization);
            speedData.push(mData.summary.avg_tokens_per_sec);
            memoryData.push(mData.summary.avg_peak_vram_mb);
        }
    }

    chartInstances.speedMemory = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Generation Speed (tokens/s)',
                    data: speedData,
                    backgroundColor: 'rgba(0, 242, 254, 0.45)',
                    borderColor: '#00f2fe',
                    borderWidth: 2,
                    yAxisID: 'ySpeed',
                    order: 2
                },
                {
                    label: 'VRAM Usage (MB)',
                    data: memoryData,
                    type: 'line',
                    borderColor: '#fbc2eb',
                    backgroundColor: 'rgba(251, 194, 235, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: '#fbc2eb',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    yAxisID: 'yVram',
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#9ca3af', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                ySpeed: {
                    type: 'linear',
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: {
                        display: true,
                        text: 'Speed (tokens/sec)',
                        color: '#00f2fe',
                        font: { weight: '600' }
                    }
                },
                yVram: {
                    type: 'linear',
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#9ca3af' },
                    title: {
                        display: true,
                        text: 'VRAM footprint (MB)',
                        color: '#fbc2eb',
                        font: { weight: '600' }
                    }
                }
            }
        }
    });
}

/**
 * Chart 2: Double Bar comparing TTFT & Quality Rating
 */
function renderTTFTQualityChart() {
    const ctx = document.getElementById('ttftQualityChart').getContext('2d');
    
    const labels = [];
    const ttftData = [];
    const qualityData = [];

    for (const key in CONFIGS) {
        const mKey = CONFIGS[key];
        const mData = benchmarkData[mKey];
        if (mData && mData.summary) {
            labels.push(mData.quantization);
            ttftData.push(mData.summary.avg_ttft_sec);
            qualityData.push(mData.summary.avg_judge_score);
        }
    }

    chartInstances.ttftQuality = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Avg TTFT (seconds)',
                    data: ttftData,
                    backgroundColor: 'rgba(79, 172, 254, 0.45)',
                    borderColor: '#4facfe',
                    borderWidth: 2,
                    yAxisID: 'yTtft'
                },
                {
                    label: 'Judge Rating (1-10)',
                    data: qualityData,
                    backgroundColor: 'rgba(161, 140, 209, 0.45)',
                    borderColor: '#a18cd1',
                    borderWidth: 2,
                    yAxisID: 'yScore'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#9ca3af', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                yTtft: {
                    type: 'linear',
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: {
                        display: true,
                        text: 'TTFT Latency (seconds)',
                        color: '#4facfe',
                        font: { weight: '600' }
                    }
                },
                yScore: {
                    type: 'linear',
                    position: 'right',
                    min: 0,
                    max: 10,
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#9ca3af' },
                    title: {
                        display: true,
                        text: 'Judge Rating Scale',
                        color: '#a18cd1',
                        font: { weight: '600' }
                    }
                }
            }
        }
    });
}

/**
 * Chart 3: Detailed Speed broken down by Prompt Category
 */
function renderCategorySpeedChart() {
    const ctx = document.getElementById('categorySpeedChart').getContext('2d');
    
    // Extract categories dynamically
    const categories = [];
    const firstModelKey = Object.values(CONFIGS)[0];
    const firstModelData = benchmarkData[firstModelKey];
    if (!firstModelData || !firstModelData.runs) return;

    firstModelData.runs.forEach(run => {
        if (!categories.includes(run.category)) {
            categories.push(run.category);
        }
    });

    // Extract speeds per category for each config
    const datasets = [];
    const colors = {
        'q4': { fill: 'rgba(16, 185, 129, 0.45)', border: '#10b981' },
        'q8': { fill: 'rgba(245, 158, 11, 0.45)', border: '#f59e0b' },
        'f16': { fill: 'rgba(239, 68, 68, 0.45)', border: '#ef4444' }
    };

    for (const key in CONFIGS) {
        const mKey = CONFIGS[key];
        const mData = benchmarkData[mKey];
        if (!mData) continue;

        const speedPerCategory = [];
        categories.forEach(cat => {
            const run = mData.runs.find(r => r.category === cat);
            speedPerCategory.push(run ? run.tokens_per_sec : 0);
        });

        datasets.push({
            label: mData.quantization,
            data: speedPerCategory,
            backgroundColor: colors[key].fill,
            borderColor: colors[key].border,
            borderWidth: 2
        });
    }

    chartInstances.categorySpeed = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: categories,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#9ca3af', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' },
                    title: {
                        display: true,
                        text: 'Generation Speed (tokens/s)',
                        color: '#9ca3af',
                        font: { weight: '600' }
                    }
                }
            }
        }
    });
}
