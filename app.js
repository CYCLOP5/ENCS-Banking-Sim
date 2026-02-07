// ============================
// DATA INTEGRATION LAYER
// Replace these functions with your backend API calls
// ============================

const API_CONFIG = {
    // Change this to your backend URL when ready
    BASE_URL: 'http://localhost:8000/api',
    ENDPOINTS: {
        SIMULATE: '/simulate',
        NETWORK: '/network',
        STATS: '/stats',
        TIMELINE: '/timeline'
    }
};

// Mock data generation (replace with actual API calls)
function generateMockNetworkData() {
    const nodes = [];
    const links = [];
    const numNodes = 50;
    
    // Create SIFI core nodes
    const sifis = ['JP Morgan', 'BoA', 'Citigroup', 'Wells Fargo', 'Goldman Sachs'];
    sifis.forEach((name, i) => {
        nodes.push({
            id: i,
            name: name,
            type: 'core',
            status: 'healthy',
            assets: 2000 + Math.random() * 1000,
            centrality: 0.8 + Math.random() * 0.2
        });
    });
    
    // Create peripheral nodes
    for (let i = sifis.length; i < numNodes; i++) {
        nodes.push({
            id: i,
            name: `Bank ${i}`,
            type: 'peripheral',
            status: 'healthy',
            assets: 100 + Math.random() * 500,
            centrality: Math.random() * 0.5
        });
    }
    
    // Create core-periphery topology
    for (let i = 0; i < sifis.length; i++) {
        for (let j = i + 1; j < sifis.length; j++) {
            links.push({
                source: i,
                target: j,
                exposure: 50 + Math.random() * 200
            });
        }
    }
    
    for (let i = sifis.length; i < numNodes; i++) {
        const coreNode = Math.floor(Math.random() * sifis.length);
        links.push({
            source: i,
            target: coreNode,
            exposure: 10 + Math.random() * 50
        });
    }
    
    return { nodes, links };
}

// ============================
// REPLACE WITH BACKEND CALL
// ============================
async function fetchSimulationData(params) {
    // TODO: Replace with actual API call
    // Example:
    // const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SIMULATE}`, {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify(params)
    // });
    // return await response.json();
    
    // Mock delay for demonstration
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    return {
        defaulted_banks: Math.floor(Math.random() * 50) + 10,
        systemic_loss: (Math.random() * 500 + 100).toFixed(1),
        contagion_depth: Math.floor(Math.random() * 5) + 2,
        liquidity_stress: (Math.random() * 60 + 20).toFixed(1),
        risk_metrics: {
            credit: (Math.random() * 40 + 30).toFixed(1),
            liquidity: (Math.random() * 50 + 20).toFixed(1),
            market: (Math.random() * 30 + 10).toFixed(1),
            contagion: (Math.random() * 70 + 20).toFixed(1)
        },
        timeline: [
            { time: 'T=0', event: 'Initial shock: ' + params.shockBank + ' defaults', critical: true },
            { time: 'T=1', event: '12 counterparties enter liquidity stress', critical: false },
            { time: 'T=2', event: 'Credit lines frozen, fire sales begin', critical: true },
            { time: 'T=3', event: '23 additional defaults triggered', critical: true },
            { time: 'T=4', event: 'Contagion reaches peripheral institutions', critical: false }
        ],
        loss_distribution: Array.from({ length: 20 }, (_, i) => Math.random() * 100),
        cascade_data: Array.from({ length: 10 }, (_, i) => ({
            wave: i,
            defaults: Math.floor(Math.random() * 20) + (i * 3)
        }))
    };
}

// ============================
// NETWORK VISUALIZATION
// ============================
let networkData = generateMockNetworkData();
let simulation;

function initializeNetwork() {
    const container = document.getElementById('network-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const svg = d3.select('#network-canvas')
        .attr('width', width)
        .attr('height', height);
    
    svg.selectAll('*').remove();
    
    const g = svg.append('g');
    
    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.5, 3])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    simulation = d3.forceSimulation(networkData.nodes)
        .force('link', d3.forceLink(networkData.links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(20));
    
    const link = g.append('g')
        .selectAll('line')
        .data(networkData.links)
        .enter().append('line')
        .attr('stroke', '#1e293b')
        .attr('stroke-width', d => Math.sqrt(d.exposure) / 5)
        .attr('stroke-opacity', 0.6);
    
    const node = g.append('g')
        .selectAll('circle')
        .data(networkData.nodes)
        .enter().append('circle')
        .attr('r', d => d.type === 'core' ? 12 : 6)
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    const labels = g.append('g')
        .selectAll('text')
        .data(networkData.nodes.filter(d => d.type === 'core'))
        .enter().append('text')
        .text(d => d.name)
        .attr('font-size', '10px')
        .attr('fill', '#94a3b8')
        .attr('text-anchor', 'middle')
        .attr('dy', 20);
    
    node.append('title')
        .text(d => `${d.name}\nAssets: $${d.assets.toFixed(0)}B\nCentrality: ${d.centrality.toFixed(2)}`);
    
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        labels
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
    
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function getNodeColor(node) {
    if (node.status === 'default') return '#ef4444';
    if (node.status === 'stressed') return '#f59e0b';
    if (node.type === 'core') return '#3b82f6';
    return '#10b981';
}

function updateNetwork(simulationResults) {
    // Randomly mark some nodes as defaulted/stressed for demo
    const numDefaults = Math.min(simulationResults.defaulted_banks, networkData.nodes.length);
    
    networkData.nodes.forEach((node, i) => {
        if (i === 0) {
            node.status = 'default'; // Initial shock
        } else if (i < numDefaults) {
            node.status = Math.random() > 0.5 ? 'default' : 'stressed';
        } else {
            node.status = 'healthy';
        }
    });
    
    d3.select('#network-canvas')
        .selectAll('circle')
        .transition()
        .duration(1000)
        .attr('fill', d => getNodeColor(d));
}

// ============================
// CHARTS
// ============================
let lossChart, cascadeChart;

function initializeCharts() {
    // Loss Distribution Chart
    const lossCtx = document.getElementById('loss-distribution-chart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'bar',
        data: {
            labels: Array.from({ length: 20 }, (_, i) => `${i * 50}-${(i + 1) * 50}B`),
            datasets: [{
                label: 'Frequency',
                data: Array(20).fill(0),
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1a2247',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#1e293b' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { 
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });

    // Cascade Chart
    const cascadeCtx = document.getElementById('cascade-chart').getContext('2d');
    cascadeChart = new Chart(cascadeCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Cumulative Defaults',
                data: [],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.4,
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    display: true,
                    labels: { color: '#94a3b8' }
                },
                tooltip: {
                    backgroundColor: '#1a2247',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#1e293b' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { color: '#1e293b' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function updateCharts(data) {
    lossChart.data.datasets[0].data = data.loss_distribution;
    lossChart.update();

    cascadeChart.data.labels = data.cascade_data.map(d => `Wave ${d.wave}`);
    cascadeChart.data.datasets[0].data = data.cascade_data.map(d => d.defaults);
    cascadeChart.update();
}

// ============================
// UI UPDATES
// ============================
function updateMetrics(data) {
    document.getElementById('defaulted-banks').textContent = data.defaulted_banks;
    document.getElementById('default-change').textContent = `+${((data.defaulted_banks / 247) * 100).toFixed(1)}%`;
    document.getElementById('systemic-loss').textContent = `$${data.systemic_loss}B`;
    document.getElementById('contagion-depth').textContent = data.contagion_depth;
    document.getElementById('depth-change').textContent = 'waves';
    document.getElementById('liquidity-stress').textContent = `${data.liquidity_stress}%`;

    document.getElementById('credit-risk').textContent = `${data.risk_metrics.credit}%`;
    document.getElementById('liquidity-risk').textContent = `${data.risk_metrics.liquidity}%`;
    document.getElementById('market-risk').textContent = `${data.risk_metrics.market}%`;
    document.getElementById('contagion-risk').textContent = `${data.risk_metrics.contagion}%`;
}

function updateTimeline(events) {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    
    events.forEach(event => {
        const item = document.createElement('div');
        item.className = `timeline-item${event.critical ? ' critical' : ''}`;
        item.innerHTML = `
            <div class="timeline-time">${event.time}</div>
            <div class="timeline-content">${event.event}</div>
        `;
        timeline.appendChild(item);
    });
}

function updateRiskTable() {
    const tbody = document.getElementById('risk-table-body');
    const banks = [
        { name: 'JP Morgan Chase', assets: 3200, centrality: 0.95, exposure: 450, status: 'default' },
        { name: 'Bank of America', assets: 2500, centrality: 0.92, exposure: 380, status: 'stressed' },
        { name: 'Citigroup', assets: 2100, centrality: 0.88, exposure: 320, status: 'stressed' },
        { name: 'Wells Fargo', assets: 1900, centrality: 0.85, exposure: 290, status: 'healthy' },
        { name: 'Goldman Sachs', assets: 1200, centrality: 0.82, exposure: 210, status: 'healthy' }
    ];
    
    tbody.innerHTML = banks.map((bank, i) => `
        <tr>
            <td>${i + 1}</td>
            <td class="bank-name">${bank.name}</td>
            <td>$${bank.assets}</td>
            <td>${bank.centrality}</td>
            <td>$${bank.exposure}</td>
            <td><span class="status-badge ${bank.status}">${bank.status.toUpperCase()}</span></td>
        </tr>
    `).join('');
}

// ============================
// MAIN SIMULATION FUNCTION
// ============================
async function runSimulation() {
    const loading = document.getElementById('loading');
    loading.classList.add('active');
    
    const params = {
        shockBank: document.getElementById('shock-bank').value,
        shockMagnitude: document.getElementById('shock-magnitude').value / 100,
        panicThreshold: document.getElementById('panic-threshold').value,
        iterations: parseInt(document.getElementById('iterations').value)
    };
    
    try {
        const data = await fetchSimulationData(params);
        
        updateMetrics(data);
        updateTimeline(data.timeline);
        updateCharts(data);
        updateNetwork(data);
        updateRiskTable();
        
    } catch (error) {
        console.error('Simulation error:', error);
        alert('Simulation failed. Check console for details.');
    } finally {
        loading.classList.remove('active');
    }
}

function resetSimulation() {
    // Reset all metrics
    document.getElementById('defaulted-banks').textContent = '0';
    document.getElementById('default-change').textContent = '+0%';
    document.getElementById('systemic-loss').textContent = '$0B';
    document.getElementById('contagion-depth').textContent = '0';
    document.getElementById('liquidity-stress').textContent = '0%';
    
    document.getElementById('credit-risk').textContent = '0%';
    document.getElementById('liquidity-risk').textContent = '0%';
    document.getElementById('market-risk').textContent = '0%';
    document.getElementById('contagion-risk').textContent = '0%';
    
    // Reset timeline
    document.getElementById('timeline').innerHTML = `
        <div class="timeline-item">
            <div class="timeline-time">T=0 (Initial State)</div>
            <div class="timeline-content">System at equilibrium</div>
        </div>
    `;
    
    // Reset network
    networkData = generateMockNetworkData();
    initializeNetwork();
    
    // Reset charts
    lossChart.data.datasets[0].data = Array(20).fill(0);
    lossChart.update();
    cascadeChart.data.labels = [];
    cascadeChart.data.datasets[0].data = [];
    cascadeChart.update();
}

// ============================
// INITIALIZATION
// ============================
window.addEventListener('load', () => {
    initializeNetwork();
    initializeCharts();
    updateRiskTable();
});

window.addEventListener('resize', () => {
    initializeNetwork();
});
/////////////////////////////////////////////////////////INDEX KA JS
const glow = document.querySelector('.glow');

if (glow) {
  document.addEventListener('mousemove', e => {
    glow.style.left = e.clientX + 'px';
    glow.style.top = e.clientY + 'px';
  });
}

function typewriter(el, speed = 40) {
  const text = el.textContent;
  el.textContent = '';
  let i = 0;

  const timer = setInterval(() => {
    el.textContent += text[i];
    i++;
    if (i >= text.length) clearInterval(timer);
  }, speed);
}

window.addEventListener('load', () => {
  document.querySelectorAll('.typewriter').forEach(el => {
    typewriter(el);
  });
});

// window.addEventListener('load', () => {
//   setTimeout(() => {
//     document.querySelectorAll('.typewriter').forEach(el => {
//       el.classList.add('cursor-stop');
//     });
//   }, 10000); 
// });
