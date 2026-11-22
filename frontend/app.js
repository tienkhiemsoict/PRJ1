const IS_EXTERNAL_PREVIEW = window.location.protocol === 'file:' || window.location.port === '5500';
const API_ORIGIN = IS_EXTERNAL_PREVIEW ? 'http://127.0.0.1:8000' : window.location.origin;
const WS_ORIGIN = IS_EXTERNAL_PREVIEW
    ? 'ws://127.0.0.1:8000'
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

const CONFIG = {
    MAP: {
        CENTER: [21.0045, 105.8431],
        INITIAL_ZOOM: 14,
        MIN_ZOOM: 5,
        MAX_ZOOM: 19
    },
    API: {
        BASE_URL: API_ORIGIN,
        WS_URL: `${WS_ORIGIN}/ws/solve`,
        OSRM_URL: 'https://router.project-osrm.org/route/v1/driving',
        ENDPOINTS: {
            INSTANCES: '/api/instances',
            UPLOAD: '/api/upload',
            SOLVE: '/api/solve',
            COMPARE: '/api/compare',
            DB_VEHICLES: '/api/db/vehicles',
            DB_EXPERIMENTS: '/api/db/experiments'
        }
    },
    ICONS: {
        DEPOT: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
        CUSTOMER: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
        SHADOW: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png'
    },
    COLORS: [
        '#1632cf',
        '#bc1e1e',
        '#11990f',
        '#e67e22',
        '#8e44ad',
        '#16a085',
        '#d35400',
        '#2c3e50',
        '#c0392b',
        '#2980b9'
    ],
    THRESHOLD_N: 15
};

class AppState {
    constructor() {
        this.map = null;
        this.selectingMode = null;
        this.displayMode = 'euclidean';
        this.depot = null;
        this.customers = [];
        this.depotMarker = null;
        this.customerMarkers = [];
        this.routeLines = [];
        this.currentFilePath = null;
        this.currentFileContent = null;
        this.lastResult = null;
        this.compareChart = null;
        this.convChart = null;
        this.history = [];
        this.activeWs = null;
        this.runId = 0;
    }

    cancelActiveRun() {
        this.runId += 1;
        if (this.activeWs) {
            try {
                this.activeWs.close();
            } catch (e) {
                // ignore
            }
            this.activeWs = null;
        }
        return this.runId;
    }

    reset() {
        this.cancelActiveRun();
        this.clearMarkers();
        this.clearRoutes();
        this.selectingMode = null;
        this.depot = null;
        this.customers = [];
        this.currentFilePath = null;
        this.currentFileContent = null;
        this.lastResult = null;
        this.history = [];
    }

    clearMarkers() {
        if (this.depotMarker) {
            this.map.removeLayer(this.depotMarker);
            this.depotMarker = null;
        }
        this.customerMarkers.forEach(m => this.map.removeLayer(m));
        this.customerMarkers = [];
    }

    clearRoutes() {
        this.routeLines.forEach(l => this.map.removeLayer(l));
        this.routeLines = [];
    }

    totalDimension() {
        return (this.depot ? 1 : 0) + this.customers.length;
    }

    hasValidInput() {
        return Boolean(
            this.currentFilePath ||
            this.currentFileContent ||
            (this.depot && this.customers.length > 0)
        );
    }
}

const appState = new AppState();

function initializeMap() {
    appState.map = L.map('map', {
        center: CONFIG.MAP.CENTER,
        zoom: CONFIG.MAP.INITIAL_ZOOM,
        minZoom: CONFIG.MAP.MIN_ZOOM,
        maxZoom: CONFIG.MAP.MAX_ZOOM,
        renderer: L.canvas()
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(appState.map);

    setTimeout(() => appState.map.invalidateSize(), 300);
}

function createMarkerIcon(type) {
    const key = type === 'depot' ? 'DEPOT' : 'CUSTOMER';
    return L.icon({
        iconUrl: CONFIG.ICONS[key],
        shadowUrl: CONFIG.ICONS.SHADOW,
        iconSize: [25, 41],
        iconAnchor: [12, 41]
    });
}

function syncThresholdUI(dimension) {
    const n = dimension || parseInt(document.getElementById('input-n').value) || 1;
    const infoEl = document.getElementById('threshold-info');
    const selectEl = document.getElementById('algo-select');

    if (n <= CONFIG.THRESHOLD_N) {
        infoEl.innerText = `Chế độ: Chính xác (N = ${n} <= ${CONFIG.THRESHOLD_N})`;
        selectEl.innerHTML = `
            <option value="auto">Tự động</option>
            <option value="dynamic_programming">Dynamic Programming</option>
            <option value="backtracking">Backtracking</option>
        `;
    } else {
        infoEl.innerText = `Chế độ: Metaheuristic (N = ${n} > ${CONFIG.THRESHOLD_N})`;
        selectEl.innerHTML = `
            <option value="auto">Tự động</option>
            <option value="genetic_algorithm">Genetic Algorithm</option>
            <option value="simulated_annealing">Simulated Annealing</option>
        `;
    }
}

function setSelectionMode(mode) {
    appState.selectingMode = mode;
    appState.currentFilePath = null;
    appState.currentFileContent = null;
    document.getElementById('instance-select').value = '';
    if (mode === 'depot') {
        updateStatus('Chọn vị trí kho trên bản đồ', '#003a60');
    } else {
        updateStatus('Chọn vị trí khách hàng trên bản đồ', '#003a60');
    }
}

function handleMapClick(event) {
    if (!appState.selectingMode) return;

    const lat = Number(event.latlng.lat.toFixed(6));
    const lng = Number(event.latlng.lng.toFixed(6));
    const demand = parseInt(document.getElementById('input-demand').value) || 10;

    if (appState.selectingMode === 'depot') {
        setDepot({ lat, lng });
        appState.selectingMode = null;
        updateStatus('Đã đặt kho', '#11990f');
    } else if (appState.selectingMode === 'customer') {
        addCustomer({ lat, lng, demand });
        updateStatus(`Đã thêm khách hàng (${appState.customers.length})`, '#11990f');
    }

    const dim = appState.totalDimension();
    if (dim >= 2) {
        document.getElementById('input-n').value = dim;
        syncThresholdUI(dim);
    }
}

function setDepot(point) {
    appState.depot = { lat: point.lat, lng: point.lng };
    if (appState.depotMarker) {
        appState.map.removeLayer(appState.depotMarker);
    }
    const marker = L.marker([point.lat, point.lng], {
        icon: createMarkerIcon('depot')
    }).addTo(appState.map);
    marker.bindPopup('Kho trung tâm');
    appState.depotMarker = marker;
}

function addCustomer(point) {
    const idx = appState.customers.length + 2;
    const item = {
        id: point.id || idx,
        lat: point.lat,
        lng: point.lng,
        demand: point.demand || 10
    };
    appState.customers.push(item);

    const marker = L.marker([item.lat, item.lng], {
        icon: createMarkerIcon('customer')
    }).addTo(appState.map);
    marker.bindPopup(`Khách ${item.id} - Nhu cầu: ${item.demand}`);
    appState.customerMarkers.push(marker);
}

function renderMapNodes(nodes, capacity, numTrucks) {
    appState.clearMarkers();
    appState.clearRoutes();
    appState.depot = null;
    appState.customers = [];

    if (capacity) {
        document.getElementById('input-cap').value = capacity;
    }
    if (numTrucks) {
        document.getElementById('input-k').value = numTrucks;
    }

    const bounds = [];
    nodes.forEach(node => {
        bounds.push([node.lat, node.lng]);
        if (node.is_depot) {
            setDepot({ lat: node.lat, lng: node.lng });
        } else {
            addCustomer({
                id: node.id,
                lat: node.lat,
                lng: node.lng,
                demand: node.demand
            });
        }
    });

    document.getElementById('input-n').value = nodes.length;
    syncThresholdUI(nodes.length);

    if (bounds.length > 0) {
        appState.map.fitBounds(bounds, { padding: [40, 40] });
    }
}

function generateRandomPoints() {
    const n = Math.max(2, parseInt(document.getElementById('input-n').value) || 12);
    const baseDemand = Math.max(1, parseInt(document.getElementById('input-demand').value) || 15);

    appState.reset();
    document.getElementById('instance-select').value = '';

    const center = appState.map.getCenter();
    const span = 0.05;
    const depotPt = { lat: Number(center.lat.toFixed(6)), lng: Number(center.lng.toFixed(6)) };
    setDepot(depotPt);

    const bounds = [[depotPt.lat, depotPt.lng]];
    for (let i = 0; i < n - 1; i++) {
        const lat = Number((center.lat + (Math.random() - 0.5) * span).toFixed(6));
        const lng = Number((center.lng + (Math.random() - 0.5) * span).toFixed(6));
        const d = Math.max(1, Math.round(baseDemand * (0.5 + Math.random())));
        addCustomer({ id: i + 2, lat, lng, demand: d });
        bounds.push([lat, lng]);
    }

    appState.map.fitBounds(bounds, { padding: [40, 40] });

    syncThresholdUI(n);
    updateStatus(`Đã sinh ngẫu nhiên ${n} điểm`, '#11990f');
}

async function loadInstancesList() {
    try {
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.INSTANCES}`);
        if (!res.ok) return;
        const data = await res.json();
        const select = document.getElementById('instance-select');
        (data.instances || []).forEach(inst => {
            if (inst.error) return;
            const opt = document.createElement('option');
            opt.value = inst.name;
            opt.textContent = `${inst.name} (N=${inst.dimension}, Q=${inst.capacity})`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.log('Could not load instances list');
    }
}

async function loadSelectedInstance(name) {
    if (!name) return;
    try {
        updateStatus(`Đang tải ${name}`, '#003a60');
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.INSTANCES}/${encodeURIComponent(name)}`);
        if (!res.ok) throw new Error('Không tải được testcase');
        const data = await res.json();

        appState.reset();
        appState.currentFilePath = name;
        renderMapNodes(data.map_nodes || [], data.capacity, data.num_trucks);
        updateStatus(`Đã tải ${data.name} (N=${data.dimension})`, '#11990f');
    } catch (err) {
        updateStatus(`Lỗi: ${err.message}`, '#bc1e1e');
    }
}

function triggerFileUpload() {
    document.getElementById('vrp-file-input').click();
}

async function handleFileUpload(input) {
    const file = input.files && input.files[0];
    if (!file) return;

    try {
        const text = await file.text();
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.UPLOAD}`, {
            method: 'POST',
            headers: { 'Content-Type': 'text/plain;charset=utf-8' },
            body: text
        });
        if (!res.ok) throw new Error('File không hợp lệ');
        const data = await res.json();

        appState.reset();
        appState.currentFileContent = text;
        document.getElementById('instance-select').value = '';
        renderMapNodes(data.map_nodes || [], data.capacity, data.num_trucks);
        updateStatus(`Đã tải file ${file.name} (N=${data.dimension})`, '#11990f');
    } catch (err) {
        updateStatus(`Lỗi: ${err.message}`, '#bc1e1e');
    } finally {
        input.value = '';
    }
}

function buildSolvePayload() {
    const algo = document.getElementById('algo-select').value || 'auto';
    const capacity = Math.max(1, parseInt(document.getElementById('input-cap').value) || 100);
    const numTrucks = Math.max(0, parseInt(document.getElementById('input-k').value) || 0);

    const payload = {
        algorithm: algo,
        capacity: capacity,
        num_trucks: numTrucks,
        params: {}
    };

    if (appState.currentFileContent) {
        payload.file_content = appState.currentFileContent;
    } else if (appState.currentFilePath) {
        payload.file_path = appState.currentFilePath;
    } else if (appState.depot && appState.customers.length > 0) {
        payload.depot = appState.depot;
        payload.customers = appState.customers;
    }

    return payload;
}

function validateMapConstraints(payload) {
    if (!payload.customers || payload.customers.length === 0) return true;
    const cap = payload.capacity;
    const k = payload.num_trucks;
    const demands = payload.customers.map(c => c.demand || 0);
    const maxDemand = Math.max(...demands);
    const totalDemand = demands.reduce((a, b) => a + b, 0);

    if (maxDemand > cap) {
        const msg = `Nhu cầu khách (${maxDemand}) vượt tải trọng xe (${cap})`;
        updateStatus(msg, '#bc1e1e');
        alert(msg);
        return false;
    }
    if (k > 0 && totalDemand > k * cap) {
        const msg = `Tổng nhu cầu (${totalDemand}) vượt sức chứa ${k} xe (${k * cap})`;
        updateStatus(msg, '#bc1e1e');
        alert(msg);
        return false;
    }
    return true;
}

async function calculateRoute() {
    if (!appState.hasValidInput()) {
        alert('Vui lòng chọn testcase, tải file hoặc đặt điểm trên bản đồ');
        return;
    }

    const payload = buildSolvePayload();
    if (!validateMapConstraints(payload)) return;

    const myRunId = appState.cancelActiveRun();
    const dim = appState.totalDimension();
    updateStatus('Đang chạy thuật toán', '#003a60');

    if (dim > CONFIG.THRESHOLD_N) {
        const usedWs = await trySolveViaWebSocket(payload, myRunId);
        if (usedWs || myRunId !== appState.runId) return;
    }

    try {
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.SOLVE}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (myRunId !== appState.runId) return;
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || 'Lỗi giải bài toán');
        }
        await handleSolveSuccess(data);
    } catch (err) {
        if (myRunId === appState.runId) {
            updateStatus(`Lỗi: ${err.message}`, '#bc1e1e');
        }
    }
}

function drawRealtimePaths(paths) {
    appState.clearRoutes();
    (paths || []).forEach((coords, i) => {
        if (!coords || coords.length < 2) return;
        const line = L.polyline(coords, {
            color: CONFIG.COLORS[i % CONFIG.COLORS.length],
            weight: 3,
            opacity: 0.75
        }).addTo(appState.map);
        appState.routeLines.push(line);
    });
}

function trySolveViaWebSocket(payload, myRunId) {
    return new Promise(resolve => {
        let handled = false;
        let lastStep = 0;
        try {
            const ws = new WebSocket(CONFIG.API.WS_URL);
            appState.activeWs = ws;
            appState.history = [];

            ws.onopen = () => {
                if (myRunId !== appState.runId) {
                    ws.close();
                    return;
                }
                ws.send(JSON.stringify(payload));
            };

            ws.onmessage = async (event) => {
                if (myRunId !== appState.runId) return;
                const msg = JSON.parse(event.data);
                if (msg.type === 'progress') {
                    if (msg.generation < lastStep) return;
                    lastStep = msg.generation;
                    appState.history.push(msg.best_fitness);
                    updateStatus(`Bước ${msg.generation}: Chi phí ${msg.best_fitness}`, '#003a60');
                    if (msg.paths) {
                        drawRealtimePaths(msg.paths);
                    }
                    if (appState.convChart) {
                        appState.convChart.data.labels.push(msg.generation);
                        appState.convChart.data.datasets[0].data.push(msg.best_fitness);
                        appState.convChart.update('none');
                    }
                } else if (msg.type === 'complete') {
                    handled = true;
                    if (appState.activeWs === ws) appState.activeWs = null;
                    await handleSolveSuccess(msg.result);
                    resolve(true);
                } else if (msg.type === 'error') {
                    handled = true;
                    if (appState.activeWs === ws) appState.activeWs = null;
                    updateStatus(`Lỗi: ${msg.detail}`, '#bc1e1e');
                    resolve(true);
                }
            };

            ws.onerror = () => {
                if (appState.activeWs === ws) appState.activeWs = null;
                if (!handled) resolve(myRunId !== appState.runId);
            };

            ws.onclose = () => {
                if (appState.activeWs === ws) appState.activeWs = null;
                if (!handled) resolve(myRunId !== appState.runId);
            };
        } catch (e) {
            resolve(false);
        }
    });
}

async function handleSolveSuccess(result) {
    appState.lastResult = result;
    if (result.history && result.history.length > 0) {
        appState.history = result.history;
    }
    if (result.map_nodes && appState.totalDimension() === 0) {
        renderMapNodes(result.map_nodes, result.capacity, result.num_vehicles);
    }
    await renderSolutionRoutes(result);
    renderSummary(result);
    updateStatus(`Hoàn thành: ${result.algorithm}`, '#11990f');
}

async function renderSolutionRoutes(result) {
    appState.clearRoutes();
    const details = result.route_details || [];

    for (let i = 0; i < details.length; i++) {
        const item = details[i];
        const color = CONFIG.COLORS[i % CONFIG.COLORS.length];
        const coords = item.path_coords || [];
        if (coords.length < 2) continue;

        let drawCoords = coords;
        if (appState.displayMode === 'osrm') {
            const osrmCoords = await fetchOsrmGeometry(coords);
            if (osrmCoords && osrmCoords.length >= 2) {
                drawCoords = osrmCoords;
            }
        }

        const line = L.polyline(drawCoords, {
            color: color,
            weight: 4,
            opacity: 0.9
        }).addTo(appState.map);

        appState.routeLines.push(line);
    }
}

async function fetchOsrmGeometry(latlngs) {
    try {
        const coordStr = latlngs.map(pt => `${pt[1]},${pt[0]}`).join(';');
        const url = `${CONFIG.API.OSRM_URL}/${coordStr}?overview=full&geometries=geojson`;
        const res = await fetch(url);
        if (!res.ok) return null;
        const data = await res.json();
        if (!data.routes || !data.routes[0]) return null;
        return data.routes[0].geometry.coordinates.map(c => [c[1], c[0]]);
    } catch (e) {
        return null;
    }
}

async function setDisplayMode(mode) {
    appState.displayMode = mode;
    updateStatus(mode === 'osrm' ? 'Hiển thị: Đường thực tế' : 'Hiển thị: Đường thẳng', '#003a60');
    if (appState.lastResult) {
        await renderSolutionRoutes(appState.lastResult);
    }
}

function renderSummary(result) {
    const table = document.getElementById('summary-table');
    const tbody = document.getElementById('summary-body');
    const routesBox = document.getElementById('routes-box');

    table.style.display = 'table';
    tbody.innerHTML = `
        <tr><td>Thuật toán</td><td><b>${result.algorithm}</b></td></tr>
        <tr><td>Tổng quãng đường</td><td><b>${result.total_distance}</b></td></tr>
        <tr><td>Số xe sử dụng</td><td><b>${result.num_vehicles}</b></td></tr>
        <tr><td>Lấp đầy TB</td><td><b>${result.fill_rate ?? 0}%</b></td></tr>
        <tr><td>Thời gian</td><td><b>${result.execution_time} s</b></td></tr>
    `;

    const details = result.route_details || [];
    if (details.length > 0) {
        routesBox.style.display = 'block';
        routesBox.innerHTML = details.map((d, idx) => {
            const color = CONFIG.COLORS[idx % CONFIG.COLORS.length];
            return `<div><b style="color:${color}">Xe ${d.vehicle}</b> (${d.load}/${d.capacity}): ${d.route.join(' - ')}</div>`;
        }).join('');
    } else {
        routesBox.style.display = 'none';
    }
}

async function compareAlgorithms() {
    if (!appState.hasValidInput()) {
        alert('Vui lòng chọn dữ liệu trước khi so sánh');
        return;
    }

    const payload = buildSolvePayload();
    if (!validateMapConstraints(payload)) return;

    updateStatus('Đang so sánh thuật toán', '#003a60');
    try {
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.COMPARE}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || 'Lỗi so sánh');
        }
        displayComparisonResults(data);
        updateStatus('Đã hiển thị bảng so sánh', '#11990f');
    } catch (err) {
        updateStatus(`Lỗi: ${err.message}`, '#bc1e1e');
    }
}

function displayComparisonResults(data) {
    const tbody = document.getElementById('comp-body');
    tbody.innerHTML = data.map(item => `
        <tr>
            <td><b>${item.algo}</b></td>
            <td>${item.dist}</td>
            <td>${item.vehicles}</td>
            <td>${item.fill_rate}%</td>
            <td>${item.time_ms} ms</td>
        </tr>
    `).join('');

    const ctx = document.getElementById('algoChart').getContext('2d');
    if (appState.compareChart) {
        appState.compareChart.destroy();
    }

    appState.compareChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.algo),
            datasets: [
                {
                    label: 'Quãng đường',
                    data: data.map(item => item.dist),
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1,
                    yAxisID: 'y'
                },
                {
                    label: 'Thời gian (ms)',
                    data: data.map(item => item.time_ms),
                    backgroundColor: 'rgba(230, 126, 34, 0.7)',
                    borderColor: 'rgba(230, 126, 34, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Quãng đường' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Thời gian (ms)' }
                }
            }
        }
    });

    showModal('compare-modal');
}

function openConvergenceModal() {
    if (!appState.history || appState.history.length === 0) {
        alert('Chưa có dữ liệu hội tụ. Hãy chạy thuật toán Metaheuristic (N > 15)');
        return;
    }

    const ctx = document.getElementById('convChart').getContext('2d');
    if (appState.convChart) {
        appState.convChart.destroy();
    }

    appState.convChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: appState.history.map((_, idx) => idx + 1),
            datasets: [
                {
                    label: 'Chi phí tốt nhất',
                    data: appState.history,
                    borderColor: '#003a60',
                    backgroundColor: 'rgba(0, 58, 96, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { title: { display: true, text: 'Bước lặp / Thế hệ' } },
                y: { title: { display: true, text: 'Quãng đường' } }
            }
        }
    });

    showModal('convergence-modal');
}

function resetMap() {
    appState.reset();
    appState.map.setView(CONFIG.MAP.CENTER, CONFIG.MAP.INITIAL_ZOOM);
    document.getElementById('instance-select').value = '';
    document.getElementById('summary-table').style.display = 'none';
    document.getElementById('routes-box').style.display = 'none';
    syncThresholdUI(parseInt(document.getElementById('input-n').value) || 12);
    updateStatus('Sẵn sàng', '#4e4e4e');
}

function updateStatus(message, color = '#4e4e4e') {
    const statusEl = document.getElementById('status');
    statusEl.style.display = 'block';
    statusEl.innerText = message;
    statusEl.style.color = color;
}

function showModal(modalId) {
    document.getElementById(modalId).style.display = 'flex';
}

function closeModal(modalId) {
    document.getElementById(modalId || 'compare-modal').style.display = 'none';
}

async function loadVehiclesList() {
    try {
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.DB_VEHICLES}`);
        if (!res.ok) return;
        const data = await res.json();
        const select = document.getElementById('vehicle-select');
        if (!select) return;
        (data.vehicles || []).forEach(v => {
            const opt = document.createElement('option');
            opt.value = JSON.stringify({ capacity: v.capacity, count: v.default_count });
            opt.textContent = `${v.name} (Q=${v.capacity})`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.log('Could not load vehicles list');
    }
}

function applyVehiclePreset(val) {
    if (!val) return;
    try {
        const preset = JSON.parse(val);
        if (preset.capacity) document.getElementById('input-cap').value = preset.capacity;
        if (preset.count) document.getElementById('input-k').value = preset.count;
    } catch (e) {
        // ignore invalid value
    }
}

async function openHistoryModal() {
    try {
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.DB_EXPERIMENTS}?limit=50`);
        if (!res.ok) throw new Error('Không tải được lịch sử');
        const data = await res.json();
        const rows = data.experiments || [];
        const tbody = document.getElementById('history-body');
        if (rows.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8">Chưa có dữ liệu lịch sử</td></tr>';
        } else {
            tbody.innerHTML = rows.map(item => `
                <tr>
                    <td>${item.created_at}</td>
                    <td>${item.instance_name}</td>
                    <td>${item.dimension}</td>
                    <td>${item.algorithm}</td>
                    <td>${item.total_distance}</td>
                    <td>${item.num_vehicles}</td>
                    <td>${item.fill_rate}%</td>
                    <td>${item.execution_time} s</td>
                </tr>
            `).join('');
        }
        showModal('history-modal');
    } catch (err) {
        updateStatus(`Lỗi: ${err.message}`, '#bc1e1e');
    }
}

async function clearExperimentHistory() {
    try {
        const res = await fetch(`${CONFIG.API.BASE_URL}${CONFIG.API.ENDPOINTS.DB_EXPERIMENTS}`, {
            method: 'DELETE'
        });
        if (!res.ok) throw new Error('Không xóa được lịch sử');
        document.getElementById('history-body').innerHTML = '<tr><td colspan="8">Chưa có dữ liệu lịch sử</td></tr>';
        updateStatus('Đã xóa lịch sử', '#11990f');
    } catch (err) {
        updateStatus(`Lỗi: ${err.message}`, '#bc1e1e');
    }
}

function setupEventListeners() {
    appState.map.on('click', handleMapClick);
    document.getElementById('input-n').addEventListener('change', (e) => {
        syncThresholdUI(parseInt(e.target.value) || 12);
    });

    window.onclick = (event) => {
        ['compare-modal', 'convergence-modal', 'history-modal'].forEach(id => {
            const modal = document.getElementById(id);
            if (event.target === modal) {
                closeModal(id);
            }
        });
    };
}

function init() {
    initializeMap();
    setupEventListeners();
    loadInstancesList();
    loadVehiclesList();
    syncThresholdUI(12);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

window.setMode = setSelectionMode;
window.resetMap = resetMap;
window.runRouting = calculateRoute;
window.compareAlgos = compareAlgorithms;
window.closeModal = closeModal;
window.loadSelectedInstance = loadSelectedInstance;
window.triggerFileUpload = triggerFileUpload;
window.handleFileUpload = handleFileUpload;
window.generateRandomPoints = generateRandomPoints;
window.setDisplayMode = setDisplayMode;
window.openConvergenceModal = openConvergenceModal;
window.applyVehiclePreset = applyVehiclePreset;
window.openHistoryModal = openHistoryModal;
window.clearExperimentHistory = clearExperimentHistory;
