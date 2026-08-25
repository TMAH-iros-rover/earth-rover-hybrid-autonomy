"use strict";

const state = {
  missionActive: false,
  missionConfigured: false,
  roverConnected: false,
  operationMode: "direct_bot",
  cameraAndTelemetryAllowed: false,
  missionRequestRunning: false,
  telemetryTimer: null,
  cameraTimer: null,
  statusTimer: null,
  telemetryRequestRunning: false,
  cameraRequestRunning: false,
  map: null,
  roverMarker: null,
  routeLayer: null,
  globalPathLayer: null,
  trailLayer: null,
  checkpointLayers: [],
  trail: [],
  mapHasView: false,
  checkpointRoute: [],
  latestCheckpoint: 0,
  cameraView: "raw",
  samTpAvailable: false,
  samTpAutoSelected: false,
  samTpStatusTimer: null,
  samTpStatusRequestRunning: false,
  samTpStatusInitialized: false,
  samTpNextPollMs: 0,
  samNavigation: null,
  autonomyStatusTimer: null,
  autonomyStatusRequestRunning: false,
  autonomyNextPollMs: 0,
  diagnosticsTimer: null,
  diagnosticsRequestRunning: false,
  telemetryReady: false,
  frontCameraReady: false,
  remoteUserCount: 0,
  latestMissionStatus: null,
  latestSamStatus: null,
  latestAutonomyStatus: null,
  latestTelemetry: null,
  latestTelemetryReceivedAt: null,
  telemetryError: null,
  controlTrend: [],
  pendingMotionCommand: null,
  lastMotionResponse: null,
  lastAcceptedCommandMode: null,
};

const SAM_TP_STATUS_PATH = "/sam-tp-status";
const SAM_TP_OVERLAY_PATH = "/sam-tp-overlay.jpg";
const AUTONOMY_STATUS_PATH = "/autonomy-status";
const AUTONOMY_STOP_PATH = "/autonomy-stop";
const AUTONOMY_RESUME_PATH = "/autonomy-resume";
const TRAIL_MIN_DISTANCE_M = 2.0;

const elements = {
  connection: document.querySelector("#connection-state"),
  missionState: document.querySelector("#mission-state"),
  missionDetail: document.querySelector("#mission-detail"),
  start: document.querySelector("#start-mission"),
  connect: document.querySelector("#connect-rover"),
  refresh: document.querySelector("#refresh-mission"),
  end: document.querySelector("#end-mission"),
  stopAutonomy: document.querySelector("#stop-autonomy"),
  resumeAutonomy: document.querySelector("#resume-autonomy"),
  disconnect: document.querySelector("#disconnect-rover"),
  missionSlug: document.querySelector("#mission-slug"),
  camera: document.querySelector("#front-camera"),
  pathGeometryOverlay: document.querySelector("#path-geometry-overlay"),
  candidatePathLayer: document.querySelector("#candidate-path-layer"),
  selectedFootprintLayer: document.querySelector("#selected-footprint-layer"),
  globalHeadingLayer: document.querySelector("#global-heading-layer"),
  cameraPlaceholder: document.querySelector("#camera-placeholder"),
  cameraState: document.querySelector("#camera-state"),
  cameraMeta: document.querySelector("#camera-meta"),
  telemetryTime: document.querySelector("#telemetry-time"),
  checkpointSummary: document.querySelector("#checkpoint-summary"),
  checkpointList: document.querySelector("#checkpoint-list"),
  responseLog: document.querySelector("#response-log"),
  mapSummary: document.querySelector("#map-summary"),
  mapPosition: document.querySelector("#map-position"),
  viewRaw: document.querySelector("#view-raw"),
  viewSamTp: document.querySelector("#view-sam-tp"),
  samTpState: document.querySelector("#sam-tp-state"),
  samTpMetrics: document.querySelector("#sam-tp-metrics"),
  autonomyState: document.querySelector("#autonomy-state"),
  sideSectorLeft: document.querySelector("#side-sector-left"),
  sideSectorRight: document.querySelector("#side-sector-right"),
  recoveryMetrics: document.querySelector("#recovery-metrics"),
  driveHud: document.querySelector("#drive-hud"),
  hudPath: document.querySelector("#hud-path"),
  hudCommand: document.querySelector("#hud-command"),
  pathDirectionLabel: document.querySelector("#path-direction-label"),
  pathHealth: document.querySelector("#path-health"),
  pathDirectionArrow: document.querySelector("#path-direction-arrow"),
  pathSelectedHeading: document.querySelector("#path-selected-heading"),
  pathGlobalHeading: document.querySelector("#path-global-heading"),
  pathScore: document.querySelector("#path-score"),
  pathAge: document.querySelector("#path-age"),
  pathFrameId: document.querySelector("#path-frame-id"),
  pathPlanId: document.querySelector("#path-plan-id"),
  candidateStrip: document.querySelector("#candidate-strip"),
  pathDecisionReason: document.querySelector("#path-decision-reason"),
  commandMode: document.querySelector("#command-mode"),
  commandHealth: document.querySelector("#command-health"),
  internalLinear: document.querySelector("#internal-linear"),
  internalAngular: document.querySelector("#internal-angular"),
  sdkLinear: document.querySelector("#sdk-linear"),
  sdkAngular: document.querySelector("#sdk-angular"),
  linearGaugeFill: document.querySelector("#linear-gauge-fill"),
  angularGaugeMarker: document.querySelector("#angular-gauge-marker"),
  commandTransmit: document.querySelector("#command-transmit"),
  commandAge: document.querySelector("#command-age"),
  motionSpeed: document.querySelector("#motion-speed"),
  motionRpms: document.querySelector("#motion-rpms"),
  commandId: document.querySelector("#command-id"),
  commandLatency: document.querySelector("#command-latency"),
  motionAge: document.querySelector("#motion-age"),
  motionResponse: document.querySelector("#motion-response"),
  controlTrendCanvas: document.querySelector("#control-trend-canvas"),
  commandReason: document.querySelector("#command-reason"),
};

function nowLabel() {
  return new Date().toLocaleTimeString();
}

function appendLog(label, payload, isError = false) {
  const entry = document.createElement("div");
  entry.className = `log-entry ${isError ? "log-error" : "log-success"}`;
  const body = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  entry.textContent = `[${nowLabel()}] ${label}\n${body}`;
  elements.responseLog.prepend(entry);
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = { detail: `HTTP ${response.status} returned non-JSON data` };
  }
  if (!response.ok) {
    const error = new Error(payload.detail || `HTTP ${response.status}`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

function setConnection(online) {
  elements.connection.textContent = online ? "SDK online" : "SDK unavailable";
  elements.connection.className = `status ${online ? "status-online" : "status-offline"}`;
}

function setMissionControls(active, configured = state.missionConfigured) {
  state.missionActive = active;
  state.missionConfigured = configured;
  const missionRequested = elements.missionSlug.value.trim().length > 0;
  elements.start.textContent = "Start Mission";
  elements.refresh.textContent = missionRequested ? "Get Mission" : "Get Status";
  elements.end.textContent = active ? "End Mission" : "Reset Mission";
  elements.connect.disabled = state.roverConnected || state.missionRequestRunning;
  elements.disconnect.disabled = !state.roverConnected || active || state.missionRequestRunning;
  elements.start.disabled = !missionRequested || active || state.missionRequestRunning;
  // A cloud ride can remain stale after the local SDK process was restarted.
  // In that case mission_active is false even though /end-mission is exactly
  // the recovery action the operator needs. Keep reset available whenever a
  // valid mission slug has been selected/configured.
  elements.end.disabled =
    (!missionRequested || (!active && !configured))
    || state.missionRequestRunning;
  elements.refresh.disabled = state.missionRequestRunning;
  elements.missionState.textContent = configured
    ? (active ? "ACTIVE" : state.roverConnected ? "READY" : "INACTIVE")
    : (state.roverConnected ? "ROVER CONNECTED" : "DIRECT BOT MODE");
  elements.missionState.style.color = active || !configured ? "#8cdda9" : "#ffaaa3";
  if (!state.cameraAndTelemetryAllowed) {
    stopMissionPolling();
    elements.camera.style.display = "none";
    elements.cameraPlaceholder.style.display = "grid";
    elements.cameraState.textContent = "Idle";
    elements.cameraState.className = "status status-idle";
  } else {
    startMissionPolling();
  }
}

function renderMissionStatus(status) {
  state.latestMissionStatus = status;
  setConnection(true);
  state.roverConnected = Boolean(status.rover_connected);
  elements.connection.textContent = state.roverConnected
    ? "Bridge connected · checking rover"
    : "SDK online";
  state.operationMode = status.operation_mode || "direct_bot";
  state.cameraAndTelemetryAllowed = Boolean(status.camera_and_telemetry_allowed);
  setMissionControls(
    Boolean(status.mission_active),
    Boolean(status.mission_configured),
  );
  elements.missionDetail.textContent = status.mission_configured
    ? `${status.checkpoint_count} checkpoints loaded`
    : "No mission tracking; direct camera and telemetry test mode";
  elements.checkpointSummary.textContent = status.mission_configured
    ? `count ${status.checkpoint_count} | latest ${status.latest_scanned_checkpoint ?? "-"}`
    : "Not used in direct bot mode";
  renderDriveOverview();
}

async function refreshStatus(logResult = false) {
  try {
    const status = await requestJson("/mission-status");
    renderMissionStatus(status);
    if (logResult) {
      appendLog("GET /mission-status", status);
    }
    return status;
  } catch (error) {
    setConnection(false);
    elements.missionDetail.textContent = String(error.message);
    if (logResult) {
      appendLog("GET /mission-status failed", error.payload || error.message, true);
    }
    return null;
  }
}

function normalizeCheckpoints(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }
  const value = payload.checkpoints_list ?? payload.checkpoints ?? payload.data ?? [];
  if (Array.isArray(value)) {
    return value;
  }
  return value.checkpoints_list ?? value.checkpoints ?? [];
}

function renderCheckpoints(payload) {
  const checkpoints = normalizeCheckpoints(payload);
  const latest = Number(payload.latest_scanned_checkpoint ?? 0);
  elements.checkpointList.replaceChildren();
  if (!checkpoints.length) {
    const item = document.createElement("li");
    item.textContent = "No checkpoint data.";
    elements.checkpointList.append(item);
    renderMissionMap([], latest);
    return;
  }
  checkpoints.forEach((checkpoint, index) => {
    const item = document.createElement("li");
    const sequence = checkpoint.sequence ?? checkpoint.order ?? index + 1;
    const latitude = checkpoint.latitude ?? checkpoint.lat ?? "-";
    const longitude = checkpoint.longitude ?? checkpoint.lon ?? "-";
    item.textContent = `#${sequence}  ${latitude}, ${longitude}`;
    elements.checkpointList.append(item);
  });
  renderMissionMap(checkpoints, latest);
}

function validCoordinate(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function initMissionMap() {
  if (state.map || typeof window.L === "undefined") {
    return;
  }
  state.map = L.map("mission-map", { zoomControl: true }).setView([0, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 20,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  state.trailLayer = L.polyline([], {
    color: "#d85cff",
    weight: 5,
    opacity: 1.0,
  }).addTo(state.map);
}

function checkpointIcon(label, statusClass) {
  return L.divIcon({
    className: "",
    html: `<div class="checkpoint-map-icon ${statusClass}">${label}</div>`,
    iconSize: [30, 25],
    iconAnchor: [15, 12],
  });
}

function renderMissionMap(checkpoints, latest = 0) {
  initMissionMap();
  if (!state.map) {
    elements.mapSummary.textContent = "Map library unavailable";
    return;
  }
  state.checkpointLayers.forEach((layer) => state.map.removeLayer(layer));
  state.checkpointLayers = [];
  if (state.routeLayer) {
    state.map.removeLayer(state.routeLayer);
    state.routeLayer = null;
  }

  const ordered = checkpoints
    .map((checkpoint, index) => ({
      checkpoint,
      sequence: Number(checkpoint.sequence ?? index + 1),
      lat: validCoordinate(checkpoint.latitude ?? checkpoint.lat),
      lon: validCoordinate(checkpoint.longitude ?? checkpoint.lon),
    }))
    .filter((item) => item.lat !== null && item.lon !== null)
    .sort((a, b) => a.sequence - b.sequence);
  state.checkpointRoute = ordered;
  state.latestCheckpoint = latest;

  if (!ordered.length) {
    elements.mapSummary.textContent = "Waiting for mission checkpoints";
    return;
  }

  const grouped = new Map();
  ordered.forEach((item) => {
    const key = `${item.lat.toFixed(7)},${item.lon.toFixed(7)}`;
    const group = grouped.get(key) || [];
    group.push(item);
    grouped.set(key, group);
  });
  grouped.forEach((items) => {
    const sequences = items.map((item) => item.sequence);
    const reached = sequences.every((sequence) => sequence <= latest);
    const next = sequences.includes(latest + 1);
    const marker = L.marker([items[0].lat, items[0].lon], {
      icon: checkpointIcon(
        sequences.join("/"),
        reached ? "reached" : next ? "next" : "",
      ),
    }).addTo(state.map);
    marker.bindPopup(
      items.map((item) => `Checkpoint ${item.sequence}`).join("<br>"),
    );
    state.checkpointLayers.push(marker);
  });

  const route = ordered.map((item) => [item.lat, item.lon]);
  state.routeLayer = L.polyline(route, {
    color: "#36a3ff",
    weight: 4,
    opacity: 0.85,
    dashArray: "8 6",
  }).addTo(state.map);
  state.map.fitBounds(L.latLngBounds(route), { padding: [35, 35], maxZoom: 19 });
  state.mapHasView = true;
  elements.mapSummary.textContent =
    latest >= ordered.length
      ? `${ordered.length} checkpoints | mission route complete`
      : `${ordered.length} checkpoints | completed ${latest} | next ${latest + 1}`;
  window.setTimeout(() => state.map.invalidateSize(), 0);
}

function distanceMeters(lat1, lon1, lat2, lon2) {
  const radius = 6371000;
  const toRadians = (degrees) => degrees * Math.PI / 180;
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2))
    * Math.sin(dLon / 2) ** 2;
  return radius * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function bearingDegrees(lat1, lon1, lat2, lon2) {
  const toRadians = (degrees) => degrees * Math.PI / 180;
  const toDegrees = (radians) => radians * 180 / Math.PI;
  const firstLat = toRadians(lat1);
  const secondLat = toRadians(lat2);
  const deltaLon = toRadians(lon2 - lon1);
  const x = Math.sin(deltaLon) * Math.cos(secondLat);
  const y = Math.cos(firstLat) * Math.sin(secondLat)
    - Math.sin(firstLat) * Math.cos(secondLat) * Math.cos(deltaLon);
  return (toDegrees(Math.atan2(x, y)) + 360) % 360;
}

function normalizeHeadingError(degrees) {
  return ((degrees + 180) % 360 + 360) % 360 - 180;
}

function updateRoverMap(latitude, longitude, orientation) {
  initMissionMap();
  const lat = validCoordinate(latitude);
  const lon = validCoordinate(longitude);
  if (!state.map || lat === null || lon === null) {
    return;
  }
  const heading = validCoordinate(orientation) ?? 0;
  const icon = L.divIcon({
    className: "",
    html: `<div class="rover-map-icon" style="transform: rotate(${heading - 45}deg)"></div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
  if (!state.roverMarker) {
    state.roverMarker = L.marker([lat, lon], { icon, zIndexOffset: 1000 })
      .addTo(state.map)
      .bindPopup("Current rover position");
  } else {
    state.roverMarker.setLatLng([lat, lon]);
    state.roverMarker.setIcon(icon);
  }
  const previous = state.trail[state.trail.length - 1];
  const movedDistance = previous
    ? distanceMeters(previous[0], previous[1], lat, lon)
    : null;
  if (!previous || movedDistance >= TRAIL_MIN_DISTANCE_M) {
    state.trail.push([lat, lon]);
    if (state.trail.length > 2000) {
      state.trail.shift();
    }
    state.trailLayer.setLatLngs(state.trail);
  }
  if (!state.mapHasView) {
    state.map.setView([lat, lon], 18);
    state.mapHasView = true;
  }
  elements.mapPosition.textContent =
    `${lat.toFixed(6)}, ${lon.toFixed(6)} | ${heading.toFixed(0)}°`;
  const target = state.checkpointRoute.find(
    (item) => item.sequence > state.latestCheckpoint,
  );
  if (target) {
    const distance = distanceMeters(lat, lon, target.lat, target.lon);
    const bearing = bearingDegrees(lat, lon, target.lat, target.lon);
    const headingError = normalizeHeadingError(bearing - heading);
    const navigationReached = Boolean(
      state.samNavigation?.reached
      && Number(state.samNavigation?.target_sequence) === target.sequence
    );
    if (navigationReached) {
      if (state.globalPathLayer) {
        state.globalPathLayer.setLatLngs([]);
      }
      elements.mapSummary.textContent =
        `${state.checkpointRoute.length} checkpoints | checkpoint ${target.sequence}`
        + ` reached (${distance.toFixed(1)} m) | waiting for report`;
      return;
    }
    const shortestPath = [[lat, lon], [target.lat, target.lon]];
    if (!state.globalPathLayer) {
      state.globalPathLayer = L.polyline(shortestPath, {
        color: "#35d6c1",
        weight: 5,
        opacity: 0.95,
      }).addTo(state.map);
    } else {
      state.globalPathLayer.setLatLngs(shortestPath);
    }
    elements.mapSummary.textContent =
      `${state.checkpointRoute.length} checkpoints | completed ${state.latestCheckpoint}`
      + ` | next ${target.sequence}: ${distance.toFixed(1)} m`
      + ` | bearing ${bearing.toFixed(1)}° | error ${headingError.toFixed(1)}°`;
  } else if (state.globalPathLayer) {
    state.globalPathLayer.setLatLngs([]);
  }
}

async function refreshMission(logResult = true) {
  const status = await refreshStatus(logResult);
  if (!status) {
    return;
  }
  // /mission-route is the side-effect-free cached route and already
  // includes latest_scanned_checkpoint -- it's populated once at
  // /start-mission and is all this periodic poll needs. Polling
  // /checkpoints-list here as well used to re-fetch from the cloud every
  // 2s while a mission was active and overwrite the server's in-memory
  // progress with whatever (or nothing) the cloud's checkpoints_list
  // response carries, which made an already-reported checkpoint appear to
  // reset back to 0 mid-mission and stall the route.
  try {
    const cachedRoute = await requestJson("/mission-route");
    if (cachedRoute.route_loaded) {
      renderCheckpoints(cachedRoute);
      if (logResult) {
        appendLog("GET /mission-route", cachedRoute);
      }
    }
  } catch (error) {
    appendLog("GET /mission-route failed", error.payload || error.message, true);
  }
}

async function runMissionAction(label, path, body = null) {
  if (state.missionRequestRunning) {
    return;
  }
  state.missionRequestRunning = true;
  setMissionControls(state.missionActive, state.missionConfigured);
  try {
    const options = { method: "POST" };
    if (body) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    const result = await requestJson(path, options);
    appendLog(`${label} ${path}`, result);
    const checkpointPayload = result.checkpoints ?? result.checkpoints_list;
    if (checkpointPayload) {
      renderCheckpoints(checkpointPayload);
    }
  } catch (error) {
    appendLog(`${label} ${path} failed`, error.payload || error.message, true);
  } finally {
    state.missionRequestRunning = false;
    await refreshMission(false);
  }
}

async function getSelectedMission() {
  const missionSlug = elements.missionSlug.value.trim();
  if (!missionSlug) {
    await refreshMission(true);
    return;
  }
  await runMissionAction("Get Mission", "/select-mission", {
    mission_slug: missionSlug,
  });
}

function valueOrDash(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

function epochAgeLabel(timestamp) {
  const value = Number(timestamp);
  if (!Number.isFinite(value)) {
    return "age -";
  }
  const age = Date.now() / 1000 - value;
  if (!Number.isFinite(age)) {
    return "age invalid";
  }
  if (age < -5) {
    return `clock mismatch ${age.toFixed(1)}s`;
  }
  return `age ${Math.max(0, age).toFixed(1)}s`;
}

async function pollTelemetry() {
  if (
    !state.cameraAndTelemetryAllowed
    || state.telemetryRequestRunning
  ) {
    return;
  }
  state.telemetryRequestRunning = true;
  try {
    const payload = await requestJson("/data");
    const data = payload.data && typeof payload.data === "object" ? payload.data : payload;
    const gps = data.gps && typeof data.gps === "object" ? data.gps : {};
    state.latestTelemetry = {
      ...data,
      latitude: data.latitude ?? gps.latitude ?? gps.lat,
      longitude: data.longitude ?? gps.longitude ?? gps.lon,
      orientation: data.orientation ?? data.heading,
      rpms: data.rpms ?? data.rpm,
      timestamp: data.timestamp ?? payload.timestamp,
    };
    state.latestTelemetryReceivedAt = Date.now() / 1000;
    state.telemetryError = null;
    document.querySelector("#latitude").textContent = valueOrDash(data.latitude ?? gps.latitude ?? gps.lat);
    document.querySelector("#longitude").textContent = valueOrDash(data.longitude ?? gps.longitude ?? gps.lon);
    document.querySelector("#orientation").textContent = valueOrDash(data.orientation ?? data.heading);
    document.querySelector("#speed").textContent = valueOrDash(data.speed);
    document.querySelector("#battery").textContent = valueOrDash(data.battery);
    document.querySelector("#signal").textContent = valueOrDash(data.signal_level ?? data.signal);
    document.querySelector("#gps-signal").textContent = valueOrDash(data.gps_signal ?? gps.signal);
    document.querySelector("#rpms").textContent = valueOrDash(data.rpms ?? data.rpm);
    updateRoverMap(
      data.latitude ?? gps.latitude ?? gps.lat,
      data.longitude ?? gps.longitude ?? gps.lon,
      data.orientation ?? data.heading,
    );
    state.telemetryReady = true;
    elements.telemetryTime.textContent =
      `${nowLabel()} | ${epochAgeLabel(data.timestamp ?? payload.timestamp)}`
      + (payload.server_timestamp
        ? ` | server ${epochAgeLabel(payload.server_timestamp)}`
        : "");
    renderDriveOverview();
  } catch (error) {
    state.telemetryError = error.message;
    elements.telemetryTime.textContent = `Error: ${error.message}`;
    renderDriveOverview();
  } finally {
    state.telemetryRequestRunning = false;
  }
}

async function pollCamera() {
  if (
    !state.cameraAndTelemetryAllowed
    || (state.cameraView === "sam-tp" && !state.samTpStatusInitialized)
    || state.cameraRequestRunning
  ) {
    return;
  }
  state.cameraRequestRunning = true;
  const started = performance.now();
  try {
    if (state.cameraView === "sam-tp" && state.samTpAvailable) {
      elements.camera.src = `${SAM_TP_OVERLAY_PATH}?t=${Date.now()}`;
      elements.camera.style.display = "block";
      elements.cameraPlaceholder.style.display = "none";
      elements.cameraState.textContent = "SAM-TP Live";
      elements.cameraState.className = "status status-online";
      return;
    }
    const payload = await requestJson("/v2/front");
    const encoded = payload.front_frame ?? payload.image ?? payload.frame;
    if (!encoded) {
      throw new Error("front frame is missing");
    }
    elements.camera.src = `data:image/png;base64,${encoded}`;
    elements.camera.style.display = "block";
    elements.cameraPlaceholder.style.display = "none";
    elements.cameraState.textContent = "Live";
    elements.cameraState.className = "status status-online";
    state.frontCameraReady = true;
    elements.cameraMeta.textContent =
      `${epochAgeLabel(payload.timestamp)} | request ${(performance.now() - started).toFixed(0)} ms`;
  } catch (error) {
    elements.cameraState.textContent = "Frame error";
    elements.cameraState.className = "status status-offline";
    elements.cameraMeta.textContent = error.message;
  } finally {
    state.cameraRequestRunning = false;
  }
}

async function pollConnectionDiagnostics() {
  if (!state.roverConnected || state.diagnosticsRequestRunning) {
    return;
  }
  state.diagnosticsRequestRunning = true;
  try {
    const diagnostics = await requestJson("/connection-diagnostics");
    const page = diagnostics.browser?.page ?? {};
    state.remoteUserCount = Number(page.remoteUserCount ?? 0);
    state.telemetryReady = Boolean(page.telemetryPresent);
    state.frontCameraReady = Boolean(page.frontTrackReady || page.frontFramePresent);
    const ready = state.remoteUserCount > 0
      && state.telemetryReady
      && state.frontCameraReady;
    elements.connection.textContent = ready
      ? "Rover online"
      : "Bridge connected · waiting for rover";
    elements.connection.className =
      `status ${ready ? "status-online" : "status-idle"}`;
    if (!state.frontCameraReady && !state.samTpAvailable) {
      elements.cameraState.textContent = "Waiting rover";
      elements.cameraState.className = "status status-idle";
      elements.cameraMeta.textContent =
        `RTC remote users ${state.remoteUserCount} | front track not available`;
    }
    if (!state.telemetryReady) {
      elements.telemetryTime.textContent =
        `Waiting for RTM telemetry | remote users ${state.remoteUserCount}`;
    }
    pollTelemetry();
    pollCamera();
  } catch (error) {
    elements.connection.textContent = "Bridge diagnostics failed";
    elements.connection.className = "status status-offline";
    elements.cameraMeta.textContent = error.message;
  } finally {
    state.diagnosticsRequestRunning = false;
  }
}

function setCameraView(view) {
  state.cameraView = view === "sam-tp" ? "sam-tp" : "raw";
  elements.viewRaw.classList.toggle("selected", state.cameraView === "raw");
  elements.viewSamTp.classList.toggle("selected", state.cameraView === "sam-tp");
  elements.viewSamTp.disabled = !state.samTpAvailable;
  renderDriveOverview();
  pollCamera();
}

function updateSideSectorOverlay(sideSector) {
  const left = elements.sideSectorLeft;
  const right = elements.sideSectorRight;
  if (!left || !right) {
    return;
  }
  const sides = [
    [left, sideSector && sideSector.left, "LEFT"],
    [right, sideSector && sideSector.right, "RIGHT"],
  ];
  for (const [element, side, key] of sides) {
    if (!sideSector || !side || typeof side !== "object") {
      element.classList.remove("visible", "viable", "not-viable", "chosen");
      element.textContent = "";
      continue;
    }
    const viable = Boolean(side.viable);
    const chosen = sideSector.chosen === key;
    element.classList.add("visible");
    element.classList.toggle("viable", viable);
    element.classList.toggle("not-viable", !viable);
    element.classList.toggle("chosen", chosen);
    const composite = Number(side.composite);
    const lowPercentile = Number(side.low_percentile);
    element.textContent =
      `${key}${chosen ? " ★" : ""}`
      + ` composite ${Number.isFinite(composite) ? composite.toFixed(2) : "-"}`
      + ` low ${Number.isFinite(lowPercentile) ? lowPercentile.toFixed(2) : "-"}`
      + (viable ? "" : " UNSAFE");
  }
}

function updateRecoveryMetrics(recovery) {
  const element = elements.recoveryMetrics;
  if (!element) {
    return;
  }
  if (!recovery || typeof recovery !== "object" || !recovery.maneuver_phase) {
    element.hidden = true;
    element.textContent = "";
    return;
  }
  const elapsed = Number(recovery.recovery_elapsed_sec);
  const cooldown = Number(recovery.cooldown_remaining_sec);
  element.hidden = false;
  element.textContent =
    `recovery ${recovery.maneuver_phase}`
    + (recovery.maneuver_direction ? ` ${recovery.maneuver_direction}` : "")
    + (Number.isFinite(elapsed) ? ` | ${elapsed.toFixed(1)}s` : "")
    + (recovery.pulse_count_max
      ? ` | pulse ${recovery.pulse_count}/${recovery.pulse_count_max}`
      : "")
    + (recovery.safe_frame_confirm_required
      ? ` | safe ${recovery.safe_frame_confirm_count}/${recovery.safe_frame_confirm_required}`
      : "")
    + (recovery.direction_confirm_required
      ? ` | direction ${recovery.direction_confirm_count}/${recovery.direction_confirm_required}`
      : "")
    + (Number.isFinite(cooldown) && cooldown > 0 ? ` | cooldown ${cooldown.toFixed(1)}s` : "");
}

function finiteStatusNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function statusAgeSeconds(timestamp) {
  const value = finiteStatusNumber(timestamp);
  if (value === null) {
    return null;
  }
  const age = Date.now() / 1000 - value;
  return Number.isFinite(age) ? Math.max(0, age) : null;
}

function signed(value, digits = 1, suffix = "") {
  const number = finiteStatusNumber(value);
  if (number === null) {
    return "-";
  }
  return `${number > 0 ? "+" : ""}${number.toFixed(digits)}${suffix}`;
}

function setCompactState(element, label, tone) {
  if (!element) {
    return;
  }
  element.textContent = label;
  element.classList.remove("state-safe", "state-warning", "state-danger", "state-offline");
  element.classList.add(`state-${tone}`);
}

function pathDirection(heading) {
  if (heading === null) {
    return "Waiting for planner";
  }
  if (Math.abs(heading) <= 3) {
    return "STRAIGHT";
  }
  return heading > 0 ? "TURN RIGHT" : "TURN LEFT";
}

function renderCandidateStrip(planner) {
  const container = elements.candidateStrip;
  if (!container) {
    return;
  }
  const candidates = Array.isArray(planner?.candidate_scores)
    ? planner.candidate_scores
    : [];
  container.replaceChildren();
  if (!candidates.length) {
    const empty = document.createElement("span");
    empty.className = "empty-candidates";
    empty.textContent = "Candidate paths waiting";
    container.append(empty);
    return;
  }
  const selectedIndex = finiteStatusNumber(planner.selected_candidate_index);
  const pendingIndex = finiteStatusNumber(planner.candidate_switch_pending);
  candidates.forEach((candidate) => {
    const item = document.createElement("span");
    const index = finiteStatusNumber(candidate.index);
    const score = finiteStatusNumber(candidate.final_score);
    const heading = finiteStatusNumber(candidate.heading_deg);
    const rejected = Boolean(candidate.hard_rejected);
    item.className = "candidate-item";
    item.classList.toggle("selected", index !== null && index === selectedIndex);
    item.classList.toggle("pending", index !== null && index === pendingIndex);
    item.classList.toggle("rejected", rejected);
    item.style.setProperty(
      "--candidate-score",
      String(Math.max(0, Math.min(1, score ?? 0))),
    );
    item.textContent = heading === null ? "-" : `${heading > 0 ? "+" : ""}${heading.toFixed(0)}°`;
    const near = finiteStatusNumber(candidate.near_field);
    const traversability = finiteStatusNumber(candidate.traversability);
    item.title = rejected
      ? `${item.textContent}: rejected · ${candidate.reject_reason || "unknown"}`
      : `${item.textContent}: score ${score === null ? "-" : score.toFixed(2)}`
        + ` · near ${near === null ? "-" : near.toFixed(2)}`
        + ` · traversability ${traversability === null ? "-" : traversability.toFixed(2)}`;
    container.append(item);
  });
}

function normalizedRpms(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    value = Object.values(value);
  }
  if (!Array.isArray(value)) {
    value = value === null || value === undefined ? [] : [value];
  }
  return value
    .flatMap((item) => Array.isArray(item) ? item : [item])
    .map(finiteStatusNumber)
    .filter((item) => item !== null)
    .slice(-4);
}

const SVG_NS = "http://www.w3.org/2000/svg";

function normalizedImagePoints(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((point) => Array.isArray(point) && point.length >= 2)
    .map((point) => [finiteStatusNumber(point[0]), finiteStatusNumber(point[1])])
    .filter((point) => point[0] !== null && point[1] !== null);
}

function appendSvgShape(layer, tag, attributes, titleText = "") {
  const shape = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([name, value]) => shape.setAttribute(name, value));
  if (titleText) {
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = titleText;
    shape.append(title);
  }
  layer.append(shape);
  return shape;
}

function renderPathGeometry(status, planner, stale) {
  const overlay = elements.pathGeometryOverlay;
  if (!overlay) {
    return;
  }
  elements.candidatePathLayer.replaceChildren();
  elements.selectedFootprintLayer.replaceChildren();
  elements.globalHeadingLayer.replaceChildren();

  const candidates = Array.isArray(planner?.candidate_scores)
    ? planner.candidate_scores
    : [];
  const visible = state.cameraView === "sam-tp"
    && Boolean(status?.ready)
    && candidates.some((candidate) => normalizedImagePoints(candidate.centerline_uv).length > 1);
  overlay.classList.toggle("visible", visible);
  overlay.classList.toggle("stale", Boolean(stale));
  if (!visible) {
    return;
  }

  const width = finiteStatusNumber(status.frame_width) ?? 1024;
  const height = finiteStatusNumber(status.frame_height) ?? 576;
  overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const selectedIndex = finiteStatusNumber(planner.selected_candidate_index);
  let selectedCandidate = null;
  candidates.forEach((candidate) => {
    const points = normalizedImagePoints(candidate.centerline_uv);
    if (points.length < 2) {
      return;
    }
    const index = finiteStatusNumber(candidate.index);
    const selected = index !== null && index === selectedIndex;
    if (selected) {
      selectedCandidate = candidate;
    }
    const shape = appendSvgShape(
      elements.candidatePathLayer,
      "polyline",
      { points: points.map((point) => point.join(",")).join(" ") },
      candidate.hard_rejected
        ? `Rejected ${signed(candidate.heading_deg, 0, " deg")}: ${candidate.reject_reason || "unsafe"}`
        : `Candidate ${signed(candidate.heading_deg, 0, " deg")}, score ${finiteStatusNumber(candidate.final_score)?.toFixed(2) ?? "-"}`,
    );
    shape.classList.add("candidate-path");
    shape.classList.toggle("rejected", Boolean(candidate.hard_rejected));
    shape.classList.toggle("selected", selected);
  });

  if (selectedCandidate) {
    const left = normalizedImagePoints(selectedCandidate.left_boundary_uv);
    const right = normalizedImagePoints(selectedCandidate.right_boundary_uv);
    if (left.length > 1 && right.length > 1) {
      appendSvgShape(
        elements.selectedFootprintLayer,
        "polygon",
        {
          points: [...left, ...right.slice().reverse()]
            .map((point) => point.join(","))
            .join(" "),
          class: "selected-footprint",
        },
        "Selected rover footprint",
      );
    }
  }

  const globalHeading = finiteStatusNumber(status.global_target_heading_error_deg);
  if (globalHeading !== null) {
    const clamped = Math.max(-60, Math.min(60, globalHeading));
    const radians = clamped * Math.PI / 180;
    const startX = width / 2;
    const startY = height * 0.94;
    const length = height * 0.34;
    const endX = startX + Math.sin(radians) * length;
    const endY = startY - Math.cos(radians) * length;
    appendSvgShape(
      elements.globalHeadingLayer,
      "line",
      {
        x1: startX,
        y1: startY,
        x2: endX,
        y2: endY,
        class: "global-heading-line",
      },
      `GPS heading error ${signed(globalHeading, 1, " deg")}`,
    );
    appendSvgShape(elements.globalHeadingLayer, "circle", {
      cx: endX,
      cy: endY,
      r: 6,
      class: "global-heading-tip",
    });
  }
}

function currentTelemetrySnapshot() {
  const now = Date.now() / 1000;
  const directAge = state.latestTelemetryReceivedAt === null
    ? null
    : Math.max(0, now - state.latestTelemetryReceivedAt);
  const directTimestamp = finiteStatusNumber(state.latestTelemetry?.timestamp);
  const directSourceAge = directTimestamp === null ? null : now - directTimestamp;
  const directCombinedAge = directAge === null || directSourceAge === null
    ? null
    : Math.max(directAge, directSourceAge);
  if (
    state.latestTelemetry
    && !state.telemetryError
    && directCombinedAge !== null
    && directSourceAge >= -1
    && directCombinedAge <= 2.5
  ) {
    return { data: state.latestTelemetry, age: directCombinedAge, fresh: true, source: "SDK" };
  }
  const sam = state.latestSamStatus;
  const samAge = finiteStatusNumber(sam?.telemetry_age_sec);
  const samFrameAge = statusAgeSeconds(
    sam?.frame_published_timestamp ?? sam?.published_timestamp,
  );
  if (
    sam?.state === "CLEAR"
    && sam?.telemetry_valid === true
    && sam?.telemetry
    && samAge !== null
    && samAge <= 2.5
    && samFrameAge !== null
    && samFrameAge <= 2.5
  ) {
    return { data: sam.telemetry, age: samAge, fresh: true, source: "SAM" };
  }
  const data = state.latestTelemetry ?? sam?.telemetry ?? {};
  return {
    data,
    age: directCombinedAge ?? directAge ?? samAge,
    fresh: false,
    source: state.telemetryError ? `ERROR: ${state.telemetryError}` : "STALE",
  };
}

function recordControlTrend(linear, sdkAngular, telemetry) {
  const now = Date.now() / 1000;
  const last = state.controlTrend.at(-1);
  if (last && now - last.time < 0.18) {
    return;
  }
  const rpms = normalizedRpms(telemetry.data.rpms ?? telemetry.data.rpm);
  const meanRpm = telemetry.fresh && rpms.length
    ? rpms.reduce((sum, value) => sum + Math.abs(value), 0) / rpms.length
    : null;
  const heading = telemetry.fresh
    ? finiteStatusNumber(telemetry.data.orientation ?? telemetry.data.heading)
    : null;
  state.controlTrend.push({ time: now, linear, angular: sdkAngular, rpm: meanRpm, heading });
  state.controlTrend = state.controlTrend.filter((sample) => now - sample.time <= 10);
}

function headingDeltaDegrees(value, reference) {
  if (value === null || reference === null) {
    return null;
  }
  return ((value - reference + 540) % 360) - 180;
}

function commandResponseStart(timestamp, now) {
  const requestTimestamp = finiteStatusNumber(timestamp);
  if (requestTimestamp === null || Math.abs(now - requestTimestamp) > 10) {
    return now;
  }
  return Math.min(now, requestTimestamp);
}

function drawControlTrend() {
  const canvas = elements.controlTrendCanvas;
  if (!canvas || typeof canvas.getContext !== "function") {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = Math.max(300, rect.width || 600);
  const cssHeight = Math.max(70, rect.height || 92);
  const width = Math.round(cssWidth * dpr);
  const height = Math.round(cssHeight * dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);
  context.strokeStyle = "#2f3639";
  context.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((ratio) => {
    context.beginPath();
    context.moveTo(0, cssHeight * ratio);
    context.lineTo(cssWidth, cssHeight * ratio);
    context.stroke();
  });
  const samples = state.controlTrend;
  if (samples.length < 2) {
    return;
  }
  const now = Date.now() / 1000;
  const headingReference = samples.find((sample) => sample.heading !== null)?.heading ?? null;
  const series = [
    { color: "#38d782", value: (sample) => sample.linear / 0.25 },
    { color: "#ffcf72", value: (sample) => sample.angular / 0.40 },
    { color: "#ef6b62", value: (sample) => sample.rpm === null ? null : sample.rpm / 30 },
    {
      color: "#28d7d1",
      value: (sample) => {
        const delta = headingDeltaDegrees(sample.heading, headingReference);
        return delta === null ? null : delta / 45;
      },
    },
  ];
  series.forEach((item) => {
    let drawing = false;
    context.beginPath();
    context.strokeStyle = item.color;
    context.lineWidth = 1.8;
    samples.forEach((sample) => {
      const value = item.value(sample);
      if (value === null || !Number.isFinite(value)) {
        drawing = false;
        return;
      }
      const x = Math.max(0, Math.min(cssWidth, cssWidth - (now - sample.time) / 10 * cssWidth));
      const y = cssHeight / 2 - Math.max(-1, Math.min(1, value)) * cssHeight * 0.42;
      if (!drawing) {
        context.moveTo(x, y);
        drawing = true;
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  });
}

function renderPathIntent() {
  const status = state.latestSamStatus;
  const ready = Boolean(status?.ready);
  const planner = status?.planner && typeof status.planner === "object"
    ? status.planner
    : {};
  const selectedHeading = finiteStatusNumber(
    planner.selected_candidate_heading_deg ?? status?.local_path_selected_heading_deg,
  );
  const globalHeading = finiteStatusNumber(status?.global_target_heading_error_deg);
  const score = finiteStatusNumber(
    planner.selected_candidate_score ?? status?.path_mean_score,
  );
  const planAge = finiteStatusNumber(planner.plan_age_sec ?? status?.plan_age_sec);
  const statusAge = statusAgeSeconds(
    status?.frame_published_timestamp ?? status?.published_timestamp,
  );
  const stale = ready && (status?.state !== "CLEAR" || statusAge === null || statusAge > 2.5);
  const trajectoryValid = Boolean(planner.trajectory_valid ?? status?.trajectory_valid ?? status?.path_valid);
  const nearSafe = Boolean(planner.near_field_safe ?? status?.near_field_safe);
  const switchStop = Boolean(planner.switch_stop_required);

  elements.pathDirectionLabel.textContent = pathDirection(selectedHeading);
  elements.pathSelectedHeading.textContent = signed(selectedHeading, 1, "°");
  elements.pathGlobalHeading.textContent = signed(globalHeading, 1, "°");
  elements.pathScore.textContent = score === null ? "-" : score.toFixed(3);
  elements.pathAge.textContent = planAge === null ? "-" : `${planAge.toFixed(2)} s`;
  elements.pathFrameId.textContent = valueOrDash(status?.frame_id ?? status?.frame_index);
  elements.pathPlanId.textContent = valueOrDash(status?.plan_id);
  elements.pathDirectionArrow.style.transform =
    `rotate(${Math.max(-45, Math.min(45, selectedHeading ?? 0))}deg)`;
  renderCandidateStrip(planner);
  renderPathGeometry(status, planner, stale);

  if (!ready) {
    setCompactState(elements.pathHealth, "OFFLINE", "offline");
  } else if (stale) {
    setCompactState(elements.pathHealth, "STALE", "danger");
  } else if (!trajectoryValid || !nearSafe || switchStop) {
    setCompactState(elements.pathHealth, "STOP", "danger");
  } else if (planner.using_held_plan) {
    setCompactState(elements.pathHealth, "HELD", "warning");
  } else {
    setCompactState(elements.pathHealth, "CLEAR", "safe");
  }

  const switchReason = planner.switch_reason;
  const pathReason = status?.last_error || status?.path_reason;
  elements.pathDecisionReason.textContent = !ready
    ? "Start SAM-TP to inspect local path decisions."
    : `frame ${status.frame_index ?? "-"}`
      + ` · ${switchReason || pathReason || "planner ready"}`
      + (statusAge === null ? "" : ` · status ${statusAge.toFixed(1)}s old`);
  elements.hudPath.textContent = ready
    ? stale
      ? `STALE PATH · ${valueOrDash(status?.frame_id ?? status?.frame_index)}`
      : `PATH ${signed(selectedHeading, 1, "°")} ${pathDirection(selectedHeading)}`
    : "PATH OFFLINE";
}

function renderCommandMotion() {
  const status = state.latestAutonomyStatus;
  const offline = !status || status.state === "OFFLINE";
  const linear = finiteStatusNumber(status?.linear) ?? 0;
  const angular = finiteStatusNumber(status?.angular) ?? 0;
  const sdkLinear = finiteStatusNumber(status?.sdk_linear) ?? linear;
  const sdkAngular = finiteStatusNumber(status?.sdk_angular);
  const transmitted = Boolean(status?.command_transmitted);
  const commandAccepted = status?.command_accepted === undefined
    ? transmitted
    : Boolean(status.command_accepted);
  const armed = Boolean(status?.armed);
  const movingCommand = Math.abs(linear) > 0.0001 || Math.abs(sdkAngular ?? angular) > 0.0001;
  const autonomyAge = statusAgeSeconds(status?.updated_timestamp);
  const sdkCommandAge = finiteStatusNumber(state.latestMissionStatus?.last_control_command_age_sec);
  const telemetry = currentTelemetrySnapshot();
  const speed = telemetry.fresh ? finiteStatusNumber(telemetry.data.speed) : null;
  const rpms = telemetry.fresh
    ? normalizedRpms(telemetry.data.rpms ?? telemetry.data.rpm)
    : [];
  const observedMotion = telemetry.fresh && ((speed !== null && Math.abs(speed) > 0.03)
    || rpms.some((rpm) => Math.abs(rpm) > 1.0));

  let mode = "STOP";
  if (linear > 0.0001 && Math.abs(sdkAngular ?? angular) <= 0.0001) {
    mode = "FORWARD";
  } else if (Math.abs(sdkAngular ?? angular) > 0.0001) {
    mode = (sdkAngular ?? -angular) > 0 ? "TURN LEFT" : "TURN RIGHT";
  } else if (linear < -0.0001) {
    mode = "REVERSE";
  }

  const now = Date.now() / 1000;
  if (!offline && armed && movingCommand && commandAccepted) {
    if (
      !state.pendingMotionCommand
      || state.pendingMotionCommand.expect !== "MOTION"
      || state.pendingMotionCommand.mode !== mode
    ) {
      state.pendingMotionCommand = {
        id: status.command_id ?? "unidentified",
        mode,
        expect: "MOTION",
        startedAt: commandResponseStart(status.command_request_timestamp, now),
      };
    }
    if (observedMotion) {
      state.lastMotionResponse = {
        id: state.pendingMotionCommand.id,
        kind: "MOTION",
        delay: Math.max(0, now - state.pendingMotionCommand.startedAt),
      };
      state.pendingMotionCommand = null;
    }
    state.lastAcceptedCommandMode = mode;
  } else if (!offline && armed && !movingCommand && commandAccepted) {
    const stopNeedsConfirmation = observedMotion
      || state.pendingMotionCommand?.expect === "STOP"
      || (state.lastAcceptedCommandMode && state.lastAcceptedCommandMode !== "STOP");
    if (stopNeedsConfirmation) {
      if (state.pendingMotionCommand?.expect !== "STOP") {
        state.pendingMotionCommand = {
          id: status.command_id ?? "unidentified",
          mode: "STOP",
          expect: "STOP",
          startedAt: commandResponseStart(status.command_request_timestamp, now),
        };
      }
      if (telemetry.fresh && !observedMotion) {
        state.lastMotionResponse = {
          id: state.pendingMotionCommand.id,
          kind: "STOP",
          delay: Math.max(0, now - state.pendingMotionCommand.startedAt),
        };
        state.pendingMotionCommand = null;
      }
    } else {
      state.pendingMotionCommand = null;
    }
    state.lastAcceptedCommandMode = "STOP";
  } else if (offline || !armed || !commandAccepted) {
    state.pendingMotionCommand = null;
    if (offline || !armed) {
      state.lastAcceptedCommandMode = null;
    }
  }
  const pendingAge = state.pendingMotionCommand
    ? Math.max(0, now - state.pendingMotionCommand.startedAt)
    : null;
  const noResponse = telemetry.fresh && pendingAge !== null && pendingAge > 1.5;
  const planMismatch = Boolean(
    status?.plan_id
    && state.latestSamStatus?.plan_id
    && status.plan_id !== state.latestSamStatus.plan_id,
  );
  const sourcePlanAge = statusAgeSeconds(status?.frame_published_timestamp);
  const planSourceStale = planMismatch && (sourcePlanAge === null || sourcePlanAge > 2.5);

  elements.commandMode.textContent = offline
    ? "Autonomy offline"
    : `${mode} · ${String(status.state || "UNKNOWN").replaceAll("_", " ")}`;
  elements.internalLinear.textContent = linear.toFixed(3);
  elements.internalAngular.textContent = angular.toFixed(3);
  elements.sdkLinear.textContent = sdkLinear.toFixed(3);
  elements.sdkAngular.textContent = sdkAngular === null ? "-" : sdkAngular.toFixed(3);
  const maxLinear = finiteStatusNumber(status?.control_limits?.linear_max) ?? 0.25;
  const maxAngular = finiteStatusNumber(status?.control_limits?.angular_max) ?? 0.40;
  elements.linearGaugeFill.style.width =
    `${Math.max(0, Math.min(100, Math.abs(linear) / maxLinear * 100))}%`;
  const angularForGauge = sdkAngular ?? 0;
  elements.angularGaugeMarker.style.left =
    `${50 + Math.max(-50, Math.min(50, angularForGauge / maxAngular * 50))}%`;
  elements.commandTransmit.textContent = offline
    ? "-"
    : commandAccepted ? "HTTP ACCEPTED" : armed ? "NOT SENT" : "DRY RUN";
  elements.commandAge.textContent = autonomyAge === null
    ? "-"
    : `${autonomyAge.toFixed(2)} s`
      + (sdkCommandAge === null ? "" : ` · SDK ${sdkCommandAge.toFixed(2)} s`);
  elements.motionSpeed.textContent = speed === null ? "-" : speed.toFixed(3);
  elements.motionRpms.textContent = rpms.length
    ? rpms.map((rpm) => rpm.toFixed(1)).join(" / ")
    : telemetry.fresh ? "-" : "STALE";
  elements.commandId.textContent = valueOrDash(status?.command_id);
  const latency = finiteStatusNumber(status?.command_response_latency_ms);
  elements.commandLatency.textContent = latency === null ? "-" : `${latency.toFixed(0)} ms`;
  elements.motionAge.textContent = telemetry.age === null
    ? telemetry.source
    : `${telemetry.age.toFixed(2)} s · ${telemetry.source}`;
  if (!telemetry.fresh) {
    elements.motionResponse.textContent = "TELEMETRY STALE";
  } else if (noResponse) {
    elements.motionResponse.textContent =
      `${state.pendingMotionCommand.expect} NO RESPONSE ${pendingAge.toFixed(1)} s`;
  } else if (pendingAge !== null) {
    elements.motionResponse.textContent = state.pendingMotionCommand.expect === "STOP"
      ? `STOPPING ${pendingAge.toFixed(1)} s`
      : `PENDING ${pendingAge.toFixed(1)} s`;
  } else if (state.lastMotionResponse && (observedMotion || !movingCommand)) {
    elements.motionResponse.textContent =
      `${state.lastMotionResponse.kind} ${state.lastMotionResponse.delay.toFixed(2)} s`;
  } else if (!movingCommand) {
    elements.motionResponse.textContent = observedMotion ? "COASTING" : "STOPPED";
  } else {
    elements.motionResponse.textContent = "MOTION OBSERVED";
  }

  recordControlTrend(linear, sdkAngular ?? angular, telemetry);
  drawControlTrend();

  if (offline) {
    setCompactState(elements.commandHealth, "OFFLINE", "offline");
  } else if (armed && !commandAccepted && (status?.command_id || movingCommand)) {
    setCompactState(elements.commandHealth, "NOT SENT", "danger");
  } else if (armed && commandAccepted && !telemetry.fresh) {
    setCompactState(elements.commandHealth, "NO TELEMETRY", "danger");
  } else if (noResponse) {
    setCompactState(elements.commandHealth, "NO RESPONSE", "danger");
  } else if (planSourceStale) {
    setCompactState(elements.commandHealth, "PLAN STALE", "danger");
  } else if (planMismatch) {
    setCompactState(elements.commandHealth, "PLAN LAG", "warning");
  } else if (commandAccepted && movingCommand && !observedMotion) {
    setCompactState(elements.commandHealth, "WAIT MOTION", "warning");
  } else if (state.pendingMotionCommand?.expect === "STOP") {
    setCompactState(elements.commandHealth, "STOPPING", "warning");
  } else if (commandAccepted) {
    setCompactState(elements.commandHealth, "SENT", "safe");
  } else if (!armed) {
    setCompactState(elements.commandHealth, "DRY RUN", "warning");
  } else {
    setCompactState(elements.commandHealth, "STOP", "safe");
  }

  elements.commandReason.textContent = offline
    ? "Start Mission1 autonomy to inspect commands."
    : (status.command_error || status.reason || "Controller status available")
      + (planMismatch
        ? ` · controller ${status.plan_id}, display ${state.latestSamStatus.plan_id}`
        : "");
  elements.hudCommand.textContent = offline
    ? "CONTROL OFFLINE"
    : `${mode} · SDK L ${sdkLinear.toFixed(2)} A ${sdkAngular === null ? "-" : sdkAngular.toFixed(2)}`;
}

function renderDriveOverview() {
  if (!elements.driveHud) {
    return;
  }
  renderPathIntent();
  renderCommandMotion();
  elements.driveHud.hidden = !(state.latestSamStatus?.ready || state.latestAutonomyStatus);
}

async function pollSamTpStatus() {
  const nowMs = Date.now();
  if (nowMs < state.samTpNextPollMs) {
    return;
  }
  if (state.samTpStatusRequestRunning) {
    return;
  }
  state.samTpStatusRequestRunning = true;
  try {
    const response = await fetch(SAM_TP_STATUS_PATH, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const status = await response.json();
    state.latestSamStatus = status;
    state.samTpNextPollMs = 0;
    state.samNavigation = status.navigation ?? null;
    state.samTpAvailable = Boolean(status.ready);
    elements.viewSamTp.disabled = !state.samTpAvailable;
    elements.samTpState.textContent = status.ready ? status.state : "SAM-TP loading";
    elements.samTpState.className =
      `status ${status.state === "CLEAR" ? "status-online" : "status-idle"}`;
    if (status.ready) {
      if (!state.samTpAutoSelected) {
        state.samTpAutoSelected = true;
        setCameraView("sam-tp");
      }
      const globalHeading = Number(status.global_target_heading_error_deg);
      const localHeading = Number(status.local_path_selected_heading_deg);
      const residual = Number(status.local_path_heading_residual_deg);
      const targetBearing = Number(status.navigation?.target_bearing_deg);
      const roverHeading = Number(status.navigation?.current_heading_deg);
      const positionReason = status.localization?.position_status_reason;
      const headingReason = status.localization?.heading_status_reason;
      const localizationMetrics = positionReason || headingReason
        ? ` | GPS ${positionReason || "unknown"} / heading ${headingReason || "unknown"}`
        : "";
      const guidanceMetrics = status.global_target_heading_error_deg !== null
        && status.global_target_heading_error_deg !== undefined
        && Number.isFinite(globalHeading)
        && status.local_path_selected_heading_deg !== null
        && status.local_path_selected_heading_deg !== undefined
        && Number.isFinite(localHeading)
        && status.local_path_heading_residual_deg !== null
        && status.local_path_heading_residual_deg !== undefined
        && Number.isFinite(residual)
        ? ` | global ${globalHeading.toFixed(1)}° → local ${localHeading.toFixed(1)}°`
          + ` | residual ${residual.toFixed(1)}°`
        : " | GPS path waiting";
      elements.samTpMetrics.textContent =
        `frame ${status.frame_index} | infer ${Number(status.inference_latency_ms).toFixed(0)} ms`
        + ` | e2e ${Number(status.end_to_end_latency_ms).toFixed(0)} ms`
        + ` | ${Number(status.effective_fps).toFixed(1)} FPS`
        + ` | score ${Number(status.score_mean).toFixed(3)}`
        + ` | path ${status.path_valid ? "valid" : status.path_reason}`
        + (status.local_path_goal_alignment_weight === null
          || status.local_path_goal_alignment_weight === undefined
          ? ""
          : ` | goal weight ${Number(status.local_path_goal_alignment_weight).toFixed(2)}`)
        + (status.local_path_smoothing_method
          ? ` | spline ${status.local_path_smoothing_applied ? "active" : "safe fallback"}`
          : "")
        + guidanceMetrics
        + (Number.isFinite(targetBearing) && Number.isFinite(roverHeading)
          ? ` | bearing ${targetBearing.toFixed(1)}° / rover ${roverHeading.toFixed(1)}°`
          : "")
        + localizationMetrics
        + (status.sdk_clock_offset_hours === null
          || status.sdk_clock_offset_hours === undefined
          ? ""
          : ` | camera clock ${status.sdk_clock_offset_hours > 0 ? "+" : ""}`
            + `${status.sdk_clock_offset_hours}h corrected`);
      if (state.cameraView === "sam-tp" && state.cameraAndTelemetryAllowed) {
        elements.cameraMeta.textContent =
          `SAM-TP perception | frame ${status.frame_index} | controller shown separately`;
        pollCamera();
      }
    }
    // side_sector is only present when planner.side_sector_enabled is on
    // for the active profile (fail-closed default) -- absent elsewhere.
    updateSideSectorOverlay(status.planner && status.planner.side_sector);
    renderDriveOverview();
  } catch (_error) {
    state.samTpNextPollMs = Date.now() + 5000;
    state.samTpAvailable = false;
    state.samNavigation = null;
    elements.viewSamTp.disabled = true;
    elements.samTpState.textContent = "SAM-TP offline";
    elements.samTpState.className = "status status-idle";
    elements.samTpMetrics.textContent =
      "Start the SAM-TP shadow process to enable the overlay.";
    state.latestSamStatus = null;
    updateSideSectorOverlay(null);
    renderDriveOverview();
    if (state.cameraView === "sam-tp") {
      setCameraView("raw");
    }
  } finally {
    state.samTpStatusInitialized = true;
    state.samTpStatusRequestRunning = false;
  }
}

async function pollAutonomyStatus() {
  const nowMs = Date.now();
  if (nowMs < state.autonomyNextPollMs) {
    return;
  }
  if (state.autonomyStatusRequestRunning) {
    return;
  }
  state.autonomyStatusRequestRunning = true;
  try {
    const response = await fetch(AUTONOMY_STATUS_PATH, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const status = await response.json();
    state.latestAutonomyStatus = status;
    state.autonomyNextPollMs = 0;
    const linear = Number(status.linear);
    const angular = Number(status.angular);
    const driving = status.state === "DRIVING"
      || status.state === "STG_DRIVE_STRAIGHT"
      || (Number.isFinite(linear) && Math.abs(linear) > 0.0001)
      || (Number.isFinite(angular) && Math.abs(angular) > 0.0001);
    const complete = status.state === "MISSION_COMPLETE";
    elements.autonomyState.textContent = driving
      ? `AUTO ${linear.toFixed(2)} / ${angular.toFixed(2)}`
      : complete ? "Mission complete" : status.state.replaceAll("_", " ");
    elements.autonomyState.title = status.reason || "";
    elements.autonomyState.className =
      `status ${driving || complete ? "status-online" : "status-idle"}`;
    // status.recovery is only present while a ROTATE_ESCAPE recovery is
    // active (or was just aborted/cooling down) -- absent during normal
    // driving, so this stays hidden and the existing dashboard is unchanged.
    updateRecoveryMetrics(status.recovery);
    renderDriveOverview();
  } catch (_error) {
    state.autonomyNextPollMs = Date.now() + 5000;
    elements.autonomyState.textContent = "Autonomy offline";
    elements.autonomyState.title = "Start scripts/run_mission1_autonomy.sh";
    elements.autonomyState.className = "status status-idle";
    state.latestAutonomyStatus = null;
    updateRecoveryMetrics(null);
    renderDriveOverview();
  } finally {
    state.autonomyStatusRequestRunning = false;
  }
}

async function runAutonomyAction(label, path) {
  try {
    const target = path === "/resume" ? AUTONOMY_RESUME_PATH : AUTONOMY_STOP_PATH;
    const response = await fetch(target, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    appendLog(label, payload);
    await pollAutonomyStatus();
    return true;
  } catch (error) {
    appendLog(`${label} failed`, error.message, true);
    return false;
  }
}

async function tryAutonomyStopBeforeEndMission() {
  try {
    const response = await fetch(AUTONOMY_STOP_PATH, { method: "POST" });
    if (!response.ok) {
      return false;
    }
    const payload = await response.json();
    appendLog("EMERGENCY STOP before End Mission", payload);
    await pollAutonomyStatus();
    return true;
  } catch (_error) {
    return false;
  }
}

function startMissionPolling() {
  pollConnectionDiagnostics();
  if (state.diagnosticsTimer === null) {
    state.diagnosticsTimer = window.setInterval(pollConnectionDiagnostics, 2000);
  }
  if (state.telemetryTimer === null) {
    pollTelemetry();
    state.telemetryTimer = window.setInterval(pollTelemetry, 1000);
  }
  if (state.cameraTimer === null) {
    pollCamera();
    state.cameraTimer = window.setInterval(pollCamera, 250);
  }
}

function stopMissionPolling() {
  if (state.diagnosticsTimer !== null) {
    clearInterval(state.diagnosticsTimer);
    state.diagnosticsTimer = null;
  }
  state.telemetryReady = false;
  state.latestTelemetry = null;
  state.latestTelemetryReceivedAt = null;
  state.telemetryError = "telemetry polling stopped";
  state.frontCameraReady = false;
  state.remoteUserCount = 0;
  if (state.telemetryTimer !== null) {
    clearInterval(state.telemetryTimer);
    state.telemetryTimer = null;
  }
  if (state.cameraTimer !== null) {
    clearInterval(state.cameraTimer);
    state.cameraTimer = null;
  }
  renderDriveOverview();
}

elements.start.addEventListener("click", () => {
  const missionSlug = elements.missionSlug.value.trim();
  if (!missionSlug) {
    appendLog("Start Mission failed", "Select a mission slug first", true);
    return;
  }
  runMissionAction("Start Mission", "/start-mission", {
    mission_slug: missionSlug,
  });
});
elements.connect.addEventListener("click", () => {
  runMissionAction("Connect Rover", "/connect-rover");
});
elements.refresh.addEventListener("click", getSelectedMission);
elements.end.addEventListener("click", () => {
  const missionSlug = elements.missionSlug.value.trim();
  if (!missionSlug) {
    appendLog("End Mission failed", "Select a mission slug first", true);
    return;
  }
  const prompt = state.missionActive
    ? `Stop autonomy now and end ${missionSlug}?`
    : `Reset ${missionSlug}'s stale cloud ride? This clears its current progress.`;
  if (window.confirm(prompt)) {
    tryAutonomyStopBeforeEndMission()
      .finally(() => runMissionAction("End Mission", "/end-mission", {
        mission_slug: missionSlug,
      }));
  }
});
elements.stopAutonomy.addEventListener("click", () => {
  runAutonomyAction("EMERGENCY STOP", "/stop");
});
window.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.repeat) {
    event.preventDefault();
    runAutonomyAction("EMERGENCY STOP (Space)", "/stop");
  }
});
elements.resumeAutonomy.addEventListener("click", () => {
  if (window.confirm("Resume autonomous control using the current LOCAL PATH?")) {
    runAutonomyAction("Resume Auto", "/resume");
  }
});
elements.disconnect.addEventListener("click", () => {
  if (window.confirm("Disconnect the rover? A stop command will be sent first.")) {
    state.cameraAndTelemetryAllowed = false;
    stopMissionPolling();
    runMissionAction("Disconnect Rover", "/disconnect-rover");
  }
});
elements.missionSlug.addEventListener("input", () => {
  setMissionControls(state.missionActive, state.missionConfigured);
});
elements.viewRaw.addEventListener("click", () => setCameraView("raw"));
elements.viewSamTp.addEventListener("click", () => {
  if (state.samTpAvailable) {
    setCameraView("sam-tp");
  }
});
document.querySelector("#clear-log").addEventListener("click", () => {
  elements.responseLog.replaceChildren();
});

initMissionMap();
refreshStatus(true);
state.statusTimer = window.setInterval(() => refreshMission(false), 2000);
pollSamTpStatus();
state.samTpStatusTimer = window.setInterval(pollSamTpStatus, 1000);
pollAutonomyStatus();
state.autonomyStatusTimer = window.setInterval(pollAutonomyStatus, 500);
