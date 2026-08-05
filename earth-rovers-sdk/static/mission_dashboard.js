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
  try {
    const cachedRoute = await requestJson("/mission-route");
    if (cachedRoute.route_loaded) {
      renderCheckpoints(cachedRoute);
    }
  } catch (_error) {
    // Older SDK servers may not expose the side-effect-free cached route.
  }
  if (!status.mission_configured || !status.mission_active) {
    return;
  }
  try {
    const checkpoints = await requestJson("/checkpoints-list");
    renderCheckpoints(checkpoints);
    if (logResult) {
      appendLog("GET /checkpoints-list", checkpoints);
    }
  } catch (error) {
    appendLog("GET /checkpoints-list failed", error.payload || error.message, true);
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
  } catch (error) {
    elements.telemetryTime.textContent = `Error: ${error.message}`;
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
  pollCamera();
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
  } catch (_error) {
    state.samTpNextPollMs = Date.now() + 5000;
    state.samTpAvailable = false;
    state.samNavigation = null;
    elements.viewSamTp.disabled = true;
    elements.samTpState.textContent = "SAM-TP offline";
    elements.samTpState.className = "status status-idle";
    elements.samTpMetrics.textContent =
      "Start the SAM-TP shadow process to enable the overlay.";
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
    state.autonomyNextPollMs = 0;
    const driving = status.state === "DRIVING";
    const complete = status.state === "MISSION_COMPLETE";
    elements.autonomyState.textContent = driving
      ? `AUTO ${Number(status.linear).toFixed(2)} / ${Number(status.angular).toFixed(2)}`
      : complete ? "Mission complete" : status.state.replaceAll("_", " ");
    elements.autonomyState.title = status.reason || "";
    elements.autonomyState.className =
      `status ${driving || complete ? "status-online" : "status-idle"}`;
  } catch (_error) {
    state.autonomyNextPollMs = Date.now() + 5000;
    elements.autonomyState.textContent = "Autonomy offline";
    elements.autonomyState.title = "Start scripts/run_mission1_autonomy.sh";
    elements.autonomyState.className = "status status-idle";
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
  const prompt = state.missionActive
    ? "Stop autonomy now and end the active mission?"
    : "Reset the selected mission's stale cloud ride? This clears its current progress.";
  if (window.confirm(prompt)) {
    tryAutonomyStopBeforeEndMission()
      .finally(() => runMissionAction("End Mission", "/end-mission"));
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
