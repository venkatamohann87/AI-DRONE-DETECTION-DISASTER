"""
Live Rescue Operations web application.

This dashboard deliberately reads survivor_detection_result.csv as its only
source for survivor locations. Start it before or during the notebook/video
run; the browser polls the CSV through the local API and moves existing map
markers as fresh coordinates arrive.

Usage from the deployment folder:
    python live_rescue_web_app.py

The server is local-only by default. To show it to devices on a trusted LAN,
explicitly opt in with --host 0.0.0.0 and protect that network appropriately.
All rescue-team actions are logged locally in simulation mode. This program
does not contact real rescue services or embed credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


# ============================================================
# NEW: LIVE RESCUE WEB APPLICATION CONFIGURATION
# ============================================================

# The detector's actual live data source.  The web dashboard never fabricates
# locations; every visible marker and ID label is read from this CSV.
LIVE_RESULTS_CSV = (
    r"survivor_detection_result.csv"  # Relative to the working directory or deployment folder
)
RESCUE_ACTION_LOG = "rescue_team_action_log.jsonl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

REQUIRED_COLUMNS = {
    "ID",
    "Confidence",
    "Latitude",
    "Longitude",
    "Priority",
    "Priority Score",
    "Confirmed",
    "Frames",
}

_ACTION_LOG_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_float(value: Any) -> float | None:
    """Return a finite number or None; never invent a coordinate."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "confirmed"
    }


def rescue_protocol(survivor: dict[str, Any]) -> dict[str, str]:
    """Return the operations protocol derived from the current risk level."""
    priority = survivor["priority"]
    confirmed = survivor["confirmed"]

    if priority == "CRITICAL":
        return {
            "level": "CRITICAL",
            "action": "Dispatch the nearest rescue and medical team immediately.",
            "notify": "Notify incident command, medical coordination, and evacuation support.",
            "follow_up": "Keep the team supplied with every material location update.",
        }

    if priority == "HIGH" or confirmed:
        return {
            "level": "HIGH",
            "action": "Issue an urgent rescue request and assign a field team.",
            "notify": "Notify the rescue coordinator and confirm team acknowledgement.",
            "follow_up": "Reconfirm visual evidence and monitor location changes.",
        }

    if priority == "MEDIUM":
        return {
            "level": "MEDIUM",
            "action": "Queue field verification and prepare rescue resources.",
            "notify": "Send the observation to the operations desk for triage.",
            "follow_up": "Escalate if confirmation, confidence, or risk increases.",
        }

    return {
        "level": "LOW",
        "action": "Monitor the location and schedule another visual pass.",
        "notify": "Record the observation for the operations log.",
        "follow_up": "Escalate when confidence, confirmation, or priority changes.",
    }


def normalise_survivor(row: dict[str, Any]) -> dict[str, Any] | None:
    """Safely turn one CSV row into data that can be sent to the browser."""
    survivor_id = str(row.get("ID", "")).strip()
    latitude = _finite_float(row.get("Latitude"))
    longitude = _finite_float(row.get("Longitude"))
    confidence = _finite_float(row.get("Confidence"))

    if (
        not survivor_id
        or latitude is None
        or longitude is None
        or confidence is None
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        return None

    priority_score = _finite_float(row.get("Priority Score"))
    frames = _finite_float(row.get("Frames"))
    last_frame = _finite_float(row.get("Last Frame"))
    priority = str(row.get("Priority", "LOW")).strip().upper() or "LOW"

    survivor = {
        "id": survivor_id,
        "confidence": confidence,
        "latitude": latitude,
        "longitude": longitude,
        "priority": priority,
        "priority_score": priority_score if priority_score is not None else 0.0,
        "confirmed": _to_bool(row.get("Confirmed", False)),
        "frames": int(frames) if frames is not None else 0,
        "last_frame": int(last_frame) if last_frame is not None else None,
        "last_updated": str(row.get("Last Updated", "")).strip(),
    }
    survivor["protocol"] = rescue_protocol(survivor)
    return survivor


def resolve_live_csv_path(configured_path: Path) -> Path:
    """Choose the newest valid live CSV when the app is run from either folder.

    The notebook runs the detector from the project root, whereas this server
    may also be started directly from ``deployment``.  Checking both locations
    prevents a dashboard from silently watching an empty sibling CSV.
    """
    if configured_path.is_absolute():
        return configured_path

    candidates = [
        Path.cwd() / configured_path,
        SCRIPT_DIRECTORY / configured_path,
        SCRIPT_DIRECTORY.parent / configured_path,
    ]
    unique_candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    existing = [path for path in unique_candidates if path.is_file()]

    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime_ns)

    # Keep the configured working-directory target in the response while the
    # detector has not yet created its first CSV snapshot.
    return unique_candidates[0]


def read_live_survivors(csv_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read a stable CSV snapshot and keep only valid current survivor rows."""
    if not csv_path.is_file():
        return [], f"Waiting for {csv_path.name}"

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or [])

            missing = REQUIRED_COLUMNS.difference(fieldnames)
            if missing:
                return [], "CSV is missing: " + ", ".join(sorted(missing))

            # Last row wins if a third-party CSV writer appends an update for
            # an ID. The supplied detector publishes one row per ID.
            survivors_by_id: dict[str, dict[str, Any]] = {}

            for row in reader:
                survivor = normalise_survivor(row)
                if survivor is not None:
                    survivors_by_id[survivor["id"]] = survivor

    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return [], f"CSV read error: {error}"

    priority_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    survivors = sorted(
        survivors_by_id.values(),
        key=lambda item: (
            priority_rank.get(item["priority"], 4),
            not item["confirmed"],
            -item["confidence"],
            item["id"],
        ),
    )
    return survivors, None


def append_rescue_action(log_path: Path, action: dict[str, Any]) -> None:
    """Store simulation-mode actions locally as newline-delimited JSON."""
    with _ACTION_LOG_LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(action, ensure_ascii=False) + "\n")


def load_recent_actions(log_path: Path, limit: int = 12) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []

    try:
        with _ACTION_LOG_LOCK:
            lines = log_path.read_text(encoding="utf-8").splitlines()

        actions = []
        for line in lines[-limit:]:
            try:
                actions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(actions))
    except OSError:
        return []


class LiveRescueApplication:
    """Owns the CSV location and safe, local rescue-action recording."""

    def __init__(self, csv_path: Path, action_log_path: Path) -> None:
        self.csv_path = csv_path
        self.action_log_path = action_log_path

    def survivors_payload(self) -> dict[str, Any]:
        active_csv_path = resolve_live_csv_path(self.csv_path)
        survivors, error = read_live_survivors(active_csv_path)
        return {
            "updated_at": _utc_now(),
            "source": str(active_csv_path),
            "survivors": survivors,
            "error": error,
        }

    def record_action(self, request_data: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        survivor_id = str(request_data.get("survivor_id", "")).strip()
        action_name = str(request_data.get("action", "")).strip().upper()
        operator = str(request_data.get("operator", "Operations dashboard")).strip()
        allowed_actions = {"ACKNOWLEDGE", "DISPATCH", "REQUEST_UPDATE"}

        if not survivor_id:
            return None, "A survivor ID is required."
        if action_name not in allowed_actions:
            return None, "Unsupported rescue action."

        active_csv_path = resolve_live_csv_path(self.csv_path)
        survivors, error = read_live_survivors(active_csv_path)
        if error:
            return None, error

        survivor = next(
            (item for item in survivors if item["id"] == survivor_id),
            None,
        )
        if survivor is None:
            return None, "Survivor is not present in the latest valid CSV snapshot."

        event = {
            "timestamp": _utc_now(),
            "mode": "SIMULATION",
            "event_type": "RESCUE_TEAM_ACTION",
            "action": action_name,
            "operator": operator or "Operations dashboard",
            "survivor": survivor,
            "message": "Action recorded locally; no real rescue service was contacted.",
        }

        try:
            append_rescue_action(self.action_log_path, event)
        except OSError as error:
            return None, f"Could not write rescue action log: {error}"

        return event, None


class LiveRescueRequestHandler(BaseHTTPRequestHandler):
    """Serve the dashboard and same-origin JSON API."""

    application: LiveRescueApplication

    server_version = "LiveRescueDashboard/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        # Keep normal polling out of the terminal; errors are returned as JSON.
        return

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _send_html(self) -> None:
        response = DASHBOARD_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:  # noqa: N802 - method name required by BaseHTTPRequestHandler
        path = urlparse(self.path).path

        if path in {"/", "/index.html"}:
            self._send_html()
        elif path == "/api/survivors":
            self._send_json(self.application.survivors_payload())
        elif path == "/api/rescue-actions":
            self._send_json({"actions": load_recent_actions(self.application.action_log_path)})
        else:
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - method name required by BaseHTTPRequestHandler
        if urlparse(self.path).path != "/api/rescue-actions":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0

        if not 0 < content_length <= 16_384:
            self._send_json(
                {"error": "A small JSON request body is required."},
                HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            request_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Request body must be valid JSON."}, HTTPStatus.BAD_REQUEST)
            return

        if not isinstance(request_data, dict):
            self._send_json({"error": "Request body must be a JSON object."}, HTTPStatus.BAD_REQUEST)
            return

        event, error = self.application.record_action(request_data)
        if error:
            self._send_json({"error": error}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"ok": True, "event": event}, HTTPStatus.CREATED)


# This is a self-contained dashboard. Leaflet and OpenStreetMap tiles are
# loaded by the browser; the local server never sends survivor data elsewhere.
DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Rescue Operations</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <style>
    :root { color-scheme: dark; --panel:#101a2b; --panel2:#16243a; --line:#29425f; --text:#edf5ff; --muted:#94abc3; --accent:#2dd4bf; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 Inter, ui-sans-serif, system-ui, sans-serif; background:#08101d; color:var(--text); }
    header { min-height:64px; padding:12px 20px; display:flex; align-items:center; gap:16px; border-bottom:1px solid var(--line); background:#0d1726; }
    h1 { margin:0; font-size:18px; letter-spacing:.01em; }
    .subtitle { color:var(--muted); font-size:12px; }
    .connection { margin-left:auto; display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; }
    .pulse { width:9px; height:9px; border-radius:99px; background:#64748b; } .pulse.live { background:#2dd4bf; box-shadow:0 0 0 4px #2dd4bf22; } .pulse.error { background:#fb7185; }
    main { display:grid; grid-template-columns:minmax(420px, 1.35fr) minmax(340px, .9fr); height:calc(100vh - 64px); }
    #map { min-height:430px; height:100%; background:#0c1727; }
    aside { overflow:auto; padding:16px; border-left:1px solid var(--line); background:#0d1726; }
    .summary { display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-bottom:16px; }
    .count { padding:9px; border-radius:8px; background:var(--panel); border:1px solid var(--line); text-align:center; font-weight:700; }
    .count small { display:block; color:var(--muted); font-weight:500; font-size:10px; }
    .card { padding:13px; border:1px solid var(--line); border-left:4px solid #64748b; border-radius:8px; background:var(--panel); margin:10px 0; }
    .card.CRITICAL { border-left-color:#ef4444; } .card.HIGH { border-left-color:#f97316; } .card.MEDIUM { border-left-color:#38bdf8; } .card.LOW { border-left-color:#34d399; }
    .topline { display:flex; gap:8px; justify-content:space-between; align-items:start; } .topline h2 { font-size:15px; margin:0; }
    .risk { font-size:11px; font-weight:800; border-radius:99px; padding:3px 7px; background:#24364d; } .confirmed { color:#2dd4bf; font-weight:700; font-size:11px; }
    dl { display:grid; grid-template-columns:1fr 1fr; gap:5px 12px; margin:11px 0; } dt { color:var(--muted); } dd { margin:0; text-align:right; font-variant-numeric:tabular-nums; }
    .protocol { padding:9px; border-radius:6px; background:#0a1423; color:#d9e8f7; font-size:12px; } .protocol strong { color:#fff; }
    .buttons { display:flex; gap:7px; margin-top:11px; flex-wrap:wrap; } button { cursor:pointer; color:#ecfeff; background:#164e63; border:1px solid #2dd4bf88; border-radius:5px; padding:6px 8px; font-weight:650; } button:hover { background:#155e75; } button.dispatch { background:#9a3412; border-color:#fb923c; }
    .empty { color:var(--muted); padding:20px 5px; text-align:center; } .section { margin-top:20px; font-size:12px; font-weight:800; letter-spacing:.08em; color:var(--muted); text-transform:uppercase; }
    .event { margin:7px 0; padding:9px; font-size:12px; border-radius:6px; background:var(--panel2); border:1px solid var(--line); } .event time { color:var(--muted); }
    .data-source { margin:8px 0; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
    .dataset-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:7px; background:var(--panel); }
    .dataset { width:100%; border-collapse:collapse; font-size:11px; font-variant-numeric:tabular-nums; }
    .dataset th, .dataset td { padding:7px 8px; white-space:nowrap; text-align:right; border-bottom:1px solid #213650; }
    .dataset th { position:sticky; top:0; background:#16243a; color:#b9cadb; text-align:right; } .dataset th:first-child, .dataset td:first-child { text-align:left; }
    .dataset tr:last-child td { border-bottom:0; }
    .location-pin { width:34px; height:34px; border-radius:50% 50% 50% 0; transform:rotate(-45deg); color:#fff; display:grid; place-items:center; border:2px solid #fff; box-shadow:0 2px 6px #0009; }
    .location-pin span { transform:rotate(45deg); max-width:27px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:10px; font-weight:850; line-height:1; }
    .survivor-id-label { background:#07111eee; border:1px solid #eaf4ff; border-radius:4px; box-shadow:0 1px 5px #000c; color:#fff; font-size:12px; font-weight:850; padding:3px 6px; white-space:nowrap; }
    .survivor-id-label::before { display:none; }
    .pin-CRITICAL { background:#dc2626; } .pin-HIGH { background:#ea580c; } .pin-MEDIUM { background:#0284c7; } .pin-LOW { background:#059669; }
    .leaflet-popup-content-wrapper, .leaflet-popup-tip { background:#101a2b; color:#edf5ff; } .leaflet-popup-content { margin:12px 14px; }
    @media (max-width:850px) { main { display:block; height:auto; } #map { height:55vh; } aside { border-left:0; border-top:1px solid var(--line); } .summary { grid-template-columns:repeat(2,1fr); } }
  </style>
</head>
<body>
  <header>
    <div><h1>Live Rescue Operations</h1><div class="subtitle">CSV-backed survivor location and risk triage dashboard</div></div>
    <label class="connection"><input type="checkbox" id="auto-follow" checked> auto-fit new locations</label>
    <div class="connection"><i class="pulse" id="pulse"></i><span id="connection-text">Connecting…</span></div>
  </header>
  <main>
    <div id="map"></div>
    <aside>
      <div class="summary" id="summary"></div>
      <div id="survivor-list"><div class="empty">Waiting for valid survivor coordinates…</div></div>
      <div class="section">Live CSV dataset</div>
      <div class="data-source" id="data-source">Waiting for CSV source…</div>
      <div id="live-dataset"><div class="empty">No live rows available yet.</div></div>
      <div class="section">Rescue team action log</div>
      <div id="events"><div class="empty">No actions recorded in this dashboard session.</div></div>
    </aside>
  </main>
  <script>
    const map = L.map('map', { zoomControl: true }).setView([20, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(map);
    const markers = new Map();
    let initialFitComplete = false;
    const priorities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
    const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
    const format = (value, digits=2) => Number(value).toLocaleString(undefined, {minimumFractionDigits:digits, maximumFractionDigits:digits});
    const pinIcon = (priority, id) => L.divIcon({ className:'', iconSize:[34,34], iconAnchor:[17,34], popupAnchor:[0,-34], html:`<div class="location-pin pin-${escapeHtml(priority)}"><span>${escapeHtml(id)}</span></div>` });
    const popup = survivor => `<strong>Survivor ${escapeHtml(survivor.id)}</strong><br>Priority: ${escapeHtml(survivor.priority)}<br>Coordinates: ${format(survivor.latitude, 6)}, ${format(survivor.longitude, 6)}<br>Confidence: ${format(survivor.confidence)}<br>Confirmed: ${survivor.confirmed ? 'YES' : 'NO'}<br>Frames: ${survivor.frames}<br>Latest frame: ${survivor.last_frame ?? 'n/a'}`;

    function syncMarkers(survivors) {
      const liveIds = new Set(survivors.map(s => String(s.id)));
      for (const [id, marker] of markers) { if (!liveIds.has(id)) { map.removeLayer(marker); markers.delete(id); } }
      const bounds = [];
      for (const survivor of survivors) {
        const id = String(survivor.id), location = [survivor.latitude, survivor.longitude];
        let marker = markers.get(id);
        if (marker) { marker.setLatLng(location); marker.setIcon(pinIcon(survivor.priority, id)); marker.setPopupContent(popup(survivor)); marker.setTooltipContent(`ID: ${escapeHtml(id)}`); }
        else { marker = L.marker(location, {icon:pinIcon(survivor.priority, id), title:`Survivor ${id}`}).addTo(map).bindPopup(popup(survivor)).bindTooltip(`ID: ${escapeHtml(id)}`, {permanent:true, direction:'right', offset:[18,-17], className:'survivor-id-label'}); markers.set(id, marker); }
        bounds.push(location);
      }
      if (bounds.length && (document.getElementById('auto-follow').checked || !initialFitComplete)) { map.fitBounds(bounds, {padding:[36,36], maxZoom:18}); initialFitComplete = true; }
    }

    function renderSummary(survivors) {
      const counts = Object.fromEntries(priorities.map(priority => [priority, survivors.filter(s => s.priority === priority).length]));
      document.getElementById('summary').innerHTML = priorities.map(priority => `<div class="count ${priority}">${counts[priority]}<small>${priority}</small></div>`).join('');
    }

    function renderSurvivors(survivors) {
      const target = document.getElementById('survivor-list');
      if (!survivors.length) { target.innerHTML = '<div class="empty">No valid GPS survivor rows are currently available.</div>'; return; }
      target.innerHTML = survivors.map(s => `<article class="card ${escapeHtml(s.priority)}"><div class="topline"><h2>Survivor ${escapeHtml(s.id)} ${s.confirmed ? '<span class="confirmed">CONFIRMED</span>' : ''}</h2><span class="risk">${escapeHtml(s.protocol.level)}</span></div><dl><dt>Exact position</dt><dd>${format(s.latitude,6)}, ${format(s.longitude,6)}</dd><dt>Confidence</dt><dd>${format(s.confidence)}</dd><dt>Priority score</dt><dd>${format(s.priority_score,1)}</dd><dt>Frames / latest</dt><dd>${escapeHtml(s.frames)} / ${s.last_frame ?? 'n/a'}</dd></dl><div class="protocol"><strong>Protocol — ${escapeHtml(s.protocol.level)}</strong><br>${escapeHtml(s.protocol.action)}<br><span>${escapeHtml(s.protocol.notify)}</span><br><span>${escapeHtml(s.protocol.follow_up)}</span></div><div class="buttons"><button data-survivor-id="${escapeHtml(s.id)}" data-action="ACKNOWLEDGE">Acknowledge</button><button class="dispatch" data-survivor-id="${escapeHtml(s.id)}" data-action="DISPATCH">Dispatch / assign</button><button data-survivor-id="${escapeHtml(s.id)}" data-action="REQUEST_UPDATE">Request update</button></div></article>`).join('');
    }

    function renderDataset(survivors, source) {
      document.getElementById('data-source').textContent = `Active source: ${source || 'not available'}`;
      const target = document.getElementById('live-dataset');
      if (!survivors.length) { target.innerHTML = '<div class="empty">No valid live rows available yet.</div>'; return; }
      target.innerHTML = `<div class="dataset-wrap"><table class="dataset"><thead><tr><th>ID</th><th>Confidence</th><th>Latitude</th><th>Longitude</th><th>Priority</th><th>Score</th><th>Confirmed</th><th>Frames</th></tr></thead><tbody>${survivors.map(s => `<tr><td>${escapeHtml(s.id)}</td><td>${format(s.confidence)}</td><td>${format(s.latitude,6)}</td><td>${format(s.longitude,6)}</td><td>${escapeHtml(s.priority)}</td><td>${format(s.priority_score,1)}</td><td>${s.confirmed ? 'YES' : 'NO'}</td><td>${escapeHtml(s.frames)}</td></tr>`).join('')}</tbody></table></div>`;
    }

    document.getElementById('survivor-list').addEventListener('click', event => {
      const button = event.target.closest('button[data-action]');
      if (button) recordAction(button.dataset.survivorId, button.dataset.action);
    });

    async function recordAction(survivorId, action) {
      try {
        const response = await fetch('/api/rescue-actions', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({survivor_id:survivorId, action})});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Action was not recorded.');
        await refreshEvents();
      } catch (error) { window.alert(`Rescue action not recorded: ${error.message}`); }
    }

    async function refreshEvents() {
      const response = await fetch('/api/rescue-actions', {cache:'no-store'}); const result = await response.json(); const target = document.getElementById('events');
      target.innerHTML = result.actions?.length ? result.actions.map(event => `<div class="event"><strong>${escapeHtml(event.action)}</strong> — Survivor ${escapeHtml(event.survivor?.id)}<br><time>${escapeHtml(event.timestamp)}</time><br>${escapeHtml(event.message)}</div>`).join('') : '<div class="empty">No actions recorded in this dashboard session.</div>';
    }

    async function refresh() {
      const pulse = document.getElementById('pulse'), status = document.getElementById('connection-text');
      try {
        const response = await fetch('/api/survivors', {cache:'no-store'}); const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Data unavailable');
        syncMarkers(data.survivors); renderSummary(data.survivors); renderSurvivors(data.survivors); renderDataset(data.survivors, data.source);
        pulse.className = `pulse ${data.error ? 'error' : 'live'}`; status.textContent = data.error || `${data.survivors.length} live survivor${data.survivors.length === 1 ? '' : 's'} · ${new Date(data.updated_at).toLocaleTimeString()}`;
      } catch (error) { pulse.className = 'pulse error'; status.textContent = `Disconnected: ${error.message}`; }
    }
    refresh(); refreshEvents(); setInterval(refresh, 800); setInterval(refreshEvents, 4000);
  </script>
</body>
</html>"""


def run_server(host: str, port: int, csv_path: Path, action_log_path: Path) -> None:
    LiveRescueRequestHandler.application = LiveRescueApplication(csv_path, action_log_path)
    server = ThreadingHTTPServer((host, port), LiveRescueRequestHandler)
    host_for_browser = "127.0.0.1" if host == "0.0.0.0" else host

    print("\n============================================================")
    print("LIVE RESCUE WEB APPLICATION")
    print("============================================================")
    print(f"CSV source: {csv_path.resolve()}")
    print(f"Local dashboard: http://{host_for_browser}:{port}")
    print("Mode: simulation only — no rescue service is contacted.")
    print("Press Ctrl+C to stop the web server.\n")

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nLive rescue web application stopped.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live rescue operations dashboard.")
    parser.add_argument("--csv", default=LIVE_RESULTS_CSV, help="Live survivor CSV path.")
    parser.add_argument("--action-log", default=RESCUE_ACTION_LOG, help="Local simulation action log path.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port (default: 8765).")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_server(
        arguments.host,
        arguments.port,
        Path(arguments.csv),
        Path(arguments.action_log),
    )
