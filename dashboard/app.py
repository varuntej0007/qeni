import os
import json
from collections import defaultdict, deque
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from loguru import logger

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "qeni-prod-key-change-me")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

# State
_results = defaultdict(lambda: deque(maxlen=200))
_latest = {}
_unit_anomaly_streaks = defaultdict(int)
_global_stats = {
    "total_inferences": 0,
    "total_anomalies": 0,
    "avg_quantum_latency": 0.0,
    "avg_classical_latency_ms": 0.0,
    "system_start": datetime.utcnow().isoformat(),
    "backend": "AerSimulator",
    "privacy_preserved": True,
}


def push_result(result_dict: dict):
    uid = result_dict["unit_id"]
    _results[uid].append(result_dict)
    _latest[uid] = result_dict

    # Update stats
    n = _global_stats["total_inferences"] + 1
    _global_stats["total_inferences"] = n
    if result_dict["prediction"] == 1:
        _global_stats["total_anomalies"] += 1
        _unit_anomaly_streaks[uid] += 1
    else:
        _unit_anomaly_streaks[uid] = 0

    prev_q = _global_stats["avg_quantum_latency"]
    _global_stats["avg_quantum_latency"] = (
        prev_q + (result_dict["quantum_latency_sec"] - prev_q) / n
    )
    prev_c = _global_stats["avg_classical_latency_ms"]
    _global_stats["avg_classical_latency_ms"] = (
        prev_c + (result_dict["classical_latency_sec"] * 1000 - prev_c) / n
    )

    result_dict["anomaly_streak"] = _unit_anomaly_streaks[uid]

    socketio.emit("inference_result", result_dict)
    socketio.emit("stats_update", _global_stats)


def set_backend_name(name: str):
    _global_stats["backend"] = name


@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/latest")
def api_latest():
    return jsonify(_latest)

@app.route("/api/history/<int:unit_id>")
def api_history(unit_id: int):
    return jsonify(list(_results.get(unit_id, [])))

@app.route("/api/stats")
def api_stats():
    return jsonify(_global_stats)

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "uptime_sec": (datetime.utcnow() -
            datetime.fromisoformat(_global_stats["system_start"])).total_seconds(),
        "units": len(_latest),
    })

@socketio.on("connect")
def on_connect():
    emit("initial_state", {"latest": _latest, "stats": _global_stats})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
