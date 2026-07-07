"""
ANE Experiment Wizard — Flask Application
==========================================
Professional web interface for running LLM benchmarks on the Apple Neural Engine.

Usage:
    sudo python3.11 app.py
    Open http://localhost:5050 in your browser.
"""

import os
import sys
import json

# Ensure the wizard package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_socketio import SocketIO

# Import wizard modules
from wizard.models import discover_local_models, get_known_models_list, download_model, get_available_compressions
from wizard.engine import ExperimentEngine
from wizard.results import list_experiments, get_experiment, compare_experiments, save_notes, export_report, get_output_dir
from wizard.validator import validate

# ── App Setup ─────────────────────────────────────────────────────────────

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'ane-wizard-secret-key'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global engine instance
engine = None


def get_all_models():
    """Combine locally discovered models with known models list."""
    local = discover_local_models()
    known = get_known_models_list()

    # Merge: use local info if available, otherwise use known
    seen_ids = {m['id'] for m in local}
    combined = list(local)
    for k in known:
        if k['id'] not in seen_ids:
            combined.append(k)

    return combined


# ── Page Routes ───────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    models = get_all_models()
    experiments = list_experiments()
    loaded = ExperimentEngine.get_loaded_model()
    return render_template('dashboard.html',
                           active_page='dashboard',
                           models=models,
                           experiments=experiments,
                           loaded_model=loaded)


@app.route('/wizard')
def wizard():
    models = get_all_models()
    loaded = ExperimentEngine.get_loaded_model()
    return render_template('wizard.html',
                           active_page='wizard',
                           models=models,
                           loaded_model=loaded)


@app.route('/experiment')
def experiment_page():
    loaded = ExperimentEngine.get_loaded_model()
    return render_template('experiment.html',
                           active_page='experiment',
                           loaded_model=loaded)


@app.route('/experiments')
def experiments_list():
    experiments = list_experiments()
    loaded = ExperimentEngine.get_loaded_model()
    return render_template('dashboard.html',
                           active_page='experiments',
                           models=get_all_models(),
                           experiments=experiments,
                           loaded_model=loaded)


@app.route('/results/<experiment_id>')
def results_page(experiment_id):
    exp = get_experiment(experiment_id)
    if not exp:
        return redirect('/')

    metadata = exp.get('metadata', {})
    summary = exp.get('summary', {})
    notes = exp.get('notes', '')

    # Prepare chart data
    chart_data = None
    if 'power_samples' in exp:
        samples = exp['power_samples']
        chart_data = {
            'labels': list(range(1, len(samples) + 1)),
            'cpu_mw': [float(s.get('cpu_mw', 0)) for s in samples],
            'ane_mw': [float(s.get('ane_mw', 0)) for s in samples],
            'gpu_mw': [float(s.get('gpu_mw', 0)) for s in samples],
        }

    loaded = ExperimentEngine.get_loaded_model()
    return render_template('results.html',
                           active_page='experiments',
                           metadata=metadata,
                           summary=summary,
                           notes=notes,
                           chart_data=chart_data,
                           loaded_model=loaded)


@app.route('/compare')
def compare_page():
    experiments = list_experiments()
    exp1_id = request.args.get('exp1', '')
    exp2_id = request.args.get('exp2', '')

    comparison = None
    if exp1_id and exp2_id:
        comparison = compare_experiments(exp1_id, exp2_id)

    loaded = ExperimentEngine.get_loaded_model()
    return render_template('compare.html',
                           active_page='compare',
                           experiments=experiments,
                           selected_1=exp1_id,
                           selected_2=exp2_id,
                           comparison=comparison,
                           loaded_model=loaded)


# ── API Routes ────────────────────────────────────────────────────────────

@app.route('/api/models')
def api_models():
    return jsonify(get_all_models())


@app.route('/api/models/<path:model_id>/memory')
def api_model_memory(model_id):
    compression = request.args.get('compression', 'int8')
    from wizard.models import PARAM_COUNTS
    params_b = PARAM_COUNTS.get(model_id, 1.0)

    if compression == 'int4':
        est = f"~{params_b * 0.5:.1f} GB"
    elif compression == 'int8':
        est = f"~{params_b * 1.0:.1f} GB"
    else:
        est = f"~{params_b * 2.0:.1f} GB"

    return jsonify({"estimate": est})


@app.route('/api/experiment/start', methods=['POST'])
def api_start_experiment():
    global engine

    if engine and engine.running:
        return jsonify({"error": "An experiment is already running"}), 400

    config = request.json

    # Map prompt dataset from the select element
    if config.get('prompt_source') == 'dataset':
        config['prompt_dataset'] = config.get('prompt_dataset', 'general')

    # Set defaults from metric profile
    profile = config.get('metric_profile', 'quick')
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            app_config = json.load(f)
        profile_config = app_config.get('metric_profiles', {}).get(profile, {})
        if 'interval_ms' not in config:
            config['interval_ms'] = profile_config.get('interval_ms', 1000)
        config['idle_baseline'] = app_config.get('idle_baseline', {})
        config['ane_threshold'] = app_config.get('ane_active_threshold_mw', 10.0)

    # Set output dir
    config['output_dir'] = str(get_output_dir())

    def emit_fn(event, data):
        socketio.emit(event, data)

    engine = ExperimentEngine(emit_fn=emit_fn)
    engine.run_experiment(config)

    return jsonify({"status": "started"})


@app.route('/api/experiment/cancel', methods=['POST'])
def api_cancel_experiment():
    global engine
    if engine and engine.running:
        engine.cancel()
        return jsonify({"status": "cancelling"})
    return jsonify({"status": "no experiment running"})


@app.route('/api/experiment/<experiment_id>/notes', methods=['POST'])
def api_save_notes(experiment_id):
    data = request.json
    notes = data.get('notes', '')
    success = save_notes(experiment_id, notes)
    return jsonify({"success": success})


@app.route('/api/experiment/<experiment_id>/export')
def api_export(experiment_id):
    fmt = request.args.get('format', 'json')
    content = export_report(experiment_id, format=fmt)
    if content is None:
        return jsonify({"error": "Experiment not found"}), 404

    if fmt == 'json':
        return Response(content, mimetype='application/json',
                        headers={"Content-Disposition": f"attachment; filename={experiment_id}.json"})
    elif fmt == 'csv':
        return Response(content, mimetype='text/csv',
                        headers={"Content-Disposition": f"attachment; filename={experiment_id}.csv"})

    return jsonify({"error": "Invalid format"}), 400


@app.route('/api/experiment/<experiment_id>/summary')
def api_experiment_summary(experiment_id):
    exp = get_experiment(experiment_id)
    if not exp:
        return jsonify({}), 404
    return jsonify(exp.get('summary', {}))


@app.route('/api/validate', methods=['POST'])
def api_validate():
    config = request.json
    results = validate(config)
    return jsonify(results)


# ── WebSocket Events ──────────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    print(f"Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected")


# ── Entry Point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Check flag
    if '--check' in sys.argv:
        print("✓ App initialized successfully")
        print(f"  Models found: {len(get_all_models())}")
        print(f"  Experiments: {len(list_experiments())}")
        sys.exit(0)
        
    # Enforce running as root for powermetrics
    if os.geteuid() != 0:
        print("\n" + "=" * 60)
        print("ERROR: ANE Wizard must be run as root to access Apple power sensors.")
        print("Please restart the application from your terminal using sudo:")
        print("    cd \"/Users/kartik/Documents/paper tools\"")
        print("    sudo python3.11 wizard/app.py")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ANE Experiment Wizard")
    print("  Open http://localhost:5050 in your browser")
    print("=" * 60 + "\n")

    socketio.run(app, host='127.0.0.1', port=5050, debug=False, allow_unsafe_werkzeug=True)
