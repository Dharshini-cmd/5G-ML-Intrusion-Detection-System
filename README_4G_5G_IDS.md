# AEGIS 4G / 5G IDS

This generated version keeps the original IDS prediction and IP reputation features, then adds:

- 4G LTE and 5G SA IDS mode selection.
- A `/network_context` API for 4G/5G traffic differentiation.
- A `/network_profiles` API for network topology and baseline metadata.
- A dashboard diagram showing radio, transport, core, IDS, AI classifier, and SOC response flow.
- A live telemetry chart for latency, traffic load, anomaly pressure, and core/slice risk.
- A live SOC alert console with severity/network filters.
- In-memory IDS alert history through `/alerts`.
- Alert export to JSON from the dashboard.
- One-click alert clearing through `/alerts/clear`.
- Attack categorization for traffic alerts across 4G and 5G.
- Supported attack types through `/attack_categories`.
- SOC alert table now shows attack type and attack family.
- Alert filtering by attack type.

## Run

```bash
cd /Users/ashishtitus/Documents/Codex/2026-06-12/files-mentioned-by-the-user-app/outputs/aegis_4g5g_ids
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Optional API Keys

Create or edit `.env` with:

```bash
ABUSEIPDB_API_KEY=your_key
GEMINI_API_KEY=your_key
```

The IDS model works without these keys. IP reputation is richer when they are available.
