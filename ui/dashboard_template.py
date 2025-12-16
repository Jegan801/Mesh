def render_dashboard(
    severity_count,
    action_count,
    health_score,
    output_path
):
    status = "✅ Healthy" if health_score > 80 else "⚠ Needs Improvement"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<title>AI Mesh Quality Dashboard</title>
<style>
body {{
    background: #0f172a;
    color: #e5e7eb;
    font-family: Arial, sans-serif;
}}
.card {{
    background: #020617;
    padding: 20px;
    margin: 15px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,0,0,0.6);
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}}
h1 {{
    text-align: center;
}}
.big {{
    font-size: 48px;
}}
.action {{
    font-size: 18px;
}}
</style>
</head>
<body>

<h1>🤖 AI Mesh Quality Copilot</h1>

<div class="grid">
    <div class="card">
        <h2>Mesh Health Score</h2>
        <div class="big">{health_score} / 100</div>
        <p>{status}</p>
    </div>

    <div class="card">
        <h2>Issue Distribution</h2>
        🔴 High: {severity_count["HIGH"]}<br>
        🟠 Medium: {severity_count["MEDIUM"]}<br>
        🟡 Low: {severity_count["LOW"]}
    </div>

    <div class="card">
        <h2>Recommended Actions</h2>
        <div class="action">🧹 Delete Nodes: {action_count["DELETE"]}</div>
        <div class="action">➕ Add Nodes: {action_count["ADD"]}</div>
        <div class="action">🎯 Move Nodes: {action_count["MOVE"]}</div>
        <div class="action">🔁 Rebuild Mesh: {action_count["REMESH"]}</div>
    </div>
</div>

<div class="card">
<h2>AI Insight</h2>
<p>
Most mesh quality issues are caused by CAD deviation and poor element transitions.
AI recommends improving mesh density near complex geometry regions.
</p>
</div>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
