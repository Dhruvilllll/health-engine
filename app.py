# ==========================================
# Statmize Health Engine - HuggingFace App
# ==========================================

import gradio as gr
from inference import HealthInferenceEngine

engine = HealthInferenceEngine()

STRAIN_COLORS = {
    "Low": "🟢",
    "Moderate": "🟡",
    "High": "🟠",
    "Critical": "🔴"
}

RISK_COLORS = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🔴"
}

def predict_health(hr, spo2, hrv):
    try:
        hrv_val = hrv if hrv > 0 else None
        result = engine.predict(hr, spo2, hrv_val)

        strain_score = result["strain_score"]
        strain_level = result["predicted_strain_level"]
        confidence = result["model_confidence"]
        overexertion = result["overexertion_risk"]
        hydration = result["hydration_alert"]

        emoji = STRAIN_COLORS.get(strain_level, "⚪")
        risk_emoji = RISK_COLORS.get(overexertion, "⚪")

        strain_display = f"{emoji} {strain_level}  ({strain_score:.1f}/100)"
        confidence_display = f"{confidence * 100:.1f}%" if confidence else "N/A"
        overexertion_display = f"{risk_emoji} {overexertion}"
        hydration_display = "⚠️ Alert — Possible Dehydration" if hydration else "✅ Normal"

        summary = generate_summary(strain_level, overexertion, hydration, hr, spo2)

        return strain_display, confidence_display, overexertion_display, hydration_display, summary

    except Exception as e:
        err = f"Error: {str(e)}"
        return err, err, err, err, "Analysis failed. Please check your inputs."


def generate_summary(strain_level, overexertion, hydration, hr, spo2):
    lines = [f"**Heart Rate:** {hr} BPM &nbsp;|&nbsp; **SpO₂:** {spo2}%\n"]

    if strain_level == "Critical":
        lines.append("🔴 **Critical strain detected.** Immediate rest is strongly recommended. Monitor vitals closely.")
    elif strain_level == "High":
        lines.append("🟠 **High physiological strain.** Consider reducing activity intensity and allowing recovery.")
    elif strain_level == "Moderate":
        lines.append("🟡 **Moderate strain.** Performance zone — sustainable with adequate hydration and rest between efforts.")
    else:
        lines.append("🟢 **Low strain.** Body is in a rested or lightly active state.")

    if overexertion == "High":
        lines.append("⚠️ **Overexertion risk is elevated.** HR reserve is near maximum capacity.")
    elif overexertion == "Medium":
        lines.append("⚡ Moderate cardiovascular load detected. Manage effort to avoid overexertion.")

    if hydration:
        lines.append("💧 **Hydration alert triggered.** SpO₂ drop may indicate dehydration or circulatory stress.")

    if spo2 < 92:
        lines.append("🚨 **Critical SpO₂ level.** Seek medical attention if this persists.")
    elif spo2 < 94:
        lines.append("⚠️ SpO₂ is below normal range. Monitor closely.")

    return "\n\n".join(lines)


# ── CSS ──────────────────────────────────────────────
custom_css = """
body { font-family: 'Inter', sans-serif; }
#title { text-align: center; padding: 10px 0 0; }
#subtitle { text-align: center; color: #888; margin-bottom: 20px; }
.metric-box { 
    background: #1e1e2e; 
    border-radius: 12px; 
    padding: 16px; 
    border: 1px solid #333;
}
.summary-box {
    background: #111827;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 16px;
}
footer { display: none; }
"""

# ── UI ───────────────────────────────────────────────
with gr.Blocks(
    title="Statmize Health Intelligence",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate"
    ),
    css=custom_css
) as app:

    gr.Markdown("# 🏋️ Statmize Physiological Strain Analyzer", elem_id="title")
    gr.Markdown("*Real-time fatigue & overexertion detection powered by hybrid ML*", elem_id="subtitle")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Vitals")

            hr_input = gr.Slider(
                minimum=40, maximum=200, value=80, step=1,
                label="❤️ Heart Rate (BPM)"
            )
            spo2_input = gr.Slider(
                minimum=80, maximum=100, value=98, step=0.1,
                label="🫁 SpO₂ — Blood Oxygen (%)"
            )
            hrv_input = gr.Slider(
                minimum=0, maximum=150, value=0, step=1,
                label="📡 HRV — Heart Rate Variability (set 0 to skip)"
            )

            predict_btn = gr.Button("⚡ Analyze Strain", variant="primary", size="lg")

            gr.Markdown(
                "_Baseline calibrated from training data: "
                "Resting HR ~66 BPM | Max HR ~155 BPM | Resting SpO₂ ~97.5%_",
                elem_classes=["subtitle"]
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")

            strain_level_out = gr.Textbox(label="Strain Level", interactive=False)
            confidence_out = gr.Textbox(label="Model Confidence", interactive=False)

            with gr.Row():
                overexertion_out = gr.Textbox(label="Overexertion Risk", interactive=False)
                hydration_out = gr.Textbox(label="Hydration Status", interactive=False)

    gr.Markdown("---")
    gr.Markdown("### 🧠 Analysis Summary")
    summary_out = gr.Markdown("*Run an analysis to see your personalized health summary.*")

    gr.Markdown("---")
    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown("""
        **Statmize Health Engine** uses a hybrid approach:

        - **HR Reserve Load (45%)** — How hard your heart is working relative to your personal resting & max HR
        - **HRV Suppression (30%)** — Nervous system fatigue indicator
        - **SpO₂ Drop (15%)** — Oxygen efficiency vs your resting baseline
        - **Recovery Slope (10%)** — How quickly HR recovers between efforts

        A LightGBM classifier trained on synthetic physiological data classifies strain into:
        `Low → Moderate → High → Critical`

        The strain *score* (0–100) is computed deterministically from the formula above for full transparency.
        """)

    predict_btn.click(
        predict_health,
        inputs=[hr_input, spo2_input, hrv_input],
        outputs=[strain_level_out, confidence_out, overexertion_out, hydration_out, summary_out]
    )

if __name__ == "__main__":
    app.launch()