import matplotlib
matplotlib.use('Agg')  # Must be set before any other matplotlib import for HF Spaces

import pandas as pd
import shap
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# ── Load model ──────────────────────────────────────────────────────────────
loaded_model = xgb.Booster()
loaded_model.load_model("Final_Features_model.json")

# Setup SHAP using the Booster directly
explainer = shap.TreeExplainer(loaded_model)  # PLEASE DO NOT CHANGE THIS.

SCALE_TOOLTIP = "1 = Strongly Disagree  ·  5 = Strongly Agree"

#  ── Main prediction function ─────────────────────────────────────────────────
def main_func(LearningDevelopment, AppreciatedAtWork, Voice, WorkEnvironment, SupportiveGM, Merit, Engagement):
    feature_names = ['LearningDevelopment', 'AppreciatedAtWork', 'Voice', 'WorkEnvironment', 
                     'Engagement', 'SupportiveGM', 'Merit']

    new_row_df = pd.DataFrame(
        [[LearningDevelopment, AppreciatedAtWork, Voice, WorkEnvironment, Engagement, SupportiveGM, Merit]],
        columns=feature_names
    )
    dmat = xgb.DMatrix(new_row_df)

    # Prediction
    leave_prob  = float(loaded_model.predict(dmat)[0])
    stay_prob = 1.0 - leave_prob

    # SHAP
    shap_values = explainer.shap_values(new_row_df)
    exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=new_row_df.values[0],
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap.plots.bar(exp, max_display=7, show=False, ax=ax)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')
    plt.tight_layout()
    plt.close('all')

    # ── Risk levels ──
    stay_pct  = round(stay_prob  * 100, 1)
    leave_pct = round(leave_prob * 100, 1)

    # SVGS
    GREEN_CIRCLE = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="8" fill="#22c55e"/></svg>"""

    YELLOW_CIRCLE = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="8" fill="#eab308"/></svg>"""

    RED_CIRCLE = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="8" fill="#ef4444"/></svg>"""

    if stay_prob >= 0.70:
        badge_color  = "#0D5C6B"
        badge_bg     = "rgba(13,92,107,0.08)"
        badge_border = "rgba(13,92,107,0.28)"
        bar_color    = "linear-gradient(90deg, #1A7A8A, #2A9AAE)"
        badge_icon   = GREEN_CIRCLE
        badge_label  = "Low Turnover Risk"
        badge_sub    = f"Model predicts <strong>{stay_pct}%</strong> likelihood of retention."
    elif stay_prob >= 0.45:
        badge_color  = "#9A6F1E"
        badge_bg     = "rgba(201,151,58,0.09)"
        badge_border = "rgba(201,151,58,0.32)"
        bar_color    = "linear-gradient(90deg, #C9973A, #E8B455)"
        badge_icon   = YELLOW_CIRCLE
        badge_label  = "Moderate Turnover Risk"
        badge_sub    = f"Model predicts <strong>{stay_pct}%</strong> likelihood of retention."
    else:
        badge_color  = "#962B20"
        badge_bg     = "rgba(192,57,43,0.08)"
        badge_border = "rgba(192,57,43,0.28)"
        bar_color    = "linear-gradient(90deg, #C0392B, #E05A4A)"
        badge_icon   = RED_CIRCLE
        badge_label  = "High Turnover Risk"
        badge_sub    = f"Model predicts only <strong>{stay_pct}%</strong> likelihood of retention."

    results_html = f"""
    <div class="result-reveal" style="font-family:'DM Sans',sans-serif;">

      <!-- Risk badge -->
      <div style="
          background:{badge_bg};
          border:1.5px solid {badge_border};
          border-radius:10px;
          padding:16px 20px;
          display:flex;
          align-items:center;
          gap:14px;
          margin-bottom:12px;
      ">
          <span style="font-size:1.9rem;line-height:1;flex-shrink:0;">{badge_icon}</span>
          <div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:{badge_color};
                          letter-spacing:0.01em;margin-bottom:3px;">
                  {badge_label}
              </div>
              <div style="font-size:0.83rem;color:#4A6278;line-height:1.5;">
                  {badge_sub}
              </div>
          </div>
      </div>

      <!-- Probability bar breakdown -->
      <div style="
          background:#FFFFFF;
          border:1.5px solid rgba(13,92,107,0.14);
          border-radius:10px;
          padding:18px 20px;
      ">
          <div style="font-family:'DM Serif Display',serif;font-size:0.82rem;color:#4A6278;
                      text-transform:uppercase;letter-spacing:0.08em;margin-bottom:14px;">
              Probability Breakdown
          </div>

          <!-- Stay bar -->
          <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                  <span style="font-size:0.85rem;font-weight:600;color:{badge_color};">Stay</span>
                  <span style="font-size:0.92rem;font-weight:700;color:#0D5C6B;">{stay_pct}%</span>
              </div>
              <div style="background:rgba(13,92,107,0.09);border-radius:999px;height:10px;overflow:hidden;">
                  <div style="width:{stay_pct}%;height:100%;background:{bar_color};border-radius:999px;"></div>
              </div>
          </div>

          <!-- Leave bar -->
          <div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                  <span style="font-size:0.85rem;font-weight:600;color:#4A6278;">Leave</span>
                  <span style="font-size:0.92rem;font-weight:700;color:#0D1B2A;">{leave_pct}%</span>
              </div>
              <div style="background:rgba(74,98,120,0.10);border-radius:999px;height:10px;overflow:hidden;">
                  <div style="width:{leave_pct}%;height:100%;
                              background:linear-gradient(90deg,#6B8299,rgba(74,98,120,0.5));
                              border-radius:999px;"></div>
              </div>
          </div>
      </div>

    </div>
    """

    # NOTE: Returns only 2 values — fig and results_html. gr.Label removed.
    return fig, results_html


# ── Theme ────────────────────────────────────────────────────────────────────
teal_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
).set(
    slider_color="#1A7A8A",
    button_primary_background_fill="linear-gradient(135deg, #0D1B2A 0%, #0D5C6B 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0D1B2A 0%, #1A7A8A 100%)",
    button_primary_text_color="#FFFFFF",
    block_border_width="0px",
    block_shadow="none",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:      #0D1B2A;
    --ink:       #1A2E42;
    --teal:      #1A7A8A;
    --teal-dark: #0D5C6B;
    --gold:      #C9973A;
    --gold-dk:   #9A6F1E;
    --cream:     #F4EFE6;
    --white:     #FFFFFF;
    --text:      #1A2E42;
    --subtext:   #4A6278;
    --radius:    11px;
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Fade-in animation ── */
@keyframes resultReveal {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
.result-reveal {
    animation: resultReveal 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--ink) 55%, #193D5A 100%);
    border-radius: var(--radius);
    padding: 32px 38px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(201,151,58,0.16) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.95rem !important;
    color: #FFFFFF !important;
    margin: 0 0 6px !important;
    letter-spacing: -0.02em;
}
.app-header p {
    color: rgba(255,255,255,0.62) !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
    font-weight: 300;
    max-width: 580px;
}
.gold-accent { color: #E8B455 !important; }

/* ── Stepper ── */
.stepper-row {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 40px;
    padding: 6px 0;
    width: 100%;
    margin-bottom: 6px;
}
.step-item { display: flex; align-items: center; flex: 1; }
.step-circle {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #1A2E62 !important;
    color: #ffffff !important;
    font-family: 'DM Serif Display', serif;
    font-size: 0.9rem;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    border: 2px solid #1A7A8A !important;
    box-shadow: 0 0 0 3px rgba(26,122,138,0.15);
}
.step-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #0D5C6B !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-left: 7px;
    line-height: 1.2;
    max-width: 85px;
}
.step-connector {
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #0D5C6B 0%, rgba(13,92,107,0.18) 100%) !important;
    margin: 0 7px;
}

/* ══════════════════════════════════
   GROUP BOX
   ══════════════════════════════════ */
.group-box {
    border: 2px solid var(--teal-dark) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    margin-bottom: 12px !important;
    padding: 0 !important;
    gap: 0 !important;
    box-shadow: 0 2px 14px rgba(13,27,42,0.10) !important;
    background: var(--white) !important;
}
.group-box-gold {
    border-color: var(--gold) !important;
    box-shadow: 0 2px 14px rgba(201,151,58,0.16) !important;
}

/* Header bar — rendered as gr.HTML inside gr.Group */
.group-box .group-header {
    background: var(--teal-dark);
    padding: 10px 18px;
    /* flex lives on the inner span, not needed here */
}
.group-box-gold .group-header {
    background: var(--gold);
}

/* 
  The number badge + title sit inside a single <span> with display:flex
  so Gradio's block wrapper doesn't matter.
*/
.group-header-inner {
    display: flex;
    align-items: center;
    gap: 9px;
}
.group-header-num {
    width: 22px; height: 22px;
    background: rgba(255,255,255,0.22);
    border-radius: 50%;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'DM Serif Display', serif;
    flex-shrink: 0;
}
.group-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 0.93rem;
    color: #FFFFFF;
    letter-spacing: 0.01em;
    font-weight: 400;
    line-height: 1;
}

/* Sliders: clean up Gradio default padding inside group */
.group-box .gr-form,
.group-box .gradio-group,
.group-box > .form {
    background: var(--white) !important;
    border: none !important;
    padding: 14px 18px 8px !important;
    gap: 0 !important;
}

/* Slider labels */
.group-box label span {
    font-size: 0.86rem !important;
    color: var(--subtext) !important;
    font-weight: 400 !important;
}
.group-box .info {
    font-size: 0.75rem !important;
    color: var(--teal) !important;
    font-style: italic;
    opacity: 0.80;
}

/* Teal slider thumb + track */
.group-box input[type="range"]::-webkit-slider-thumb {
    background: var(--teal-dark) !important;
    transition: transform 0.15s, box-shadow 0.15s;
}
.group-box input[type="range"]:hover::-webkit-slider-thumb {
    background: var(--navy) !important;
    transform: scale(1.18);
    box-shadow: 0 0 0 5px rgba(13,92,107,0.18);
}
.group-box input[type="range"]::-webkit-slider-runnable-track {
    background: rgba(26,122,138,0.15) !important;
}
.group-box input[type="range"]::-moz-range-thumb {
    background: var(--teal-dark) !important;
}

.mean-marker-wrap { position: relative; width: 100%; }
.mean-marker {
    position: absolute;
    top: -18px;
    transform: translateX(-50%);
    font-size: 0.65rem;
    color: #0D5C6B !important;
    font-weight: 600;
    white-space: nowrap;
}
.mean-marker::after {
    content: '▼';
    display: block;
    text-align: center;
    font-size: 0.55rem;
    color: #1A7A8A !important;
}

/* ── Analyze button ── */
#analyze-btn {
    background: linear-gradient(135deg, var(--navy) 0%, var(--teal-dark) 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 13px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
    box-shadow: 0 4px 16px rgba(13,27,42,0.22) !important;
    width: 100% !important;
    margin-top: -30px !important;
}
#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(13,27,42,0.30) !important;
}

/* ── Profile buttons ── */
#profile1-btn, #profile2-btn, #profile3-btn {
    background: linear-gradient(135deg, #1A2E62 0%, #0D5C6B 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px 9px 0 0 !important;
    padding: 13px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
    box-shadow: 0 4px 16px rgba(13,27,42,0.22) !important;
    margin-top: 4px !important;
    margin-bottom: 0 !important;
}
#profile1-btn:hover, #profile2-btn:hover, #profile3-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(13,27,42,0.30) !important;
}

/* ── Solution buttons ── */
#profile4-btn, #profile5-btn, #profile6-btn {
    background: linear-gradient(135deg, #0D5C6B 0%, #1A7A8A 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0 0 9px 9px !important;
    padding: 8px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
    box-shadow: 0 4px 16px rgba(13,27,42,0.22) !important;
    margin-top: 0 !important;
}
#profile4-btn:hover, #profile5-btn:hover, #profile6-btn:hover {
    transform: translateY(2px) !important;
    box-shadow: 0 2px 8px rgba(13,27,42,0.20) !important;
}

/* ── Row spacing ── */
#profile-btn-row {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    gap: 24px !important;
}
#solution-btn-row {
    margin-top: -8px !important;
    padding-top: 0 !important;
    gap: 24px !important;
}

#profile-btn-row > div,
#solution-btn-row > div {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

#reset-btn {
    background: linear-gradient(135deg, #4a4a4a 0%, #6b6b6b 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 13px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
    box-shadow: 0 4px 16px rgba(13,27,42,0.22) !important;
    width: 100% !important;
    margin-top: -65px !important;
}
#reset-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(13,27,42,0.30) !important;
}

/* ── Output panel ── */
.output-panel {
    background: var(--white);
    border: 2px solid rgba(13,92,107,0.20);
    border-radius: var(--radius);
    padding: 22px 22px 16px;
    box-shadow: 0 3px 20px rgba(13,27,42,0.09);
    position: sticky;
    top: 16px;
}
.output-panel-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--teal-dark);
    margin-bottom: 16px;
    padding-bottom: 11px;
    border-bottom: 1.5px solid rgba(13,92,107,0.14);
}

/* ── Examples ── */
.gr-samples-table th,
.gr-samples-table td,
.gr-samples-table,
.examples-holder label,
.examples-holder .label-wrap span {
    color: var(--teal-dark) !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(26,122,138,0.18) 20%, rgba(26,122,138,0.18) 80%, transparent);
    margin: 20px 0;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    font-size: 0.75rem;
    color: var(--subtext);
    padding: 12px 0 4px;
    opacity: 0.6;
}
"""

# ── Layout ───────────────────────────────────────────────────────────────────
with gr.Blocks(theme=teal_theme, css=custom_css, title="Employee Turnover Predictor") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>Employee Turnover <span class="gold-accent">Predictor</span> &amp; Interpreter</h1>
        <p>Predict employee retention risk using key workplace satisfaction indicators.
           Complete each section and click <strong style="color:rgba(255,255,255,0.9)">Analyze</strong>
           to generate a prediction and SHAP explanation.</p>
           <p>(1 = Strongly Disagree  |  5 = Strongly Agree)</p>
    </div>
    """)

    gr.HTML("""
    <div class="stepper-row">
        <div class="step-item">
            <div class="step-circle"style="border-color:#0D5C6B;box-shadow:0 0 0 3px rgba(13,92,107,0.16);">1</div>
            <div class="step-label"style="color:#0D5C6B;">Focal Variables</div>
            <div class="step-connector"></div>
        </div>
        <div class="step-item">
            <div class="step-circle" style="border-color:#0D5C6B;box-shadow:0 0 0 3px rgba(13,92,107,0.16);">2</div>
            <div class="step-label" style="color:#0D5C6B;">Important Predictors</div>
            <div class="step-connector"></div>
        </div>
    </div>
    """)

    gr.HTML('<div class="divider"></div>')
    gr.HTML(f'<h3 style="color:#0D5C6B;font-family:\'DM Serif Display\',serif;font-weight:400;font-size:1.2rem;margin:0 0 0px;">Employee Profiles</h3>')
    gr.HTML('''<p style="color:#555;font-size:0.85rem;font-weight:600;margin:0 0 0px;">
        <span style="color:#0D5C6B;font-weight:700;">Step 1:</span> Select an employee profile to pre-populate the sliders, then scroll down and click <span style="text-decoration:underline;">Analyze</span> to see predictions.<br>
        <span style="color:#0D5C6B;font-weight:700;">Step 2:</span> Then select the corresponding solution to see the change in predictions.
        </p>''')

    with gr.Row(elem_id="profile-btn-row"):
        profile1_btn = gr.Button("👤 Picture Perfect Employee", elem_id="profile1-btn")
        profile2_btn = gr.Button("👤 At-Risk Employee", elem_id="profile2-btn")
        profile3_btn = gr.Button("👤 Moderate-Risk Employee", elem_id="profile3-btn")
    with gr.Row(elem_id="solution-btn-row"):
        profile4_btn = gr.Button("Solution 1", elem_id="profile4-btn")
        profile5_btn = gr.Button("Solution 2", elem_id="profile5-btn")
        profile6_btn = gr.Button("Solution 3", elem_id="profile6-btn")  

    with gr.Row(equal_height=False):

        # ── LEFT ──
        with gr.Column(scale=1):

            # Group 1
            with gr.Group(elem_classes=["group-box"]):
                gr.HTML("""
                <div class="group-header">
                  <span class="group-header-inner">
                    <span class="group-header-num">1</span>
                    <span class="group-header-title">Focal Variables</span>
                  </span>
                </div>""")
                LearningDevelopment = gr.Slider(label="Learning and Development Opportunities",
                                info="Category Average: 4.20", minimum=1, maximum=5, value=4.20, step=.1)
                AppreciatedAtWork = gr.Slider(label="Appreciation at Work",
                                info="Category Average: 4.22", minimum=1, maximum=5, value=4.22, step=.1)
                Voice = gr.Slider(label="Ability to Voice Opinions and Ideas",
                                  info="Category Average: 4.10", minimum=1, maximum=5, value=4.10, step=.1)

            # Group 2
            with gr.Group(elem_classes=["group-box"]):
                gr.HTML("""
                <div class="group-header">
                  <span class="group-header-inner">
                    <span class="group-header-num">2</span>
                    <span class="group-header-title">Important Predictors</span>
                  </span>
                </div>""")
                Engagement = gr.Slider(label="Employee Engagement Score",
                                       info="Category Average: 4.51",
                                       minimum=1, maximum=5, value=4.51, step=.1)
                WorkEnvironment = gr.Slider(label="Safe and Comfortable Work Environment",
                                      info="Category Average: 4.35", minimum=1, maximum=5, value=4.35, step=.1)
                SupportiveGM = gr.Slider(label="Having a Supportive General Manager",
                                              info="Category Average: 4.39", minimum=1, maximum=5, value=4.39, step=.1)
                Merit = gr.Slider(label="Higher Performance Leads to Higher Rewards",
                                     info="Category Average: 3.99", minimum=1, maximum=5, value=3.99, step=.1)
            gr.HTML('<div style="margin-top: 0px;"></div>')
            submit_btn = gr.Button("▶ Analyze Employee Profile ◀", elem_id="analyze-btn")

        # ── RIGHT ──
        with gr.Column(scale=1, min_width=460):
            gr.HTML(f"""<div class="output-panel"><div class="output-panel-header" style="color:#0D5C6B;">Prediction Results</div>""")

            results_box = gr.HTML(
                value="<div style='color:#4A6278;font-size:0.83rem;font-family:DM Sans,sans-serif;"
                      "padding:8px 0 16px;font-style:italic;opacity:0.72;'>"
                      "Complete the form and click Analyze to see results.</div>"
            )
            local_plot = gr.Plot(label="SHAP Feature Importance")
            gr.HTML('</div>')
            
            gr.HTML('<div style="margin-top: 0px;"></div>')
            reset_btn = gr.Button("↺ Reset to Category Averages", elem_id="reset-btn")

            # 2 outputs only — gr.Label removed
            submit_btn.click(
                main_func,
                [LearningDevelopment, AppreciatedAtWork, Voice, WorkEnvironment, SupportiveGM, Merit, Engagement],
                [local_plot, results_box],
                api_name="Employee_Turnover"
            )
            
    #reset button
    reset_btn.click(
        fn=lambda: [4.20, 4.22, 4.10, 4.51, 4.35, 4.39, 3.99],
        inputs=[],
        outputs=[LearningDevelopment, AppreciatedAtWork, Voice, Engagement, WorkEnvironment, SupportiveGM, Merit]
    )
    #profile buttons
    profile1_btn.click(
        fn=lambda: [4.7, 4, 4, 5, 4, 5, 5],
        inputs=[],
        outputs=[LearningDevelopment, AppreciatedAtWork, Voice, Engagement, WorkEnvironment, SupportiveGM, Merit]
    )
    profile2_btn.click(
        fn=lambda: [2.64, 3, 3, 4, 3.14, 3.07, 3],
        inputs=[],
        outputs=[LearningDevelopment, AppreciatedAtWork, Voice, Engagement, WorkEnvironment, SupportiveGM, Merit]
    )
    profile3_btn.click(
        fn=lambda: [5, 4, 4, 4.1, 5, 5, 4],
        inputs=[],
        outputs=[LearningDevelopment, AppreciatedAtWork, Voice, Engagement, WorkEnvironment, SupportiveGM, Merit]
    )
    #Solution buttons
    profile4_btn.click(
        fn=lambda: [4.7, 4.5, 4.22, 5, 4.7, 5, 5],
        inputs=[],
        outputs=[LearningDevelopment, AppreciatedAtWork, Voice, Engagement, WorkEnvironment, SupportiveGM, Merit]
    )
    profile5_btn.click(
        fn=lambda: [5, 4.2, 4.2, 4.56, 4.9, 5, 4.5],
        inputs=[],
        outputs=[LearningDevelopment, AppreciatedAtWork, Voice, Engagement, WorkEnvironment, SupportiveGM, Merit]
    )
    profile6_btn.click(
        fn=lambda: [4.7, 4.22, 4.40, 4.9, 4.8, 5, 5],
        inputs=[],
        outputs=[LearningDevelopment, AppreciatedAtWork, Voice, Engagement, WorkEnvironment, SupportiveGM, Merit]
    )

    gr.HTML('<div class="footer-text" style="color:#0D5C6B;opacity:1;font-weight:500;">University of Virginia MSBA · Team 8 · Employee Retention Analysis · 2026</div>')

demo.launch()
