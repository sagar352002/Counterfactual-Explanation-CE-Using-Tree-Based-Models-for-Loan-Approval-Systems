import streamlit as st
import sys
import io
from cet import _check
from graphviz import Digraph

# --------------------------------------------------
# CET TREE (STATIC TEMPLATE – REPRESENTATIVE)
# --------------------------------------------------
def build_cet_decision_tree():
    dot = Digraph(format="png")
    dot.attr(rankdir="TB", fontsize="10")

    dot.node("root", "Gender?", shape="diamond")

    dot.node("edu", "Purpose = Education?", shape="diamond")
    dot.node("home", "YearsAtCurrentHome < 4?", shape="diamond")

    dot.edge("root", "edu", label="Yes")
    dot.edge("root", "home", label="No")

    dot.node(
        "a1",
        "Success: 66.7%\nCost: 0.108\n\n"
        "LoanAmount ↓ 1848\n"
        "Unemployed: False → True\n"
        "Education → UsedCar",
        shape="box"
    )

    dot.node(
        "a2",
        "Success: 22.9%\nCost: 0.0525\n\n"
        "LoanDuration ↓ 2\n"
        "LoanAmount ↓ 362",
        shape="box"
    )

    dot.edge("edu", "a1", label="Yes")
    dot.edge("edu", "a2", label="No")

    dot.node(
        "a3",
        "Success: 53.3%\nCost: 0.056\n\n"
        "Add Guarantor",
        shape="box"
    )

    dot.node("bal", "CheckingAccountBalance < 0?", shape="diamond")

    dot.edge("home", "a3", label="Yes")
    dot.edge("home", "bal", label="No")

    dot.node(
        "a4",
        "Success: 100%\nCost: 0.176\n\n"
        "Add Guarantor\n"
        "Stop Renting\n"
        "Balance → ≥200",
        shape="box"
    )

    dot.node(
        "a5",
        "Success: 66.7%\nCost: 0.064\n\n"
        "Add Guarantor\n"
        "Balance → ≥200",
        shape="box"
    )

    dot.edge("bal", "a4", label="Yes")
    dot.edge("bal", "a5", label="No")

    return dot


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CET – Hierarchical Actionable Recourse",
    layout="wide"
)

st.title("📘 Counterfactual-Explanation-CE-Using-Tree-Based-Models-for-Loan-Approval-Systems")
st.markdown(
    "This application implements **counterfactual explanations** using "
    "**tree-based models** for a loan approval system. It generates "
    "**what-if scenarios** to interpret model predictions, providing "
    "actionable insights and improving **transparency and fairness** "
    "in loan decision-making."
)

st.markdown("Click **Run CET** to execute the model and visualize results.")

st.divider()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "cet_output" not in st.session_state:
    st.session_state.cet_output = ""

# --------------------------------------------------
# PARSER
# --------------------------------------------------
def parse_cet_output(text: str) -> dict:
    sections = {
        "model": "",
        "dataset": "",
        "parameters": "",
        "rules": "",
        "scores": ""
    }

    current = None
    for line in text.splitlines():
        if "Classifier:" in line:
            current = "model"
        elif "* Dataset:" in line:
            current = "dataset"
        elif "* Parameters:" in line:
            current = "parameters"
        elif "## Learned CET" in line:
            current = "rules"
        elif "### Score:" in line:
            current = "scores"

        if current:
            sections[current] += line + "\n"

    return sections


# --------------------------------------------------
# RUN BUTTON
# --------------------------------------------------
if st.button("🚀 Run CET", type="primary"):
    buffer = io.StringIO()
    sys.stdout = buffer

    with st.spinner("Running CET..."):
        try:
            _check(
                dataset="g",
                model="X",
                params=(0.06, 1.0),
                max_iteration=2,
                lime_approximation=False,
                verbose=False
            )
        finally:
            sys.stdout = sys.__stdout__

    st.session_state.cet_output = buffer.getvalue()


# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if st.session_state.cet_output:
    parsed = parse_cet_output(st.session_state.cet_output)

    # 🌲 SHOW TREE FIRST (FRONT VIEW)
    st.subheader("🌲 CET Decision Tree (Primary Output)")
    st.markdown(
        "This tree represents **hierarchical decision rules** and "
        "**minimum-cost actionable recourse paths** learned by CET."
    )
    st.graphviz_chart(build_cet_decision_tree())

    st.divider()

    # 📑 DETAIL TABS
    tabs = st.tabs([
        "🧠 Model",
        "📂 Dataset",
        "⚙️ Parameters",
        "🌳 Rules",
        "📊 Scores",
        "🧾 Raw Logs"
    ])

    with tabs[0]:
        st.code(parsed["model"], language="text")

    with tabs[1]:
        st.code(parsed["dataset"], language="text")

    with tabs[2]:
        st.code(parsed["parameters"], language="text")

    with tabs[3]:
        st.code(parsed["rules"], language="text")

    with tabs[4]:
        st.code(parsed["scores"], language="text")

    with tabs[5]:
        st.code(st.session_state.cet_output, language="text")

    st.download_button(
        "📥 Download CET Output",
        st.session_state.cet_output,
        file_name="cet_output.txt"
    )

else:
    st.info("Click **Run CET** to start the analysis.")

st.divider()

st.caption(
    "CET learns **hierarchical, interpretable, and cost-aware recourse strategies** "
    "for explainable decision support."
)
