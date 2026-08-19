import streamlit as st

from src.llm_interface import analyze_session


st.set_page_config(
    page_title="Ecommerce Conversion Intelligence",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------
# Example session
# ---------------------------------------------------------

EXAMPLE_TEXT = (
    "This is a returning visitor shopping in November. "
    "They viewed 25 product pages and spent 720 seconds "
    "on product-related pages. "
    "They visited 2 administrative pages for 45 seconds "
    "and 1 informational page for 20 seconds. "
    "Their bounce rate is 0.02, exit rate is 0.04, "
    "page value is 35.5, and special day score is 0. "
    "Operating system is 2, browser is 2, region is 1, "
    "traffic type is 2, and they are shopping on the weekend."
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "session_input" not in st.session_state:
    st.session_state.session_input = ""


def load_example():
    """
    Load the example ecommerce session into the input box.
    """
    st.session_state.session_input = EXAMPLE_TEXT


def clear_input():
    """
    Clear the session input box.
    """
    st.session_state.session_input = ""


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Ecommerce Conversion Intelligence")

st.subheader(
    "AI-Powered Conversion Prediction & Sales Intelligence"
)

st.write(
    """
    Describe an ecommerce visitor session in plain English.
    The application uses an LLM to interpret the session,
    a trained machine learning model to estimate conversion
    probability, and AI to explain the result in business terms.
    """
)


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

st.markdown("### Describe the Visitor Session")

button_col1, button_col2 = st.columns(2)

with button_col1:
    st.button(
        "Show Example",
        on_click=load_example,
        use_container_width=True,
    )

with button_col2:
    st.button(
        "Clear",
        on_click=clear_input,
        use_container_width=True,
    )


user_input = st.text_area(
    "Enter session information:",
    key="session_input",
    height=220,
    placeholder=(
        "Example: This is a returning visitor shopping "
        "in November who viewed 25 product pages..."
    ),
)


analyze_button = st.button(
    "Analyze Conversion",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------

if analyze_button:

    if not user_input.strip():

        st.warning(
            "Please describe an ecommerce session first."
        )

    else:

        with st.spinner(
            "Analyzing visitor session..."
        ):

            try:

                result = analyze_session(
                    user_input
                )

                # -----------------------------------------
                # Missing information
                # -----------------------------------------

                if not result["success"]:

                    if (
                        result.get("error_type")
                        == "missing_features"
                    ):

                        st.warning(
                            result["message"]
                        )

                        st.markdown(
                            "### Missing Information"
                        )

                        missing = result.get(
                            "missing_features",
                            [],
                        )

                        for feature in missing:
                            st.write(
                                f"- {feature}"
                            )

                        with st.expander(
                            "View extracted information"
                        ):

                            st.json(
                                result.get(
                                    "features",
                                    {},
                                )
                            )

                    else:

                        st.error(
                            result.get(
                                "message",
                                "Unable to analyze session.",
                            )
                        )

                # -----------------------------------------
                # Successful prediction
                # -----------------------------------------

                else:

                    probability = result[
                        "conversion_percentage"
                    ]

                    prediction = result[
                        "prediction"
                    ]

                    st.markdown(
                        "## Conversion Analysis"
                    )

                    metric_col1, metric_col2 = (
                        st.columns(2)
                    )

                    with metric_col1:

                        st.metric(
                            "Conversion Probability",
                            f"{probability:.2f}%",
                        )

                    with metric_col2:

                        st.metric(
                            "Model Prediction",
                            (
                                "Likely to Convert"
                                if prediction
                                else "Unlikely to Convert"
                            ),
                        )

                    st.progress(
                        min(
                            max(
                                probability / 100,
                                0.0,
                            ),
                            1.0,
                        )
                    )

                    # -------------------------------------
                    # AI summary
                    # -------------------------------------

                    st.markdown(
                        "### AI Business Summary"
                    )

                    st.write(
                        result["summary"]
                    )

                    # -------------------------------------
                    # Recommendation
                    # -------------------------------------

                    st.markdown(
                        "### Recommended Action"
                    )

                    st.success(
                        result["recommendation"]
                    )

                    # -------------------------------------
                    # Disclaimer
                    # -------------------------------------

                    st.caption(
                        "This prediction is generated by a "
                        "machine learning model and should be "
                        "used as decision support, not as a "
                        "guarantee of customer behavior."
                    )

                    # -------------------------------------
                    # Extracted features
                    # -------------------------------------

                    with st.expander(
                        "View extracted model features"
                    ):

                        st.json(
                            result["features"]
                        )

            except Exception as error:

                st.error(
                    "An unexpected error occurred "
                    "while analyzing the session."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(error)
                    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header(
        "About the Model"
    )

    st.write(
        """
        The application predicts whether an ecommerce
        visitor session is likely to result in revenue.
        """
    )

    st.markdown(
        "**Selected Model:** Random Forest"
    )

    st.markdown(
        "**F1 Score:** 0.6520"
    )

    st.markdown(
        "**ROC-AUC:** 0.9253"
    )

    st.markdown(
        "**Dataset:** Online Shoppers Purchasing Intention"
    )

    st.divider()

    st.markdown("### Architecture")

    st.write(
        """
        Natural Language  
        ↓  
        Nebius LLM  
        ↓  
        Feature Validation  
        ↓  
        Random Forest  
        ↓  
        Conversion Probability  
        ↓  
        AI Business Explanation
        """
    )