import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.app import (
    FEATURE_SCHEMA,
    predict_conversion,
)


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Nebius client
# ---------------------------------------------------------

def get_llm_client():
    """
    Create the Nebius/OpenAI-compatible client.
    """

    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL")

    if not api_key:
        raise ValueError("NEBIUS_API_KEY is missing.")

    if not base_url:
        raise ValueError("NEBIUS_BASE_URL is missing.")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def get_model_name():
    """
    Return the configured Nebius model name.
    """

    model_name = os.getenv("NEBIUS_MODEL")

    if not model_name:
        raise ValueError("NEBIUS_MODEL is missing.")

    return model_name


# ---------------------------------------------------------
# Feature extraction prompt
# ---------------------------------------------------------

def build_feature_prompt(user_text):
    """
    Build the prompt used to extract ecommerce model features
    from natural-language input.
    """

    feature_names = list(FEATURE_SCHEMA.keys())

    prompt = f"""
You are a structured data extraction assistant for an ecommerce
conversion prediction model.

Extract ONLY information explicitly provided by the user.

The model expects these exact fields:

{feature_names}

Field meanings:

Administrative:
Number of administrative pages visited.

Administrative_Duration:
Total time spent on administrative pages, in seconds.

Informational:
Number of informational pages visited.

Informational_Duration:
Total time spent on informational pages, in seconds.

ProductRelated:
Number of product-related pages visited.

ProductRelated_Duration:
Total time spent on product-related pages, in seconds.

BounceRates:
Bounce rate for the session.

ExitRates:
Exit rate for the session.

PageValues:
Page value associated with the session.

SpecialDay:
Closeness of the visit to a special day, from 0 to 1.

Month:
Month abbreviation such as Feb, Mar, May, June, Jul,
Aug, Sep, Oct, Nov, or Dec.

OperatingSystems:
Integer operating system category.

Browser:
Integer browser category.

Region:
Integer region category.

TrafficType:
Integer traffic-source category.

VisitorType:
One of:
Returning_Visitor
New_Visitor
Other

Weekend:
Boolean true or false.

Rules:

1. Never invent values.
2. If a value is not explicitly provided, return null.
3. Use the exact field names above.
4. Return valid JSON only.
5. Do not include markdown.
6. Do not include explanations.
7. Do not add fields that are not in the schema.

User input:

{user_text}
"""

    return prompt


# ---------------------------------------------------------
# Natural language -> structured features
# ---------------------------------------------------------

def parse_user_input(user_text):
    """
    Convert natural-language ecommerce information into
    structured model features using the Nebius LLM.
    """

    client = get_llm_client()
    model_name = get_model_name()

    prompt = build_feature_prompt(user_text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the requested ecommerce features. "
                    "Return only valid JSON. "
                    "Never explain your reasoning."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=2500,
        reasoning_effort="low",
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {content}"
        ) from exc

    cleaned = {}

    for feature_name in FEATURE_SCHEMA:
        cleaned[feature_name] = parsed.get(
            feature_name
        )

    return cleaned


# ---------------------------------------------------------
# Deterministic business explanation
# ---------------------------------------------------------

def generate_business_explanation(
    features,
    prediction_result,
):
    """
    Generate a reliable business-facing explanation
    without requiring a second LLM call.
    """

    probability = prediction_result[
        "conversion_percentage"
    ]

    prediction = prediction_result[
        "prediction"
    ]

    visitor_type = features.get(
        "VisitorType",
        "the visitor"
    )

    month = features.get(
        "Month"
    )

    product_pages = features.get(
        "ProductRelated"
    )

    product_duration = features.get(
        "ProductRelated_Duration"
    )

    if prediction:

        summary = (
            f"The model estimates a {probability:.2f}% conversion "
            "probability for this ecommerce session, indicating that "
            "the visitor is likely to convert. "
            "This is a model estimate based on the supplied session "
            "information and is not a guarantee of actual customer behavior."
        )

        recommendation = (
            "Consider prioritizing this session for targeted follow-up, "
            "personalized messaging, or a relevant ecommerce offer."
        )

    else:

        summary = (
            f"The model estimates a {probability:.2f}% conversion "
            "probability for this ecommerce session, indicating that "
            "the visitor is unlikely to convert. "
            "This is a model estimate based on the supplied session "
            "information and is not a guarantee of actual customer behavior."
        )

        recommendation = (
            "Consider using lower-cost nurturing, retargeting, or additional "
            "engagement before prioritizing this visitor for direct outreach."
        )

    context_parts = []

    if visitor_type:
        context_parts.append(
            f"visitor type: {visitor_type}"
        )

    if month:
        context_parts.append(
            f"month: {month}"
        )

    if product_pages is not None:
        context_parts.append(
            f"product-related pages: {product_pages}"
        )

    if product_duration is not None:
        context_parts.append(
            f"product-related duration: {product_duration} seconds"
        )

    if context_parts:
        context = "; ".join(
            context_parts
        )

        summary += (
            f" Session context includes {context}."
        )

    return {
        "summary": summary,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------
# Complete LLM + ML pipeline
# ---------------------------------------------------------

def analyze_session(user_text):
    """
    Complete end-to-end ecommerce conversion analysis.

    Natural language
        -> Nebius LLM feature extraction
        -> validation
        -> trained ML model
        -> deterministic business explanation
    """

    features = parse_user_input(
        user_text
    )

    prediction_result = predict_conversion(
        features
    )

    # -----------------------------------------------------
    # Handle missing information
    # -----------------------------------------------------

    if not prediction_result["success"]:

        if (
            prediction_result["error_type"]
            == "missing_features"
        ):

            return {
                "success": False,
                "error_type": "missing_features",
                "features": features,
                "missing_features": prediction_result[
                    "missing_features"
                ],
                "message": (
                    "I need more information before I can "
                    "estimate conversion probability."
                ),
            }

        return {
            "success": False,
            "error_type": prediction_result[
                "error_type"
            ],
            "features": features,
            "message": prediction_result[
                "message"
            ],
        }

    # -----------------------------------------------------
    # Generate explanation
    # -----------------------------------------------------

    explanation = generate_business_explanation(
        features,
        prediction_result,
    )

    return {
        "success": True,
        "features": features,
        "prediction": prediction_result[
            "prediction"
        ],
        "conversion_probability": prediction_result[
            "conversion_probability"
        ],
        "conversion_percentage": prediction_result[
            "conversion_percentage"
        ],
        "summary": explanation[
            "summary"
        ],
        "recommendation": explanation[
            "recommendation"
        ],
    }


# ---------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------

def test_full_pipeline():
    """
    Test the complete LLM + ML workflow.
    """

    sample_query = (
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

    result = analyze_session(
        sample_query
    )

    print("\nEnd-to-End Conversion Analysis")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    test_full_pipeline()