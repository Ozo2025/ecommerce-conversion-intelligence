import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.app import (
    FEATURE_SCHEMA,
    predict_conversion,
)


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
# JSON extraction helper
# ---------------------------------------------------------

def extract_json_object(text, required_keys=None):
    """
    Extract the last valid JSON object from an LLM response.

    This makes the application resilient when a model places
    explanatory or reasoning text before the final JSON object.
    """

    required_keys = required_keys or []

    decoder = json.JSONDecoder()
    valid_objects = []

    for index, character in enumerate(text):
        if character != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(text[index:])

            if isinstance(obj, dict):
                if all(key in obj for key in required_keys):
                    valid_objects.append(obj)

        except json.JSONDecodeError:
            continue

    if not valid_objects:
        raise ValueError(
            "No valid JSON object containing the required fields "
            "was found in the LLM response."
        )

    return valid_objects[-1]


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
4. Return valid JSON.
5. Do not add fields that are not in the schema.

User input:

{user_text}
"""

    return prompt


# ---------------------------------------------------------
# Natural language -> structured model features
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
                    "Extract ecommerce model features from the "
                    "user input and provide the final result as JSON."
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

    parsed = extract_json_object(
        content,
        required_keys=list(FEATURE_SCHEMA.keys()),
    )

    cleaned = {}

    for feature_name in FEATURE_SCHEMA:
        cleaned[feature_name] = parsed.get(
            feature_name
        )

    return cleaned


# ---------------------------------------------------------
# LLM-generated business explanation
# ---------------------------------------------------------

def generate_business_explanation(
    features,
    prediction_result,
):
    """
    Ask the LLM to explain the trained model prediction
    in concise business language.
    """

    client = get_llm_client()
    model_name = get_model_name()

    probability = prediction_result[
        "conversion_percentage"
    ]

    prediction = prediction_result[
        "prediction"
    ]

    prompt = f"""
You are explaining an ecommerce conversion prediction
to a sales or ecommerce professional.

Prediction:
{"Likely to Convert" if prediction else "Unlikely to Convert"}

Conversion probability:
{probability:.2f}%

Structured session features:
{json.dumps(features, indent=2)}

Create a concise business-facing response.

Return a JSON object containing exactly:

summary
recommendation

Requirements:

- The summary must state the {probability:.2f}% conversion probability.
- Explain that the prediction is a model estimate, not a guarantee.
- Refer only to the supplied session information.
- Do not claim that an individual feature caused the result.
- Keep the summary under 70 words.
- Give one practical ecommerce or sales recommendation.
- Keep the recommendation under 40 words.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a concise business explanation of the "
                    "machine learning prediction. End with a JSON object "
                    "containing summary and recommendation."
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
    )

    content = response.choices[0].message.content

    explanation = extract_json_object(
        content,
        required_keys=[
            "summary",
            "recommendation",
        ],
    )

    return {
        "summary": str(
            explanation["summary"]
        ).strip(),
        "recommendation": str(
            explanation["recommendation"]
        ).strip(),
    }


# ---------------------------------------------------------
# Complete LLM + ML pipeline
# ---------------------------------------------------------

def analyze_session(user_text):
    """
    Complete end-to-end ecommerce conversion analysis.

    Natural language
        -> Nebius LLM feature extraction
        -> feature validation
        -> trained ML model
        -> Nebius LLM business explanation
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
    # Generate LLM explanation
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