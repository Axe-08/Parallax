"""Unit tests for Regex rule engine and unit normalizer."""

import pandas as pd

from parallax.models.rules import RegexRuleEngine, normalize_unit


def test_normalize_unit():
    assert normalize_unit("g") == "gram"
    assert normalize_unit("gms") == "gram"
    assert normalize_unit("KG") == "kilogram"
    assert normalize_unit("ml") == "millilitre"
    assert normalize_unit("cm") == "centimetre"
    assert normalize_unit("unknown_unit") == "unknown_unit"


def test_regex_rule_engine_extraction():
    engine = RegexRuleEngine(text_cols=["title", "description"])
    df = pd.DataFrame(
        {
            "index": [1, 2, 3],
            "title": [
                "Organic Almonds 500 g Pack",
                "Stainless Steel Ruler 30 cm",
                "Generic Unlabeled Item",
            ],
            "description": ["", "", ""],
        }
    )
    engine.fit(df, target_col="title")
    preds = engine.predict(df)

    assert preds[0] == "500 gram"
    assert preds[1] == "30 centimetre"
    assert preds[2] == engine.default_fallback
