from __future__ import annotations

from alpha_mining.mining.field_semantics import (
    infer_field_semantic,
    tokenize_field_name,
)


def test_tokenize_field_name_respects_word_boundaries() -> None:
    assert tokenize_field_name("moneyflow_sell_lg_amount") == (
        "moneyflow",
        "sell",
        "lg",
        "amount",
    )
    assert tokenize_field_name("tech_rsi_qfq_6") == ("tech", "rsi", "qfq", "6")


def test_moneyflow_fields_are_not_price_or_plain_liquidity() -> None:
    sell = infer_field_semantic("moneyflow_sell_lg_amount")
    buy = infer_field_semantic("moneyflow_buy_lg_amount")

    assert sell.role == "moneyflow"
    assert sell.factor_family == "moneyflow"
    assert sell.gate_family == "moneyflow_pressure"
    assert sell.is_moneyflow is True
    assert sell.is_price is False
    assert buy.role == "moneyflow"
    assert buy.is_liquidity is False


def test_price_fields_are_price() -> None:
    for name in ["close", "high", "low", "open"]:
        semantic = infer_field_semantic(name)
        assert semantic.role == "price"
        assert semantic.gate_family == "price_trend"
        assert semantic.is_price is True


def test_liquidity_fields_are_liquidity() -> None:
    for name in ["turnover_rate", "volume_ratio", "amount", "volume"]:
        semantic = infer_field_semantic(name)
        assert semantic.role == "liquidity"
        assert semantic.gate_family == "liquidity_activity"
        assert semantic.is_liquidity is True


def test_valuation_chip_and_technical_fields() -> None:
    for name in ["pe_ttm", "pb", "dv_ttm"]:
        semantic = infer_field_semantic(name)
        assert semantic.role == "valuation"
        assert semantic.bucket_family == "valuation"
        assert semantic.is_valuation is True

    chip = infer_field_semantic("cyq_winner_rate")
    assert chip.role == "chip"
    assert chip.bucket_family == "chip"
    assert chip.is_chip is True

    technical = infer_field_semantic("tech_rsi_qfq_6")
    assert technical.role == "technical"
    assert technical.bucket_family == "technical"
    assert technical.is_technical is True
