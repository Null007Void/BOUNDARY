from decimal import Decimal

import pytest

from boundary.money import parse_money, round_money, to_cents


def test_round_half_up_positive_tie():
    assert to_cents(Decimal("0.005")) == 1        # 0.005 -> 0.01
    assert round_money(Decimal("1.005")) == Decimal("1.01")
    assert to_cents(Decimal("1.015")) == 102      # 1.015 -> 1.02


def test_round_half_up_does_not_round_down():
    assert to_cents(Decimal("0.004")) == 0        # 0.004 -> 0.00
    assert to_cents(Decimal("1.014")) == 101      # 1.014 -> 1.01


def test_exact_cent_values_preserved():
    assert to_cents(Decimal("100.00")) == 10000
    assert to_cents(Decimal("0.01")) == 1
    assert to_cents(Decimal("1234567.89")) == 123456789


def test_parse_money_accepts_decimal_int_str():
    assert parse_money("10.50") == Decimal("10.50")
    assert parse_money(10) == Decimal("10")
    assert parse_money(Decimal("3.33")) == Decimal("3.33")


def test_parse_money_rejects_float():
    with pytest.raises(ValueError):
        parse_money(1.5)


def test_parse_money_rejects_sub_cent_precision():
    with pytest.raises(ValueError):
        parse_money("1.999")


def test_parse_money_rejects_bad_input():
    for bad in ["abc", "", None, True]:
        with pytest.raises(ValueError):
            parse_money(bad)


def test_parse_money_allows_signed_value():
    # sign is a business-policy question, not a representation error
    assert parse_money("-5.00") == Decimal("-5.00")


def test_multiply_to_cents_exact():
    from boundary.money import multiply_to_cents
    assert multiply_to_cents(Decimal("1.00499999999999999999999999999"), Decimal("1.00")) == 100
    assert multiply_to_cents(Decimal("1.5"), Decimal("0.01")) == 2


def test_money_magnitude_fails_controlled():
    with pytest.raises(ValueError):
        parse_money("1" + "0" * 30 + ".00")


def test_money_exponential_magnitude_rejected():
    with pytest.raises(ValueError):
        parse_money("1E+29")
    # 27 integer digits exceeds the 26-digit magnitude bound
    with pytest.raises(ValueError):
        parse_money("123456789012345678901234567")


def test_money_extreme_exponents():
    with pytest.raises(ValueError):
        parse_money("1E+29")       # magnitude bound
    with pytest.raises(ValueError):
        parse_money("1E-29")       # sub-cent precision bound


def test_money_accepted_large_values_convert():
    assert to_cents(parse_money("1E+25")) == 10 ** 27
    assert to_cents(parse_money("9999999999999999999999999")) == 999999999999999999999999900
