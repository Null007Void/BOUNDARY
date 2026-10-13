"""Deterministic monetary primitives for BOUNDARY.

Money is represented as :class:`decimal.Decimal` restricted to at most two
decimal places (USD cent precision) and rounded half-up (ties away from zero).
Floating-point is rejected at the boundary so binary-float rounding can never
enter financial arithmetic. Final amounts are carried as integer counts of USD
cents.

Cent conversions and quantity*price products run under a local decimal context
sized to the value's magnitude, so the ambient context precision cannot round an
intermediate result across a half-cent boundary nor raise ``InvalidOperation``.
Magnitude is bounded by an engineering safety limit (not business policy) so
exponential notation cannot smuggle in pathological values.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

_CENT = Decimal("0.01")

# Engineering safety bound (NOT business policy): maximum base-10 magnitude
# (adjusted exponent) permitted for a monetary value. Values whose integer part
# would occupy more than this many digits are rejected before any arithmetic.
# 26 permits values < 10**26 — far beyond any real transaction.
MAX_MONEY_MAGNITUDE = 26


def _cent_precision(value: Decimal) -> int:
    """Context precision required to represent ``value`` quantized to cents."""
    coefficient_digits = len(value.as_tuple().digits)
    integer_digits = max(value.adjusted() + 1, 1)  # digits left of the decimal point
    return max(coefficient_digits, integer_digits + 2) + 4


def round_money(value: Decimal) -> Decimal:
    """Round to the nearest cent, ties away from zero (ROUND_HALF_UP)."""
    with localcontext() as ctx:
        ctx.prec = max(_cent_precision(value), 28)
        return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def to_cents(value: Decimal) -> int:
    """Convert a monetary Decimal to an integer count of USD cents."""
    with localcontext() as ctx:
        ctx.prec = max(_cent_precision(value), 28)
        cents = value.quantize(_CENT, rounding=ROUND_HALF_UP) * 100
    return int(cents)


def parse_money(value) -> Decimal:
    """Coerce a JSON-compatible scalar to a validated monetary Decimal.

    Accepts ``str``, ``int``, or ``Decimal``. Rejects ``float`` (forbidden),
    ``bool``/``None``, non-numeric text, non-finite values, sub-cent precision
    (more than two decimal places), and values exceeding ``MAX_MONEY_MAGNITUDE``.
    """
    if value is None or isinstance(value, bool):
        raise ValueError("money value must be a numeric scalar, not bool/None")
    if isinstance(value, float):
        raise ValueError("floating-point money values are forbidden; use a decimal string")
    if isinstance(value, int):
        d = Decimal(value)
    elif isinstance(value, Decimal):
        d = value
    elif isinstance(value, str):
        try:
            d = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid money literal: {value!r}") from exc
    else:
        raise ValueError(f"unsupported money type: {type(value).__name__}")
    if not d.is_finite():
        raise ValueError("money value must be finite")
    if d.as_tuple().exponent < -2:
        raise ValueError("money value has more than two decimal places")
    if d.adjusted() >= MAX_MONEY_MAGNITUDE:
        raise ValueError(f"money value exceeds {MAX_MONEY_MAGNITUDE} integer digits")
    return d


def multiply_to_cents(quantity: Decimal, unit_price: Decimal) -> int:
    """Exact ``quantity * unit_price`` rounded half-up to integer USD cents.

    Sizes the local context by both the coefficient digits (exact product) and
    the product's magnitude (integer digits after cent quantization), so neither
    the ambient context precision nor a large exponent can cause rounding or an
    ``InvalidOperation``.
    """
    coefficient_digits = len(quantity.as_tuple().digits) + len(unit_price.as_tuple().digits)
    magnitude = quantity.adjusted() + unit_price.adjusted()
    integer_digits = max(magnitude + 1, 1) + 2
    with localcontext() as ctx:
        ctx.prec = max(coefficient_digits, integer_digits) + 4
        product = quantity * unit_price
        cents = product.quantize(_CENT, rounding=ROUND_HALF_UP) * 100
    return int(cents)
