"""Runtime data contracts for BOUNDARY's offline deterministic pipeline.

These models describe the *input* surface: a synthetic invoice-to-purchase-order
case with trusted lookup results, an optional starting-ledger snapshot, optional
evidence, a policy version, and an input identity. They carry no expected
outcomes, scenario families, or split labels; every model forbids extra fields,
so any ground-truth-like key is rejected at parse time rather than leaking into
runtime.

Representation vs. policy: this layer validates *shape and representation*
(Decimal money at <=2 decimal places, positive finite quantities, strict integer
cents, a USD-only currency, non-blank required identifiers). It does not encode
business policy such as sign rules, thresholds, or discrepancy decisions — those
are surfaced as assumptions in CONTRACTS.md and are left to later layers rather
than invented here.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictInt,
)

from .money import multiply_to_cents, parse_money, to_cents

# Cap on significant coefficient digits for a quantity value. Allows fractional
# quantities (including high-precision values) but rejects pathological input.
MAX_QUANTITY_DIGITS = 40

# Engineering safety bound (NOT business policy): maximum base-10 magnitude
# (adjusted exponent) permitted for a quantity value. Prevents exponential
# notation from smuggling in pathological magnitudes.
MAX_QUANTITY_MAGNITUDE = 26

# --- Scalar types -------------------------------------------------------------------

Money = Annotated[Decimal, BeforeValidator(parse_money)]


def _parse_positive_decimal(value) -> Decimal:
    """Coerce a positive Decimal, rejecting float/bool/None/non-numeric/non-finite."""
    if value is None or isinstance(value, bool):
        raise ValueError("quantity must be a numeric scalar, not bool/None")
    if isinstance(value, float):
        raise ValueError("floating-point quantity is forbidden; use a decimal string")
    if isinstance(value, int):
        d = Decimal(value)
    elif isinstance(value, Decimal):
        d = value
    elif isinstance(value, str):
        try:
            d = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid quantity literal: {value!r}") from exc
    else:
        raise ValueError(f"unsupported quantity type: {type(value).__name__}")
    if not d.is_finite():
        raise ValueError("quantity must be finite")
    if d <= 0:
        raise ValueError("quantity must be positive")
    if len(d.as_tuple().digits) > MAX_QUANTITY_DIGITS:
        raise ValueError(f"quantity exceeds {MAX_QUANTITY_DIGITS} significant digits")
    if d.adjusted() >= MAX_QUANTITY_MAGNITUDE:
        raise ValueError(f"quantity exceeds {MAX_QUANTITY_MAGNITUDE} integer digits")
    return d


Quantity = Annotated[Decimal, BeforeValidator(_parse_positive_decimal)]


def _reject_blank(value: str) -> str:
    """Reject whitespace-only strings without altering non-blank content."""
    if not value.strip():
        raise ValueError("identifier must contain at least one non-whitespace character")
    return value


NonBlank = Annotated[str, AfterValidator(_reject_blank)]


# --- Enumerated states --------------------------------------------------------------

class POState(str, Enum):
    """Trusted purchase-order lookup outcome (three distinct states)."""
    FOUND = "FOUND"                # a purchase order was retrieved
    UNAVAILABLE = "UNAVAILABLE"    # lookup ran but could not return a result
    NOT_FOUND = "NOT_FOUND"        # lookup ran and definitively found nothing


class SupplierMatchState(str, Enum):
    """Trusted supplier-identity comparison outcome (three distinct states)."""
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNRESOLVED = "UNRESOLVED"


# --- Document models ----------------------------------------------------------------

class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_number: Annotated[int, Field(strict=True, gt=0)]
    description: NonBlank
    quantity: Quantity
    unit_price: Money

    @property
    def line_total_cents(self) -> int:
        """Line total in integer cents, rounded half-up, computed exactly."""
        return multiply_to_cents(self.quantity, self.unit_price)


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: NonBlank
    supplier_id: NonBlank
    po_reference: Optional[NonBlank] = None  # PO the invoice claims to charge against
    currency: Literal["USD"] = "USD"
    issued_date: NonBlank  # opaque; ISO-8601 recommended
    line_items: list[LineItem] = Field(min_length=1)
    stated_total: Money

    @property
    def stated_total_cents(self) -> int:
        return to_cents(self.stated_total)

    @property
    def computed_total_cents(self) -> int:
        return sum(item.line_total_cents for item in self.line_items)


class PurchaseOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    po_id: NonBlank
    supplier_id: NonBlank
    currency: Literal["USD"] = "USD"
    issued_date: NonBlank
    line_items: list[LineItem] = Field(min_length=1)
    stated_total: Money

    @property
    def stated_total_cents(self) -> int:
        return to_cents(self.stated_total)

    @property
    def computed_total_cents(self) -> int:
        return sum(item.line_total_cents for item in self.line_items)


# --- Trusted lookup results ----------------------------------------------------------

class POLookupFound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["FOUND"] = "FOUND"
    purchase_order: PurchaseOrder


class POLookupUnavailable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["UNAVAILABLE"] = "UNAVAILABLE"


class POLookupNotFound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["NOT_FOUND"] = "NOT_FOUND"


POLookupResult = Annotated[
    Union[POLookupFound, POLookupUnavailable, POLookupNotFound],
    Field(discriminator="state"),
]


class SupplierLookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: SupplierMatchState


# --- Case input ----------------------------------------------------------------------

class LedgerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # account identifier -> signed integer USD cents balance (strict int only)
    balances: dict[str, StrictInt] = Field(default_factory=dict)


class CaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: NonBlank  # opaque; never parsed semantically
    invoice: Invoice
    po_lookup: POLookupResult
    supplier_lookup: SupplierLookupResult
    starting_ledger: Optional[LedgerSnapshot] = None
    evidence: list[str] = Field(default_factory=list)  # opaque evidence identifiers
    policy_version: NonBlank
    input_identity: NonBlank  # provenance identity of the input
