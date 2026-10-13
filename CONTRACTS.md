# BOUNDARY — Runtime Data Contracts (B1.1a, corrective pass)

Foundational input contracts for the offline deterministic invoice-to-purchase-order
processing path. This layer defines *shape and representation* only; it makes no
business-policy decisions.

## Schema

- `CaseInput` — top-level input: opaque `case_id`, `invoice`, `po_lookup`,
  `supplier_lookup`, optional `starting_ledger`, optional `evidence` (opaque IDs),
  `policy_version`, `input_identity`.
- `Invoice` / `PurchaseOrder` — `*_id`, `supplier_id`, `currency` (USD only),
  `issued_date` (opaque), non-empty `line_items`, `stated_total`, and a derived
  `computed_total_cents`. `Invoice` also carries an optional `po_reference`
  (B1.1b extension): the PO the invoice claims to charge against; `None` means no
  association is declared.
- `LineItem` — `line_number` (strict positive int), `description`, positive
  `quantity`, `unit_price`, and derived `line_total_cents`.
- `POLookupResult` — discriminated union: `FOUND` (carries a `PurchaseOrder`),
  `UNAVAILABLE`, `NOT_FOUND` (the latter two carry no order).
- `SupplierLookupResult` — `MATCH`, `MISMATCH`, or `UNRESOLVED`.
- `LedgerSnapshot` — `balances` mapping account identifier → signed integer cents
  (**strict int only**: floats, booleans, and numeric strings are rejected).

## Money

- `Decimal`, max two decimal places (USD cents), rounded `ROUND_HALF_UP`.
- Floating-point inputs are rejected at the boundary.
- Final amounts are integer USD cents (`to_cents`).
- Products (`quantity * unit_price`) are computed exactly under a local decimal
  context sized to the operands, so the ambient context precision cannot round an
  intermediate result across a half-cent boundary (`multiply_to_cents`).

## Strict typing

- `line_number` and ledger balances accept **actual integers only**; floats
  (including `1.0`), booleans, and numeric strings are rejected.
- `currency` is restricted to the literal `USD`; any other value is rejected. No
  foreign-exchange handling is implemented.
- Required identifiers reject whitespace-only values, but non-blank content is
  preserved exactly (never stripped or normalized).

## Numeric limits (surfaced, engineering-safety bounds — not business policy)

- **Integer magnitude** is bounded by the adjusted base-10 exponent
  (`MAX_MONEY_MAGNITUDE = MAX_QUANTITY_MAGNITUDE = 26`): no value may occupy more
  than 26 digits left of the decimal point (values `< 10**26`). This bounds both
  ordinary magnitudes and exponential notation (e.g. `1E+29` is rejected).
- **Quantity coefficient digits** are additionally capped at
  `MAX_QUANTITY_DIGITS = 40` significant digits, permitting high-precision
  fractional quantities while rejecting pathological fractions.
- Every accepted value converts to integer cents without an unhandled decimal
  exception: cent conversions and products run under a local context sized to the
  value's magnitude, and the magnitude bound guarantees that sizing stays finite.
- Values exceeding a limit fail with a controlled `ValueError`, never an unhandled
  arithmetic exception. These are safety bounds against pathological input, not
  business thresholds.

## Deliberate non-decisions (surfaced, not invented)

1. **Money sign** — monetary fields are signed; a negative value may represent a
   credit. No sign policy is enforced here (deferred to business-policy layer).
2. **Currency** — a single currency field is restricted to `USD`; multi-currency
   conversion and FX policy are out of scope.
3. **Dates** — `issued_date` is an opaque string (ISO-8601 recommended); no
   date-parsing or calendar policy is applied.
4. **Arithmetic discrepancy** — `stated_total` and `computed_total_cents` are both
   exposed but *not* reconciled by the contracts; the exact-match classifier
   (B1.1b) surfaces a mismatch as an `INVOICE_ARITHMETIC_MISMATCH` /
   `PO_ARITHMETIC_MISMATCH` reason code without acting on it.
5. **Evidence structure** — `evidence` is a list of opaque identifiers; content
   schema is deferred.
6. **Quantity precision** — quantities must be positive and finite; fractional
   quantities (weights/hours) are allowed up to `MAX_QUANTITY_DIGITS`.

## Ground-truth isolation

Every model forbids extra fields (`extra="forbid"`). Any expected-outcome,
scenario-family, split, label, or ground-truth key is rejected at parse time and
cannot reach runtime state.
