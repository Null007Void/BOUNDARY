import threading

from boundary.contracts import (
    CaseInput,
    Invoice,
    LineItem,
    POLookupFound,
    PurchaseOrder,
    SupplierLookupResult,
    SupplierMatchState,
)
from boundary.review import (
    AUTHORITY_BOUNDARY_CENTS,
    RequestStatus,
    ReviewDecision,
    ReviewStore,
    review_decision,
)


def _discrepancy_case(case_id, invoice_total, po_total, evidence=("ev-1",)):
    inv = Invoice(
        invoice_id="INV-" + case_id, supplier_id="SUP-1", po_reference="PO-1",
        issued_date="2026-10-12",
        line_items=[LineItem(line_number=1, description="item", quantity="1", unit_price=invoice_total)],
        stated_total=invoice_total,
    )
    po = PurchaseOrder(
        po_id="PO-1", supplier_id="SUP-1", issued_date="2026-10-12",
        line_items=[LineItem(line_number=1, description="item", quantity="1", unit_price=po_total)],
        stated_total=po_total,
    )
    return CaseInput(
        case_id=case_id, invoice=inv, po_lookup=POLookupFound(purchase_order=po),
        supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
        evidence=list(evidence), policy_version="policy-v1", input_identity="test",
    )


def _over_500_case(case_id="C-1"):
    # invoice $600.01 vs PO $100.00 -> discrepancy $500.01 (> $500)
    return _discrepancy_case(case_id, "600.01", "100.00")


def test_review_request_persists(tmp_path):
    path = str(tmp_path / "r.db")
    store = ReviewStore(path)
    r = store.request(_over_500_case())
    assert r.status == RequestStatus.CREATED
    assert r.decision == ReviewDecision.REQUIRES_AUTHORITY
    req = r.request
    assert req is not None
    assert req.case_id == "C-1"
    assert req.state == "WAITING_AUTHORITY"
    assert req.proposed_action == "APPROVE"
    assert req.discrepancy_cents == 50001
    assert req.amount_cents == 60001
    assert req.invoice_id == "INV-C-1"
    assert req.po_id == "PO-1"
    assert req.supplier_id == "SUP-1"
    assert req.policy_version == "policy-v1"
    assert store.count() == 1
    store.close()

    # durable across reopen
    store2 = ReviewStore(path)
    try:
        got = store2.get("C-1")
        assert got is not None
        assert got.request_id == req.request_id
        assert got.discrepancy_cents == 50001
    finally:
        store2.close()


def test_review_request_idempotent(tmp_path):
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        first = store.request(_over_500_case())
        assert first.status == RequestStatus.CREATED
        second = store.request(_over_500_case())
        assert second.status == RequestStatus.IDEMPOTENT
        assert second.request.request_id == first.request.request_id
        assert store.count() == 1
    finally:
        store.close()


def test_review_request_changed_input_conflict(tmp_path):
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        first = store.request(_over_500_case(case_id="C-1"))
        assert first.status == RequestStatus.CREATED
        # same case id, different discrepancy content
        changed = _discrepancy_case("C-1", "700.01", "100.00")
        second = store.request(changed)
        assert second.status == RequestStatus.CONFLICT
        assert store.count() == 1
        # original preserved
        assert store.get("C-1").discrepancy_cents == 50001
    finally:
        store.close()


def test_boundary_exact_500_not_required(tmp_path):
    # discrepancy exactly $500.00 -> not greater than $500 -> no review
    case = _discrepancy_case("C-500", "600.00", "100.00")
    assert review_decision(case) == ReviewDecision.NOT_REQUIRED
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.NOT_REQUIRED
        assert store.count() == 0
    finally:
        store.close()


def test_boundary_500_01_requires_authority(tmp_path):
    # discrepancy $500.01 -> exceeds $500 -> review required
    case = _discrepancy_case("C-501", "600.01", "100.00")
    assert review_decision(case) == ReviewDecision.REQUIRES_AUTHORITY
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.CREATED
        assert r.request.discrepancy_cents == 50001
    finally:
        store.close()


def test_boundary_applies_to_discrepancy_not_total(tmp_path):
    # huge invoice total but tiny discrepancy -> below boundary -> no review
    case = _discrepancy_case("C-big", "100000.00", "99999.50")  # discrepancy $0.50
    assert review_decision(case) == ReviewDecision.NOT_REQUIRED
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.NOT_REQUIRED
        assert store.count() == 0
    finally:
        store.close()


def test_insufficient_evidence(tmp_path):
    # discrepancy > $500 but no evidence -> insufficient, not a review request
    case = _discrepancy_case("C-noev", "600.01", "100.00", evidence=())
    assert review_decision(case) == ReviewDecision.INSUFFICIENT_EVIDENCE
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.INSUFFICIENT_EVIDENCE
        assert r.request is None
        assert store.count() == 0
    finally:
        store.close()


def test_review_request_never_grants_authority(tmp_path):
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(_over_500_case())
        req = r.request
        # the request is a durable "please review" record, not an execution grant
        assert req.state == "WAITING_AUTHORITY"
        assert req.proposed_action == "APPROVE"
        assert "human review required" in req.reason
        # no execution/approval fields exist on the request
        assert not hasattr(req, "execution_authorized")
        assert not hasattr(req, "committed_action")
    finally:
        store.close()


# --- B2.1 review-request safety corrections ---

def _assoc_case(case_id, po_ref, invoice_total="600.01", po_total="100.00", evidence=("ev-1",)):
    """A discrepancy case with a configurable invoice PO reference."""
    inv = Invoice(
        invoice_id="INV-" + case_id, supplier_id="SUP-1", po_reference=po_ref,
        issued_date="2026-10-12",
        line_items=[LineItem(line_number=1, description="item", quantity="1", unit_price=invoice_total)],
        stated_total=invoice_total,
    )
    po = PurchaseOrder(
        po_id="PO-1", supplier_id="SUP-1", issued_date="2026-10-12",
        line_items=[LineItem(line_number=1, description="item", quantity="1", unit_price=po_total)],
        stated_total=po_total,
    )
    return CaseInput(
        case_id=case_id, invoice=inv, po_lookup=POLookupFound(purchase_order=po),
        supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
        evidence=list(evidence), policy_version="policy-v1", input_identity="test",
    )


def test_po_association_mismatch_not_reviewable(tmp_path):
    case = _assoc_case("C-assoc", "PO-9")  # references PO-9 but found PO-1
    assert review_decision(case) == ReviewDecision.NOT_REVIEWABLE
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.NOT_REVIEWABLE
        assert "po_association_mismatch" in r.reasons
        assert store.count() == 0
    finally:
        store.close()


def test_po_association_missing_not_reviewable(tmp_path):
    case = _assoc_case("C-assoc-none", None)
    assert review_decision(case) == ReviewDecision.NOT_REVIEWABLE
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.NOT_REVIEWABLE
        assert "po_association_missing" in r.reasons
        assert store.count() == 0
    finally:
        store.close()


def test_duplicate_line_identifiers_not_reviewable(tmp_path):
    inv = Invoice(
        invoice_id="INV-dup", supplier_id="SUP-1", po_reference="PO-1",
        issued_date="2026-10-12",
        line_items=[
            LineItem(line_number=1, description="a", quantity="1", unit_price="300.01"),
            LineItem(line_number=1, description="b", quantity="1", unit_price="300.00"),
        ],
        stated_total="600.01",
    )
    po = PurchaseOrder(
        po_id="PO-1", supplier_id="SUP-1", issued_date="2026-10-12",
        line_items=[LineItem(line_number=1, description="item", quantity="1", unit_price="100.00")],
        stated_total="100.00",
    )
    case = CaseInput(
        case_id="C-dup", invoice=inv, po_lookup=POLookupFound(purchase_order=po),
        supplier_lookup=SupplierLookupResult(state=SupplierMatchState.MATCH),
        evidence=["ev-1"], policy_version="policy-v1", input_identity="test",
    )
    assert review_decision(case) == ReviewDecision.NOT_REVIEWABLE
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.NOT_REVIEWABLE
        assert "duplicate_line_identifiers" in r.reasons
        assert store.count() == 0
    finally:
        store.close()


def test_not_reviewable_distinct_from_not_required(tmp_path):
    invalid = _assoc_case("C-bad", "PO-9")  # structurally invalid
    valid_small = _discrepancy_case("C-small", "600.00", "100.00")  # discrepancy $500.00
    assert review_decision(invalid) == ReviewDecision.NOT_REVIEWABLE
    assert review_decision(valid_small) == ReviewDecision.NOT_REQUIRED


def test_changed_input_below_threshold_conflict(tmp_path):
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        assert store.request(_over_500_case(case_id="C-1")).status == RequestStatus.CREATED
        # same case id, now below threshold -> CONFLICT (not NOT_REQUIRED)
        changed = _discrepancy_case("C-1", "600.00", "100.00")  # discrepancy $500.00
        r = store.request(changed)
        assert r.status == RequestStatus.CONFLICT
        assert store.count() == 1
    finally:
        store.close()


def test_changed_input_loses_evidence_conflict(tmp_path):
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        assert store.request(_over_500_case(case_id="C-1")).status == RequestStatus.CREATED
        # same case id, now no evidence -> CONFLICT (not INSUFFICIENT_EVIDENCE)
        changed = _discrepancy_case("C-1", "600.01", "100.00", evidence=())
        r = store.request(changed)
        assert r.status == RequestStatus.CONFLICT
        assert store.count() == 1
    finally:
        store.close()


def test_storage_overflow(tmp_path):
    # invoice total = 2**63 cents (exceeds SQLite signed 64-bit range)
    big = _discrepancy_case("C-big", "92233720368547758.08", "0.01")
    assert review_decision(big) == ReviewDecision.REQUIRES_AUTHORITY  # eligible...
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(big)
        assert r.status == RequestStatus.STORAGE_OVERFLOW  # ...but cannot be persisted
        assert r.request is None
        assert store.count() == 0
    finally:
        store.close()


def test_concurrent_identical_requests(tmp_path):
    path = str(tmp_path / "r.db")
    results = []
    lock = threading.Lock()

    def worker():
        s = ReviewStore(path)
        try:
            r = s.request(_over_500_case())
            with lock:
                results.append(r.status)
        finally:
            s.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == sorted([RequestStatus.CREATED, RequestStatus.IDEMPOTENT])
    s = ReviewStore(path)
    try:
        assert s.count() == 1
    finally:
        s.close()


def test_concurrent_conflicting_requests(tmp_path):
    path = str(tmp_path / "r.db")
    results = []
    lock = threading.Lock()

    def worker(case):
        s = ReviewStore(path)
        try:
            r = s.request(case)
            with lock:
                results.append(r.status)
        finally:
            s.close()

    c1 = _over_500_case(case_id="C-1")  # discrepancy 50001
    c2 = _discrepancy_case("C-1", "700.01", "100.00")  # discrepancy 60001
    threads = [threading.Thread(target=worker, args=(c,)) for c in (c1, c2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == sorted([RequestStatus.CREATED, RequestStatus.CONFLICT])
    s = ReviewStore(path)
    try:
        assert s.count() == 1
    finally:
        s.close()


# --- B2.1 final review-intake corrections ---

def test_concurrent_case_id_precedence_deterministic(tmp_path):
    path = str(tmp_path / "r.db")

    # The "waiting" request: same case id, changed input that would be NOT_REQUIRED
    # if evaluated on its own (discrepancy exactly $500.00).
    changed = _discrepancy_case("C-1", "600.00", "100.00")

    class RaceStore(ReviewStore):
        def _evaluate(self, case):
            # Force another writer to commit the same case id between this
            # store's initial lookup and its eligibility evaluation.
            other = ReviewStore(path)
            try:
                other.request(_over_500_case(case_id="C-1"))
            finally:
                other.close()
            return super()._evaluate(case)

    store = RaceStore(path)
    try:
        result = store.request(changed)
    finally:
        store.close()

    # The changed input would be NOT_REQUIRED on its own, but the concurrent
    # commit of the same case id must yield CONFLICT (never an eligibility outcome).
    assert result.status == RequestStatus.CONFLICT
    s = ReviewStore(path)
    try:
        assert s.count() == 1  # original record preserved, never overwritten
    finally:
        s.close()


def test_blank_evidence_identifiers_insufficient(tmp_path):
    for bad_evidence in [[""], ["   "], ["", "  "]]:
        case = _discrepancy_case("C-blank", "600.01", "100.00", evidence=bad_evidence)
        assert review_decision(case) == ReviewDecision.INSUFFICIENT_EVIDENCE
        store = ReviewStore(str(tmp_path / "r.db"))
        try:
            r = store.request(case)
            assert r.status == RequestStatus.INSUFFICIENT_EVIDENCE
            assert store.count() == 0
        finally:
            store.close()


def test_mixed_evidence_with_one_valid_counts(tmp_path):
    # one blank + one nonblank identifier: the nonblank one makes it reviewable
    case = _discrepancy_case("C-mixed", "600.01", "100.00", evidence=["", "ev-1"])
    assert review_decision(case) == ReviewDecision.REQUIRES_AUTHORITY
    store = ReviewStore(str(tmp_path / "r.db"))
    try:
        r = store.request(case)
        assert r.status == RequestStatus.CREATED
    finally:
        store.close()
