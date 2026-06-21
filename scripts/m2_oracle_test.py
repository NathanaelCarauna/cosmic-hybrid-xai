#!/usr/bin/env python3
"""Module 2 Oracle Test — measures ceiling accuracy of the rule engine.

Creates manual NormalizedRequirement instances for 20 De Vito FPs
and runs them through CosmicRuleEngine. If IR-Full oracle produces
MAE > 0, the rule engine has a bug.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE / "models" / "hybrid-xai" / "schema.py"
ENGINE_PATH = BASE / "models" / "hybrid-xai" / "rule_engine.py"
GT_PATH = (
    BASE
    / "results"
    / "cosmet"
    / "ALBERGATE-gpt-4-turbo-20260305T023001"
    / "measurements"
    / "comparison_results.csv"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    return mod


def _init_modules():
    pkg_init = BASE / "models" / "hybrid-xai" / "__init__.py"
    _load_module("models.hybrid_xai", pkg_init)
    schema_mod = _load_module("models.hybrid_xai.schema", SCHEMA_PATH)
    engine_mod = _load_module("models.hybrid_xai.rule_engine", ENGINE_PATH)

    pkg = sys.modules["models.hybrid_xai"]
    pkg.schema = schema_mod
    pkg.rule_engine = engine_mod

    for name in ["models.hybrid_xai", "models.hybrid_xai.schema", "models.hybrid_xai.rule_engine"]:
        sys.modules[name].__spec__.loader.exec_module(sys.modules[name])

    return schema_mod, engine_mod


schema_mod, engine_mod = _init_modules()

NR = schema_mod.NormalizedRequirement
FP = schema_mod.FunctionalProcess
SP = schema_mod.SubProcess
DM = schema_mod.DataMovement
BC = schema_mod.BoundaryContext
M = schema_mod.Metadata
SF = schema_mod.SourceFormat
MT = schema_mod.MovementType
DIR = schema_mod.Direction
CosmicRuleEngine = engine_mod.CosmicRuleEngine

# ---------------------------------------------------------------------------
# Shorthand helpers
# ---------------------------------------------------------------------------

def _e(dg: str, ooi: str, fu: str, trigger: bool = False) -> DM:
    return DM(type=MT.E, data_group=dg, object_of_interest=ooi,
              functional_user=fu, is_triggering_entry=trigger)

def _x(dg: str, ooi: str, fu: str, err: bool = False) -> DM:
    return DM(type=MT.X, data_group=dg, object_of_interest=ooi,
              functional_user=fu, is_error_confirmation=err)

def _r(dg: str, ooi: str) -> DM:
    return DM(type=MT.R, data_group=dg, object_of_interest=ooi)

def _w(dg: str, ooi: str) -> DM:
    return DM(type=MT.W, data_group=dg, object_of_interest=ooi)

def _fp(name: str, event: str, actor: str, movements: list[DM]) -> FP:
    return FP(name=name, triggering_event=event,
              primary_functional_user=actor, data_movements=movements)

def _nr(fp_id: str, name: str, actor: str, movements: list[DM]) -> NR:
    return NR(
        id=fp_id, source_format=SF.USE_CASE, raw_text="(oracle)",
        boundary=BC(software_name="Albergate", functional_users=[actor]),
        functional_processes=[_fp(name, name, actor, movements)],
    )

# ---------------------------------------------------------------------------
# Oracle annotations for 20 De Vito FPs
#
# Movement analysis based on reading each UC text and applying COSMIC rules:
#   - Entries: actor enters/selects data for the system
#   - Reads: system searches/fetches from persistent storage
#   - Writes: system saves/updates/deletes to persistent storage
#   - Exits: system displays/lists data to actor
#   - Error/confirmation Exits: is_error_confirmation=True, collapsed to 1
#   - Control commands (click button, navigate): excluded
# ---------------------------------------------------------------------------

HM = "Hotel manager"
OP = "Operator"

ORACLE: list[NR] = [
    # FP1: Search for rooms (GT=4)
    # E: HM fills room code filter | R: search rooms | X: display list | X(err): no results
    _nr("FP1", "Search for rooms", HM, [
        _e("Room code filter", "Room", HM, trigger=True),
        _r("Room records", "Room"),
        _x("Rooms list", "Room", HM),
        _x("Error message", "Room", HM, err=True),
    ]),

    # FP2: View room details (GT=4)
    # E: HM selects room | R: search room by code | X: display details | X(err): not found
    _nr("FP2", "View room details", HM, [
        _e("Room selection", "Room", HM, trigger=True),
        _r("Room record", "Room"),
        _x("Room details", "Room", HM),
        _x("Error message", "Room", HM, err=True),
    ]),

    # FP3: Edit room details (GT=6)
    # E: HM selects room to edit | R: check booked room file | X: display room page
    # E: HM modifies fields | W: save room record | X(err/conf): messages (1x)
    _nr("FP3", "Edit room details", HM, [
        _e("Room selection", "Room", HM, trigger=True),
        _r("Booked room file", "Room"),
        _x("Room details page", "Room", HM),
        _e("Modified room details", "Room", HM),
        _w("Room record", "Room"),
        _x("Confirmation message", "Room", HM, err=True),
    ]),

    # FP4: Insert new room (GT=3)
    # E: HM fills room fields | W: save room record | X(err/conf): messages (1x)
    _nr("FP4", "Insert new room", HM, [
        _e("Room details", "Room", HM, trigger=True),
        _w("Room record", "Room"),
        _x("Confirmation message", "Room", HM, err=True),
    ]),

    # FP5: Search for services (GT=4) — parallel to FP1
    _nr("FP5", "Search for services", HM, [
        _e("Service code filter", "Service", HM, trigger=True),
        _r("Service records", "Service"),
        _x("Services list", "Service", HM),
        _x("Error message", "Service", HM, err=True),
    ]),

    # FP6: View service details (GT=4) — parallel to FP2
    _nr("FP6", "View service details", HM, [
        _e("Service selection", "Service", HM, trigger=True),
        _r("Service record", "Service"),
        _x("Service details", "Service", HM),
        _x("Error message", "Service", HM, err=True),
    ]),

    # FP7: Edit service details (GT=3)
    # Simpler than FP3: no "check occupancy" step
    # E: HM modifies service | W: save | X(err/conf): messages
    _nr("FP7", "Edit service details", HM, [
        _e("Modified service details", "Service", HM, trigger=True),
        _w("Service record", "Service"),
        _x("Confirmation message", "Service", HM, err=True),
    ]),

    # FP8: Insert new service (GT=3) — parallel to FP4
    _nr("FP8", "Insert new service", HM, [
        _e("Service details", "Service", HM, trigger=True),
        _w("Service record", "Service"),
        _x("Confirmation message", "Service", HM, err=True),
    ]),

    # FP9: Search for customers (GT=4) — parallel to FP1
    _nr("FP9", "Search for customers", OP, [
        _e("Customer filters", "Customer", OP, trigger=True),
        _r("Customer records", "Customer"),
        _x("Customers list", "Customer", OP),
        _x("Error message", "Customer", OP, err=True),
    ]),

    # FP12: Insert new customer (GT=3) — parallel to FP4
    _nr("FP12", "Insert new customer", OP, [
        _e("Customer details", "Customer", OP, trigger=True),
        _w("Customer record", "Customer"),
        _x("Confirmation message", "Customer", OP, err=True),
    ]),

    # FP13: Search for seasons config (GT=4) — parallel to FP1
    _nr("FP13", "Search for seasons", HM, [
        _e("Season filters", "Season", HM, trigger=True),
        _r("Season records", "Season"),
        _x("Seasons list", "Season", HM),
        _x("Error message", "Season", HM, err=True),
    ]),

    # FP14: View season details (GT=4) — parallel to FP2
    _nr("FP14", "View season details", HM, [
        _e("Season selection", "Season", HM, trigger=True),
        _r("Season record", "Season"),
        _x("Season details", "Season", HM),
        _x("Error message", "Season", HM, err=True),
    ]),

    # FP16: Remove season configuration (GT=3)
    # E: HM selects config to delete + confirms | W: remove record | X(err/conf)
    _nr("FP16", "Remove season config", HM, [
        _e("Season selection", "Season", HM, trigger=True),
        _w("Season record", "Season"),
        _x("Confirmation message", "Season", HM, err=True),
    ]),

    # FP17: Insert new season configuration (GT=3) — parallel to FP4
    _nr("FP17", "Insert season config", HM, [
        _e("Season details", "Season", HM, trigger=True),
        _w("Season record", "Season"),
        _x("Confirmation message", "Season", HM, err=True),
    ]),

    # FP18: View rooms & services config (GT=4) — parallel to FP2
    _nr("FP18", "View config", HM, [
        _e("Config request", "Configuration", HM, trigger=True),
        _r("Config parameters", "Configuration"),
        _x("Config details", "Configuration", HM),
        _x("Error message", "Configuration", HM, err=True),
    ]),

    # FP19: Edit rooms & services config (GT=3)
    _nr("FP19", "Edit config", HM, [
        _e("Modified config", "Configuration", HM, trigger=True),
        _w("Config record", "Configuration"),
        _x("Confirmation message", "Configuration", HM, err=True),
    ]),

    # FP20: View room services (GT=4) — parallel to FP2
    _nr("FP20", "View room services", HM, [
        _e("Room selection", "Room-Service", HM, trigger=True),
        _r("Room-service records", "Room-Service"),
        _x("Room services list", "Room-Service", HM),
        _x("Error message", "Room-Service", HM, err=True),
    ]),

    # FP21: Insert new room service (GT=6)
    # E: HM fills service code filter | R: search services | X: display list
    # E: HM selects service | W: save room-service | X(err/conf): messages
    _nr("FP21", "Insert room service", HM, [
        _e("Service code filter", "Service", HM, trigger=True),
        _r("Service records", "Service"),
        _x("Services list", "Service", HM),
        _e("Service selection", "Room-Service", HM),
        _w("Room-service record", "Room-Service"),
        _x("Confirmation message", "Room-Service", HM, err=True),
    ]),

    # FP22: Remove room service (GT=3)
    _nr("FP22", "Remove room service", HM, [
        _e("Service selection", "Room-Service", HM, trigger=True),
        _w("Room-service record", "Room-Service"),
        _x("Confirmation message", "Room-Service", HM, err=True),
    ]),

    # FP24: Blocked rooms management (GT=5)
    # E: Timer trigger | R: Config (max block time) | R: Reservations (overdue)
    # W: Empty warnings file | W: Save warning IDs
    _nr("FP24", "Blocked rooms mgmt", "System timer", [
        _e("Timer event", "Reservation", "System timer", trigger=True),
        _r("Config max block time", "Configuration"),
        _r("Overdue reservations", "Reservation"),
        _w("Clear warnings file", "Warning"),
        _w("Warning IDs", "Warning"),
    ]),
]


def load_ground_truth() -> dict[str, int]:
    gt: dict[str, int] = {}
    with open(GT_PATH) as f:
        for row in csv.DictReader(f):
            gt[row["fp"]] = int(row["y_true"])
    return gt


def main():
    engine = CosmicRuleEngine()
    gt = load_ground_truth()

    print("=" * 65)
    print("Module 2 Oracle Test — COSMIC Rule Engine Accuracy Ceiling")
    print("=" * 65)
    print(f"{'FP':>5} | {'GT':>3} | {'M2':>3} | {'Err':>4} | Status")
    print("-" * 65)

    errors: list[int] = []
    abs_errors: list[float] = []

    for nr in ORACLE:
        result = engine.measure(nr)
        gt_cfp = gt[nr.id]
        pred_cfp = result.total_cfp
        err = pred_cfp - gt_cfp
        errors.append(err)
        abs_errors.append(abs(err))
        status = "OK" if err == 0 else "MISMATCH"
        mark = "  " if err == 0 else " *"
        print(f"{nr.id:>5} | {gt_cfp:>3} | {pred_cfp:>3} | {err:>+4} | {status}{mark}")

        if err != 0:
            pr = result.process_results[0]
            if pr.corrections:
                for c in pr.corrections:
                    print(f"       -> correction: {c}")
            if pr.removed:
                for reason, dm in pr.removed:
                    print(f"       -> removed ({reason}): {dm.type.value} {dm.data_group}")

    print("-" * 65)

    n = len(errors)
    mae = sum(abs_errors) / n
    exact_matches = sum(1 for e in errors if e == 0)

    print(f"\nResults ({n} FPs):")
    print(f"  MAE:            {mae:.3f}")
    print(f"  Exact matches:  {exact_matches}/{n} ({100*exact_matches/n:.1f}%)")
    print(f"  Mean error:     {sum(errors)/n:+.3f}")

    if mae == 0:
        print("\n  PASS: M2 rule engine produces MAE=0 with perfect oracle IR.")
    elif mae < 0.5:
        print(f"\n  WARN: MAE={mae:.3f} — minor annotation differences, review mismatches.")
    else:
        print(f"\n  FAIL: MAE={mae:.3f} — rule engine has bugs or oracle annotations are wrong.")


if __name__ == "__main__":
    main()
