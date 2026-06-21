#!/usr/bin/env python3
"""
Build manual NER annotation JSONL files from expert annotation data.

Reads FP prompt text files, applies expert-specified movement labels and
entity annotations, computes character offsets automatically, and writes
one JSONL file per system to datasets/ner-annotations-manual/.
"""

import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASETS = BASE / "replications" / "cosmet-devito" / "datasets"
OUTPUT = BASE / "datasets" / "ner-annotations-manual"
RESULTS = BASE / "results" / "cosmet"

SYSTEMS = {
    "RiceCooker": DATASETS / "Real-time" / "RiceCooker",
    "AutomaticLineSwitching": DATASETS / "Real-time" / "AutomaticLineSwitching",
    "K01720": DATASETS / "IoT_Telemedicine" / "K01720",
    "K01719": DATASETS / "IoT_Telemedicine" / "K01719",
    "K01726": DATASETS / "IoT_Telemedicine" / "K01726",
    "U-CURE": DATASETS / "ML" / "U-CURE",
    "ALBERGATE": DATASETS / "MIS" / "ALBERGATE",
}

GT_CSVS = {
    "RiceCooker": RESULTS / "RiceCooker-gpt-4-turbo-20260323T010514" / "measurements" / "comparison_results.csv",
    "AutomaticLineSwitching": RESULTS / "AutomaticLineSwitching-gpt-4-turbo-20260323T010407" / "measurements" / "comparison_results.csv",
    "K01720": RESULTS / "K01720-gpt-4-turbo-20260323T005553" / "measurements" / "comparison_results.csv",
    "K01719": RESULTS / "K01719-gpt-4-turbo-20260323T005355" / "measurements" / "comparison_results.csv",
    "K01726": RESULTS / "K01726-gpt-4-turbo-20260323T005703" / "measurements" / "comparison_results.csv",
    "U-CURE": RESULTS / "U-CURE-gpt-4-turbo-20260323T010152" / "measurements" / "comparison_results.csv",
    "ALBERGATE": RESULTS / "ALBERGATE-gpt-4-turbo-20260305T023001" / "measurements" / "comparison_results.csv",
}


def load_gt(system: str) -> dict:
    """Load ground truth from comparison_results.csv."""
    csv_path = GT_CSVS[system]
    gt = {}
    if not csv_path.exists():
        return gt
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            fp_id = row["fp"]
            gt[fp_id] = {
                "cfp_gt": int(row["y_true"]),
                "entry_gt": int(row.get("entry_gt", 0)) if "entry_gt" in row else None,
                "read_gt": int(row.get("read_gt", 0)) if "read_gt" in row else None,
                "write_gt": int(row.get("write_gt", 0)) if "write_gt" in row else None,
                "exit_gt": int(row.get("exit_gt", 0)) if "exit_gt" in row else None,
            }
    return gt


def parse_uc_text(text: str) -> list[dict]:
    """Parse UC text into steps (step_id, actor, action_text)."""
    steps = []
    in_scenario = False
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        upper = line.upper().strip()
        if upper in ("MAIN SCENARIO", "MAIN FLOW"):
            in_scenario = True
            continue
        if any(kw in upper for kw in ["ALTERNATIVE FLOW", "EXCEPTION", "REFERENCE"]):
            in_scenario = False
            continue
        if in_scenario:
            parts = line.split("\t")
            if len(parts) >= 3:
                step_id = parts[0].strip().rstrip(".")
                actor = parts[1].strip()
                action = parts[2].strip()
                steps.append({"step_id": step_id, "actor": actor, "action_text": action})
            elif len(parts) == 2:
                step_id = parts[0].strip().rstrip(".")
                action = parts[1].strip()
                steps.append({"step_id": step_id, "actor": "System", "action_text": action})
    return steps


def parse_alt_exception_steps(text: str) -> list[dict]:
    """Parse alternative flow and exception steps."""
    steps = []
    in_alt = False
    alt_prefix = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if "ALTERNATIVE FLOW" in upper or "EXCEPTION" in upper:
            in_alt = True
            alt_prefix = "ALT" if "ALTERNATIVE" in upper else "EXC"
            continue
        if "REFERENCE" in upper and in_alt:
            in_alt = False
            continue
        if "MAIN SCENARIO" in upper or "MAIN FLOW" in upper:
            in_alt = False
            continue
        if in_alt:
            parts = line.split("\t")
            if len(parts) >= 3:
                step_id = f"{alt_prefix}-{parts[0].strip().rstrip('.')}"
                actor = parts[1].strip()
                action = parts[2].strip()
                steps.append({"step_id": step_id, "actor": actor, "action_text": action})
            elif len(parts) >= 2:
                m = re.match(r"^(\d+)\.\s*(.*)", line)
                if m:
                    step_id = f"{alt_prefix}-{m.group(1)}"
                    action = m.group(2).strip()
                    steps.append({"step_id": step_id, "actor": "System", "action_text": action})
    return steps


def _norm_q(s: str) -> str:
    """Normalize curly/smart quotes to ASCII equivalents."""
    return (s.replace("\u2018", "'").replace("\u2019", "'")
             .replace("\u201c", '"').replace("\u201d", '"'))


def find_entity_spans(action_text: str, entities: list[dict]) -> list[dict]:
    """Find character offsets for entity text in action_text."""
    result = []
    norm_action = _norm_q(action_text)
    for ent in entities:
        text = ent["text"]
        label = ent["label"]
        norm_ent = _norm_q(text)
        idx = action_text.find(text)
        if idx == -1:
            idx = action_text.lower().find(text.lower())
        if idx == -1:
            idx = norm_action.find(norm_ent)
            if idx == -1:
                idx = norm_action.lower().find(norm_ent.lower())
        if idx >= 0:
            actual = action_text[idx:idx + len(text)]
            result.append({"start": idx, "end": idx + len(text), "text": actual, "label": label})
        else:
            print(f"  WARNING: entity '{text}' not found in '{action_text}'")
    return result


def build_step(step_id, actor, action_text, movement_type, data_group, reasoning, entities):
    """Build a step annotation dict."""
    spans = find_entity_spans(action_text, entities) if entities else []
    return {
        "step_id": step_id,
        "actor": actor,
        "action_text": action_text,
        "movement_type": movement_type,
        "data_group": data_group,
        "reasoning": reasoning,
        "entities": spans,
    }


def build_fp(system, fp_id, gt_data, steps_annotations):
    """Build a full FP annotation record."""
    gt = gt_data.get(fp_id, {"cfp_gt": 0})
    record = {
        "system": system,
        "fp_id": fp_id,
        "cfp_gt": gt.get("cfp_gt", 0),
        "entry_gt": gt.get("entry_gt"),
        "read_gt": gt.get("read_gt"),
        "write_gt": gt.get("write_gt"),
        "exit_gt": gt.get("exit_gt"),
        "steps": steps_annotations,
    }
    counted = sum(1 for s in steps_annotations if s["movement_type"] is not None)
    if counted != record["cfp_gt"]:
        print(f"  WARNING: {system}/{fp_id} annotated={counted} vs gt={record['cfp_gt']}")
    return record


def save_system(system: str, records: list[dict]):
    """Save all FP annotations for a system to JSONL."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{system}.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    total_steps = sum(len(r["steps"]) for r in records)
    total_movements = sum(1 for r in records for s in r["steps"] if s["movement_type"])
    print(f"  {system}: {len(records)} FPs, {total_steps} steps, {total_movements} movements → {out_path}")


# ── Expert annotations per system ──────────────────────────────────────


def annotate_ricecooker():
    system = "RiceCooker"
    gt = load_gt(system)
    records = []

    # FP1: Start cooking (E=1, R=0, W=0, X=2)
    records.append(build_fp(system, "FP1", gt, [
        build_step("1", "Start Button", "The Start Button sends the Start signal to the system.",
                   "E", "Start signal", "Triggering entry — Start Button sends signal data into the system",
                   [{"text": "Start Button", "label": "FUNC_USER"}, {"text": "sends", "label": "ACTION_VERB"}, {"text": "the Start signal", "label": "DATA_OBJ"}]),
        build_step("2", "System", "The System responds sending the 'Turn ON' signal to the  Heater.",
                   "X", "Turn ON signal (Heater)", "System sends command signal to Heater (external device)",
                   [{"text": "System", "label": "FUNC_USER"}, {"text": "sending", "label": "ACTION_VERB"}, {"text": "the 'Turn ON' signal", "label": "DATA_OBJ"}, {"text": "Heater", "label": "FUNC_USER"}]),
        build_step("3", "System", "The System sends the 'Turn ON' signal to the Cooking Lamp.",
                   "X", "Turn ON signal (Cooking Lamp)", "System sends command signal to Cooking Lamp (external device)",
                   [{"text": "System", "label": "FUNC_USER"}, {"text": "sends", "label": "ACTION_VERB"}, {"text": "the 'Turn ON' signal", "label": "DATA_OBJ"}, {"text": "Cooking Lamp", "label": "FUNC_USER"}]),
    ]))

    # FP2: Update Target temperature (E=2, R=2, W=1, X=0)
    records.append(build_fp(system, "FP2", gt, [
        build_step("1", "Timer", "The timer ticks after 30 seconds.",
                   "E", "timer tick event", "Triggering entry — timer event with timing data enters the system",
                   [{"text": "timer", "label": "FUNC_USER"}, {"text": "ticks", "label": "ACTION_VERB"}]),
        build_step("2", "Timer", "The timer receives the current elapsed time signal.",
                   "E", "elapsed time signal", "Timer sends elapsed time data into the system",
                   [{"text": "timer", "label": "FUNC_USER"}, {"text": "receives", "label": "ACTION_VERB"}, {"text": "the current elapsed time signal", "label": "DATA_OBJ"}]),
        build_step("3", "System", "The system gets the cooking mode related to the Current Settings.",
                   "R", "Current Settings (cooking mode)", "System reads cooking mode from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "gets", "label": "ACTION_VERB"}, {"text": "the cooking mode", "label": "DATA_OBJ"}, {"text": "Current Settings", "label": "PERSIST_STORE"}]),
        build_step("4", "System", "The system gets the new target temperature related to the Target settings.",
                   "R", "Target settings (target temperature)", "System reads target temperature from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "gets", "label": "ACTION_VERB"}, {"text": "the new target temperature", "label": "DATA_OBJ"}, {"text": "Target settings", "label": "PERSIST_STORE"}]),
        build_step("5", "System", "The system stores the new target temperature related to the Target settings.",
                   "W", "Target settings (target temperature)", "System writes updated target temperature to persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "stores", "label": "ACTION_VERB"}, {"text": "the new target temperature", "label": "DATA_OBJ"}, {"text": "Target settings", "label": "PERSIST_STORE"}]),
    ]))

    # FP3: Check cooker temperature (E=2, R=1, W=0, X=1)
    records.append(build_fp(system, "FP3", gt, [
        build_step("1", "Timer", "The timer ticks after 5 seconds.",
                   "E", "timer tick event", "Triggering entry — timer event enters the system",
                   [{"text": "timer", "label": "FUNC_USER"}, {"text": "ticks", "label": "ACTION_VERB"}]),
        build_step("2", "System", "The system gets the current target temperature.",
                   "R", "target temperature", "System reads target temperature from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "gets", "label": "ACTION_VERB"}, {"text": "the current target temperature", "label": "DATA_OBJ"}]),
        build_step("3", "Temperature Sensor", "The temperature Sensor sends the actual temperature to the system.",
                   "E", "actual temperature", "External sensor device sends temperature data into the system",
                   [{"text": "temperature Sensor", "label": "FUNC_USER"}, {"text": "sends", "label": "ACTION_VERB"}, {"text": "the actual temperature", "label": "DATA_OBJ"}]),
        build_step("4", "System", "The system decides if the heater must be turned on or off.",
                   None, "", "Internal decision logic, no data movement across boundary",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "decides", "label": "ACTION_VERB"}]),
        build_step("5", "Heater", "The system requests the Heater to process the command.",
                   "X", "heater command", "System sends command to external device (Heater)",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "requests", "label": "ACTION_VERB"}, {"text": "Heater", "label": "FUNC_USER"}, {"text": "the command", "label": "DATA_OBJ"}]),
    ]))

    save_system(system, records)
    return records


def annotate_automatic_line_switching():
    system = "AutomaticLineSwitching"
    gt = load_gt(system)
    records = []

    commands = [
        ("FP1", "Remove line", "'Remove line command'", "'Remove line' command"),
        ("FP2", "Restore line", "'Restore line command'", "'Restore line' command"),
        ("FP3", "Forced Line switch", "'Forced switch' command", "'Forced switch' command"),
        ("FP4", "Conditional Line switch", "'Conditional switch' command", "'Conditional switch' command"),
        ("FP5", "Handle QLCM Line failed", "'QLMC Line failed' command", "'Line switch' command"),
        ("FP6", "Handle QLCM Line degraded signal", "'QLMC Line degraded' command", "'Line switch' command"),
        ("FP7", "Handle QLCM Line cleared signal", "'QLMC Line cleared' command", "'Line switch' command"),
    ]

    for fp_id, _name, enter_cmd, switch_cmd in commands:
        fp_path = SYSTEMS[system] / "FP_prompts" / f"{fp_id}.txt"
        text = fp_path.read_text()
        main_steps = parse_uc_text(text)
        if len(main_steps) != 5:
            print(f"  WARNING: {system}/{fp_id} has {len(main_steps)} steps (expected 5)")

        actor1 = main_steps[0]["actor"] if main_steps else "CC Panel"
        is_qlcm = "QLCM" in actor1

        steps = []
        for i, ms in enumerate(main_steps):
            if i == 0:
                steps.append(build_step(ms["step_id"], ms["actor"], ms["action_text"],
                    "E", f"command ({enter_cmd})", f"Triggering entry — {ms['actor']} sends command data with line identifier",
                    [{"text": ms["actor"], "label": "FUNC_USER"}, {"text": "enters", "label": "ACTION_VERB"},
                     {"text": enter_cmd, "label": "DATA_OBJ"}]))
            elif i == 1:
                steps.append(build_step(ms["step_id"], ms["actor"], ms["action_text"],
                    "R", "Line status", "System reads current line status from persistent storage",
                    [{"text": "system", "label": "FUNC_USER"}, {"text": "reads", "label": "ACTION_VERB"},
                     {"text": "Line status", "label": "DATA_OBJ"}]))
            elif i == 2:
                steps.append(build_step(ms["step_id"], ms["actor"], ms["action_text"],
                    "W", "Line status", "System writes updated line status to persistent storage",
                    [{"text": "system", "label": "FUNC_USER"}, {"text": "writes", "label": "ACTION_VERB"},
                     {"text": "Line status", "label": "DATA_OBJ"}]))
            elif i == 3:
                steps.append(build_step(ms["step_id"], ms["actor"], ms["action_text"],
                    "X", f"command ({switch_cmd})", "System sends command to external device (Switch device)",
                    [{"text": "system", "label": "FUNC_USER"}, {"text": "requests", "label": "ACTION_VERB"},
                     {"text": "Switch device", "label": "FUNC_USER"}, {"text": switch_cmd, "label": "DATA_OBJ"}]))
            elif i == 4:
                steps.append(build_step(ms["step_id"], ms["actor"], ms["action_text"],
                    "X", "Line status (CC Panel)", "System sends updated status data to CC Panel for display",
                    [{"text": "CC Panel", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"},
                     {"text": "Line status", "label": "DATA_OBJ"}]))

        records.append(build_fp(system, fp_id, gt, steps))

    save_system(system, records)
    return records


def annotate_k01720():
    system = "K01720"
    gt = load_gt(system)
    records = []

    # FP1: Access Threshold List (E=1, R=1, W=0, X=1)
    records.append(build_fp(system, "FP1", gt, [
        build_step("1", "Actor", "The actor selects the sub-option 'Thresholds'.",
                   None, "", "UI navigation — selecting a sub-option, no domain data movement",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "the sub-option 'Thresholds'", "label": "UI_ELEMENT"}]),
        build_step("2", "System", "The system displays a form for parameter selection.",
                   None, "", "Blank form display — no domain data output",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "a form for parameter selection", "label": "UI_ELEMENT"}]),
        build_step("3", "Actor", "The actor selects a parameter from the list.",
                   "E", "parameter selection", "Triggering entry — user provides parameter selection data",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "a parameter", "label": "DATA_OBJ"}]),
        build_step("4", "System", "The system searches  for the thresholds related to the given parameter.",
                   "R", "thresholds", "System reads thresholds from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searches", "label": "ACTION_VERB"}, {"text": "the thresholds related to the given parameter", "label": "DATA_OBJ"}]),
        build_step("5", "System", "The system shows the threshold list defined for the given parameter.",
                   "X", "threshold list", "System displays threshold data to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "shows", "label": "ACTION_VERB"}, {"text": "the threshold list defined for the given parameter", "label": "DATA_OBJ"}]),
    ]))

    # FP2: Insert threshold (E=1, R=0, W=1, X=3)
    records.append(build_fp(system, "FP2", gt, [
        build_step("1", "Actor", "The actor clicks on \"add threshold\".",
                   None, "", "UI action — button click, no data movement",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"add threshold\"", "label": "UI_ELEMENT"}]),
        build_step("2", "System", "The system shows an interface with fields to be filled out.",
                   "X", "threshold form interface", "System displays form structure with field definitions to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "shows", "label": "ACTION_VERB"}, {"text": "an interface with fields to be filled out", "label": "DATA_OBJ"}]),
        build_step("3", "Actor", "The actor fills out the fields and clicks on \"save\".",
                   "E", "threshold data", "User enters threshold data and submits",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "fills out", "label": "ACTION_VERB"}, {"text": "the fields", "label": "DATA_OBJ"}]),
        build_step("4a", "System", "The system saves the set threshold.",
                   "W", "threshold", "System writes new threshold to persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "saves", "label": "ACTION_VERB"}, {"text": "the set threshold", "label": "DATA_OBJ"}]),
        build_step("4b", "System", "The system displays it in the list.",
                   "X", "threshold list", "System displays updated threshold list to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the list", "label": "DATA_OBJ"}]),
        build_step("EXC-2", "System", "The system displays an error message \"Please enter at least one threshold\".",
                   "X", "error message", "System displays validation error to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "an error message \"Please enter at least one threshold\"", "label": "DATA_OBJ"}]),
    ]))

    # FP3: Display threshold details (E=1, R=1, W=0, X=1)
    records.append(build_fp(system, "FP3", gt, [
        build_step("1", "Actor", "The actor clicks on \"view detail\" for a given threshold.",
                   "E", "threshold identifier", "User provides threshold identifier to view",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"view detail\"", "label": "UI_ELEMENT"}, {"text": "a given threshold", "label": "DATA_OBJ"}]),
        build_step("2", "System", "The system searches for the threshold details.",
                   "R", "threshold details", "System reads threshold details from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searches", "label": "ACTION_VERB"}, {"text": "the threshold details", "label": "DATA_OBJ"}]),
        build_step("3", "System", "The system displays the threshold details in a page with editable fields.",
                   "X", "threshold details", "System displays threshold data to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the threshold details", "label": "DATA_OBJ"}]),
    ]))

    # FP4: Update threshold (E=1, R=0, W=1, X=2)
    records.append(build_fp(system, "FP4", gt, [
        build_step("1", "Actor", "The actor modifies a field and clicks on save.",
                   "E", "modified threshold data", "User provides modified threshold data and submits",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "modifies", "label": "ACTION_VERB"}, {"text": "a field", "label": "DATA_OBJ"}]),
        build_step("2a", "System", "The system saves the changes.",
                   "W", "threshold", "System writes updated threshold to persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "saves", "label": "ACTION_VERB"}, {"text": "the changes", "label": "DATA_OBJ"}]),
        build_step("2b", "System", "The system displays the updated list of thresholds of the selected parameter.",
                   "X", "threshold list", "System displays updated list to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the updated list of thresholds", "label": "DATA_OBJ"}]),
        build_step("EXC-2", "System", "The system displays an error message \"Enter at least one threshold\".",
                   "X", "error message", "System displays validation error to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "an error message \"Enter at least one threshold\"", "label": "DATA_OBJ"}]),
    ]))

    # FP5: Delete threshold (E=1, R=0, W=1, X=0)
    records.append(build_fp(system, "FP5", gt, [
        build_step("1", "Actor", "The actor clicks on \"delete\" for the given threshold.",
                   "E", "threshold identifier for deletion", "User provides threshold identifier to delete",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"delete\"", "label": "UI_ELEMENT"}, {"text": "the given threshold", "label": "DATA_OBJ"}]),
        build_step("2", "System", "The system deletes the threshold.",
                   "W", "threshold", "System removes threshold from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "deletes", "label": "ACTION_VERB"}, {"text": "the threshold", "label": "DATA_OBJ"}]),
    ]))

    save_system(system, records)
    return records


def annotate_k01719():
    system = "K01719"
    gt = load_gt(system)
    records = []

    # FP1: View vitals measurements (E=1, R=1, W=0, X=1)
    records.append(build_fp(system, "FP1", gt, [
        build_step("1", "User", "The user selects the 'View vitals measurements' tab for the given Patient.",
                   "E", "patient selection", "Triggering entry — user selects patient/tab, providing patient identifier",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "the 'View vitals measurements' tab", "label": "UI_ELEMENT"}, {"text": "the given Patient", "label": "DATA_OBJ"}]),
        build_step("2a", "System", "The system searches for the Measurements.",
                   "R", "Measurements", "System reads measurements from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searches", "label": "ACTION_VERB"}, {"text": "the Measurements", "label": "DATA_OBJ"}]),
        build_step("2b", "System", "The system displays the result list.",
                   "X", "Measurements list", "System displays measurement list to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the result list", "label": "DATA_OBJ"}]),
        build_step("3", "Actor", "The actor browses the results.",
                   None, "", "User browsing — no data movement, just viewing already displayed data",
                   [{"text": "actor", "label": "FUNC_USER"}, {"text": "browses", "label": "ACTION_VERB"}]),
    ]))

    # FP2: Add a measurement (E=2, R=1, W=1, X=3)
    records.append(build_fp(system, "FP2", gt, [
        build_step("1", "User", "The user clicks \"New measurement\".",
                   None, "", "UI action — button click, no data movement",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"New measurement\"", "label": "UI_ELEMENT"}]),
        build_step("2", "System", "The system displays a form in which it is possible to select the desired measuring instrument.",
                   "X", "measurement form", "System displays form structure to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "a form", "label": "DATA_OBJ"}]),
        build_step("3", "User", "The user selects the desired measuring instrument.",
                   "E", "measuring instrument selection", "User provides instrument selection data",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "the desired measuring instrument", "label": "DATA_OBJ"}]),
        build_step("4a", "System", "The system searches for the Vital Parameters with the specified measuring instrument ID.",
                   "R", "Vital Parameters", "System reads vital parameters from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searches", "label": "ACTION_VERB"}, {"text": "the Vital Parameters", "label": "DATA_OBJ"}]),
        build_step("4b", "System", "The system lists the Vital Parameters.",
                   "X", "Vital Parameters list", "System displays vital parameters list to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "lists", "label": "ACTION_VERB"}, {"text": "the Vital Parameters", "label": "DATA_OBJ"}]),
        build_step("5-6", "User", "The user fills in the fields and presses the \"Save\" button.",
                   "E", "measurement data", "User provides measurement data and submits",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "fills in", "label": "ACTION_VERB"}, {"text": "the fields", "label": "DATA_OBJ"}]),
        build_step("7", "System", "The system saves the changes for the new Measurement to the list of measurements for the selected vital parameter.",
                   "W", "Measurement", "System writes new measurement to persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "saves", "label": "ACTION_VERB"}, {"text": "the changes for the new Measurement", "label": "DATA_OBJ"}]),
        build_step("8-9", "System", "The system displays the message \"The measurement has been saved correctly\" and the refreshed list of measurements.",
                   "X", "confirmation + measurement list", "System displays confirmation and updated list to user (collapsed per COSMIC 3.5.11)",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the message \"The measurement has been saved correctly\"", "label": "DATA_OBJ"}]),
    ]))

    # FP3: Display graph of measurements (E=1, R=1, W=0, X=1)
    records.append(build_fp(system, "FP3", gt, [
        build_step("1", "User", "The user selects \"Display measurement graph for vital\" for the given parameter.",
                   "E", "vital parameter selection", "Triggering entry — user selects which parameter to graph",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "\"Display measurement graph for vital\"", "label": "UI_ELEMENT"}, {"text": "the given parameter", "label": "DATA_OBJ"}]),
        build_step("2a", "System", "The system searches the Measurements for the given vital parameter.",
                   "R", "Measurements", "System reads measurements from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searches", "label": "ACTION_VERB"}, {"text": "the Measurements for the given vital parameter", "label": "DATA_OBJ"}]),
        build_step("2b", "System", "The system displays the related graph.",
                   "X", "measurement graph", "System displays measurement graph to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the related graph", "label": "DATA_OBJ"}]),
    ]))

    # FP4: Add automatic measurement (E=1, R=0, W=1, X=3)
    records.append(build_fp(system, "FP4", gt, [
        build_step("1", "User", "The user uses an instrument from the kit to measure a vital sign.",
                   None, "", "Physical action outside system boundary, no data movement",
                   [{"text": "user", "label": "FUNC_USER"}]),
        build_step("2", "Sensor Device", "The Sensor Device sends the raw data to the Mobile.",
                   "E", "raw measurement data", "External device sends measurement data into the system",
                   [{"text": "Sensor Device", "label": "FUNC_USER"}, {"text": "sends", "label": "ACTION_VERB"}, {"text": "the raw data", "label": "DATA_OBJ"}]),
        build_step("3", "Mobile App", "The Mobile App requests the System to save the measurement data.",
                   None, "", "Internal request between system components, same boundary",
                   [{"text": "Mobile App", "label": "FUNC_USER"}, {"text": "requests", "label": "ACTION_VERB"}, {"text": "the measurement data", "label": "DATA_OBJ"}]),
        build_step("4", "System", "The system saves the measurement.",
                   "W", "measurement", "System writes measurement to persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "saves", "label": "ACTION_VERB"}, {"text": "the measurement", "label": "DATA_OBJ"}]),
        build_step("5a", "System", "The system sends the Mobile App the envelope data of the operation.",
                   "X", "operation envelope data", "System sends operation metadata to Mobile App",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "sends", "label": "ACTION_VERB"}, {"text": "the envelope data of the operation", "label": "DATA_OBJ"}]),
        build_step("5b", "System", "The system sends the saved measurement data to the Mobile App for display.",
                   "X", "measurement data", "System sends saved measurement data to Mobile App (distinct data group from envelope)",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "sends", "label": "ACTION_VERB"}, {"text": "the saved measurement data", "label": "DATA_OBJ"}]),
        build_step("6", "Mobile App", "The mobile App displays a message to the user to inform him (or her) that the data have been saved.",
                   "X", "confirmation message", "System displays confirmation to user via Mobile App",
                   [{"text": "mobile App", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "a message", "label": "DATA_OBJ"}]),
    ]))

    # FP5: List all alert parameters (E=2, R=2, W=0, X=2)
    records.append(build_fp(system, "FP5", gt, [
        build_step("P1", "User", "The user selects the \"Thresholds\" in a patient's medical record.",
                   None, "", "UI navigation — precondition step, selecting a tab",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "\"Thresholds\"", "label": "UI_ELEMENT"}]),
        build_step("1", "User", "The user selects the \"List all vitals\" button.",
                   "E", "list all vitals request", "Triggering entry — user requests to list vitals",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "\"List all vitals\" button", "label": "UI_ELEMENT"}]),
        build_step("2a", "System", "The system searches the list of vitals.",
                   "R", "vitals list", "System reads vitals from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searches", "label": "ACTION_VERB"}, {"text": "the list of vitals", "label": "DATA_OBJ"}]),
        build_step("2b", "System", "The system displays the list of vitals.",
                   "X", "vitals list", "System displays vitals list to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the list of vitals", "label": "DATA_OBJ"}]),
        build_step("3", "User", "The user selects the desired vital sign.",
                   "E", "vital sign selection", "User provides vital sign selection data",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "selects", "label": "ACTION_VERB"}, {"text": "the desired vital sign", "label": "DATA_OBJ"}]),
        build_step("4a", "System", "The system searches the list of alerts related to the given vital parameter.",
                   "R", "alerts list", "System reads alerts from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searches", "label": "ACTION_VERB"}, {"text": "the list of alerts related to the given vital parameter", "label": "DATA_OBJ"}]),
        build_step("4b", "System", "The system displays the list of alerts.",
                   "X", "alerts list", "System displays alerts list to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the list of alerts", "label": "DATA_OBJ"}]),
    ]))

    # FP6: Add a new alert (E=1, R=0, W=1, X=1)
    records.append(build_fp(system, "FP6", gt, [
        build_step("1", "User", "The user clicks \"Add alert\".",
                   None, "", "UI action — button click, no data movement",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"Add alert\"", "label": "UI_ELEMENT"}]),
        build_step("2", "System", "The system shows a form to add a new alert.",
                   None, "", "Blank form display — no domain data output",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "shows", "label": "ACTION_VERB"}, {"text": "a form to add a new alert", "label": "UI_ELEMENT"}]),
        build_step("3-4", "User", "The user fills in the alert fields and presses \"Save\".",
                   "E", "alert data", "User provides alert data and submits",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "fills in", "label": "ACTION_VERB"}, {"text": "the alert fields", "label": "DATA_OBJ"}]),
        build_step("5", "System", "The system saves the changes.",
                   "W", "alert", "System writes new alert to persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "saves", "label": "ACTION_VERB"}, {"text": "the changes", "label": "DATA_OBJ"}]),
        build_step("6", "System", "The system shows a confirmation message, stating \"The new alert has been saved\".",
                   "X", "confirmation message + alerts list", "System displays confirmation and returns to alert list",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "shows", "label": "ACTION_VERB"}, {"text": "a confirmation message", "label": "DATA_OBJ"}]),
    ]))

    # FP7: View alert details (E=1, R=1, W=0, X=1)
    records.append(build_fp(system, "FP7", gt, [
        build_step("1", "User", "The user clicks the \"View alter details\" button relating to the alert of his (or her) interest.",
                   "E", "alert identifier", "Triggering entry — user provides alert identifier",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"View alter details\" button", "label": "UI_ELEMENT"}, {"text": "the alert", "label": "DATA_OBJ"}]),
        build_step("2", "System", "The system searched for the alert with the given ID.",
                   "R", "alert details", "System reads alert from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "searched", "label": "ACTION_VERB"}, {"text": "the alert with the given ID", "label": "DATA_OBJ"}]),
        build_step("3", "System", "The system shows a page with the alert details.",
                   "X", "alert details", "System displays alert details to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "shows", "label": "ACTION_VERB"}, {"text": "the alert details", "label": "DATA_OBJ"}]),
    ]))

    # FP8: Edit Alert details (E=1, R=0, W=1, X=1)
    records.append(build_fp(system, "FP8", gt, [
        build_step("1", "User", "The user clicks the sub-option \"Edit alter details\".",
                   None, "", "UI navigation — selecting edit option",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"Edit alter details\"", "label": "UI_ELEMENT"}]),
        build_step("2", "System", "The system displays a page with the alert fields.",
                   None, "", "Displaying existing data already loaded — no new data movement",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "the alert fields", "label": "DATA_OBJ"}]),
        build_step("3-4", "User", "The user fills in the alert fields and presses the \"Save\" button.",
                   "E", "modified alert data", "User provides modified alert data and submits",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "fills in", "label": "ACTION_VERB"}, {"text": "the alert fields", "label": "DATA_OBJ"}]),
        build_step("5", "System", "The system saves the changes.",
                   "W", "alert", "System writes updated alert to persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "saves", "label": "ACTION_VERB"}, {"text": "the changes", "label": "DATA_OBJ"}]),
        build_step("6", "System", "The system displays a message indicating that the operation was successful.",
                   "X", "success message", "System displays success confirmation to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "a message indicating that the operation was successful", "label": "DATA_OBJ"}]),
    ]))

    # FP9: Remove alert (E=1, R=0, W=1, X=1)
    records.append(build_fp(system, "FP9", gt, [
        build_step("1", "User", "The user clicks the \"Delete alert\" button relating to the alert of his (or her) interest.",
                   "E", "alert identifier for deletion", "Triggering entry — user provides alert identifier to delete",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "clicks", "label": "ACTION_VERB"}, {"text": "\"Delete alert\" button", "label": "UI_ELEMENT"}, {"text": "the alert", "label": "DATA_OBJ"}]),
        build_step("2", "System", "The system displays a pop up window to confirm the operation.",
                   None, "", "Confirmation dialog — no domain data movement, just UI prompt",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "displays", "label": "ACTION_VERB"}, {"text": "a pop up window", "label": "UI_ELEMENT"}]),
        build_step("3", "User", "The user confirms.",
                   None, "", "User confirms deletion — UI control action, no new data",
                   [{"text": "user", "label": "FUNC_USER"}, {"text": "confirms", "label": "ACTION_VERB"}]),
        build_step("4", "System", "The system removes the given alert.",
                   "W", "alert", "System deletes alert from persistent storage",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "removes", "label": "ACTION_VERB"}, {"text": "the given alert", "label": "DATA_OBJ"}]),
        build_step("5", "System", "The system shows the confirmation message \"Alert removed\".",
                   "X", "confirmation message", "System displays deletion confirmation to user",
                   [{"text": "system", "label": "FUNC_USER"}, {"text": "shows", "label": "ACTION_VERB"}, {"text": "the confirmation message \"Alert removed\"", "label": "DATA_OBJ"}]),
    ]))

    save_system(system, records)
    return records


# ── Compact helpers ────────────────────────────────────────────────────

FU, DO, AV, UI, PS = "FUNC_USER", "DATA_OBJ", "ACTION_VERB", "UI_ELEMENT", "PERSIST_STORE"


def _e(text, label):
    return {"text": text, "label": label}


def _read_steps(system, fp_id):
    fp_path = SYSTEMS[system] / "FP_prompts" / f"{fp_id}.txt"
    return parse_uc_text(fp_path.read_text())


def _s(sid, actor, text, mvt, dg, reason, ents):
    return build_step(sid, actor, text, mvt, dg, reason, ents)


# ── K01726 ─────────────────────────────────────────────────────────────


def annotate_k01726():
    system = "K01726"
    gt = load_gt(system)
    records = []

    # FP1: View televisits list (E=1, R=1, X=1)
    records.append(build_fp(system, "FP1", gt, [
        _s("1", "User", "The user clicks the 'View televisits list' button for the given Patient.",
           "E", "patient televisit request", "Triggering entry — user requests televisit list for patient",
           [_e("user", FU), _e("clicks", AV), _e("the 'View televisits list' button", UI), _e("the given Patient", DO)]),
        _s("2a", "System", "The system searches the televisits.",
           "R", "televisits", "System reads televisits from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the televisits", DO)]),
        _s("2b", "System", "The system displays the list of all the televisit requests for the patient.",
           "X", "televisit list", "System displays televisit list to user",
           [_e("system", FU), _e("displays", AV), _e("the list of all the televisit requests", DO)]),
        _s("3", "Actor", "The actor browses the results.",
           None, "", "Browsing — no data movement",
           [_e("actor", FU), _e("browses", AV)]),
    ]))

    # FP2: Add televisit request (E=1, W=1, X=2)
    records.append(build_fp(system, "FP2", gt, [
        _s("1", "Actor", "The actor selects \"New televisit request\".",
           None, "", "UI navigation",
           [_e("actor", FU), _e("selects", AV), _e("\"New televisit request\"", UI)]),
        _s("2", "System", "The system displays a form to fill in the televisit request fields.",
           "X", "televisit form", "System displays form structure with field definitions",
           [_e("system", FU), _e("displays", AV), _e("a form to fill in the televisit request fields", DO)]),
        _s("3", "Actor", "The actor fills in the televisit request fields and presses \"Save\".",
           "E", "televisit request data", "User provides televisit request data and submits",
           [_e("actor", FU), _e("fills in", AV), _e("the televisit request fields", DO)]),
        _s("4a", "System", "The system saves the televisit request with the status \"submitted\".",
           "W", "televisit request", "System writes new televisit request to persistent storage",
           [_e("system", FU), _e("saves", AV), _e("the televisit request", DO)]),
        _s("4b", "System", "The system requests the Messanger component to notify the family doctor.",
           "X", "notification (family doctor)", "System sends notification to doctor via Messenger",
           [_e("system", FU), _e("requests", AV), _e("the Messanger component", FU)]),
    ]))

    # FP3: Remove televisit request (E=1, W=1)
    records.append(build_fp(system, "FP3", gt, [
        _s("1", "Actor", "The actor clicks the \"Delete request\" button for the given televisit request.",
           "E", "televisit request identifier", "Triggering entry — user provides request identifier to delete",
           [_e("actor", FU), _e("clicks", AV), _e("the \"Delete request\" button", UI), _e("the given televisit request", DO)]),
        _s("2", "System", "The system asks the user to confirm the operation.",
           None, "", "Confirmation dialog — no domain data movement",
           [_e("system", FU)]),
        _s("3", "Actor", "The actor confirms by clicking the \"Confirm\" button.",
           None, "", "User confirms — UI control action",
           [_e("actor", FU), _e("confirms", AV)]),
        _s("4", "System", "The system updates the request as \"withdrawn\".",
           "W", "televisit request", "System updates request status in persistent storage",
           [_e("system", FU), _e("updates", AV), _e("the request", DO)]),
    ]))

    # FP4: View televisit request details (E=1, R=1, X=1)
    records.append(build_fp(system, "FP4", gt, [
        _s("1", "Actor", "The actor clicks the \"view televisit request details\" for the televisit request of his (or her) interest.",
           "E", "televisit request identifier", "Triggering entry — user provides request identifier",
           [_e("actor", FU), _e("clicks", AV), _e("the \"view televisit request details\"", UI), _e("the televisit request", DO)]),
        _s("2a", "System", "The system searches for the televisit request.",
           "R", "televisit request details", "System reads request from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the televisit request", DO)]),
        _s("2b", "System", "The system displays a page with the televisit request details.",
           "X", "televisit request details", "System displays request details to user",
           [_e("system", FU), _e("displays", AV), _e("the televisit request details", DO)]),
        _s("3", "Actor", "The actor browses the results.",
           None, "", "Browsing — no data movement",
           [_e("actor", FU), _e("browses", AV)]),
    ]))

    # FP5: Edit televisit request details (E=1, W=1, X=1)
    records.append(build_fp(system, "FP5", gt, [
        _s("1", "Actor", "The actor clicks the sub-option \"Edit televisit request details\".",
           None, "", "UI navigation",
           [_e("actor", FU), _e("clicks", AV), _e("\"Edit televisit request details\"", UI)]),
        _s("2", "System", "The system makes editable the televisit request fields.",
           None, "", "UI state change — no new data movement",
           [_e("system", FU)]),
        _s("3", "Actor", "The actor fills in the televisit request fields of his (her) interest and presses \"Save\".",
           "E", "modified televisit request data", "User provides modified data and submits",
           [_e("actor", FU), _e("fills in", AV), _e("the televisit request fields", DO)]),
        _s("4", "System", "The system saves the changes to the televisit request.",
           "W", "televisit request", "System writes updated request to persistent storage",
           [_e("system", FU), _e("saves", AV), _e("the changes", DO)]),
        _s("4b", "System", "The system displays confirmation.",
           "X", "confirmation", "System displays update confirmation to user",
           [_e("system", FU), _e("displays", AV)]),
    ]))

    # FP6: View televisit requests received (E=1, R=1, X=1)
    records.append(build_fp(system, "FP6", gt, [
        _s("1", "Actor", "The actor clicks the \"View all my televisit requests\" to search all the televisits.",
           "E", "doctor televisit request", "Triggering entry — doctor requests to view all televisit requests",
           [_e("actor", FU), _e("clicks", AV), _e("the \"View all my televisit requests\"", UI)]),
        _s("2a", "System", "The system searches all the televisit requests for the given doctor received from patients.",
           "R", "televisit requests", "System reads televisit requests from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("all the televisit requests", DO)]),
        _s("2b", "System", "The system displays all the televisit requests.",
           "X", "televisit request list", "System displays request list to doctor",
           [_e("system", FU), _e("displays", AV), _e("all the televisit requests", DO)]),
        _s("3", "Actor", "The actor navigates the results.",
           None, "", "Browsing — no data movement",
           [_e("actor", FU), _e("navigates", AV)]),
    ]))

    # FP7: Reject televisit request (E=1, W=1, X=1)
    records.append(build_fp(system, "FP7", gt, [
        _s("1", "Actor", "The actor selects \"Reject televisit request\".",
           None, "", "UI navigation",
           [_e("actor", FU), _e("selects", AV), _e("\"Reject televisit request\"", UI)]),
        _s("2", "System", "The system displays a page containing the reason field.",
           None, "", "Blank form — no domain data output",
           [_e("system", FU), _e("displays", AV)]),
        _s("3", "Actor", "The actor fills in the reason field for the rejection and presses \"Save\".",
           "E", "rejection reason", "User provides rejection reason and submits",
           [_e("actor", FU), _e("fills in", AV), _e("the reason field", DO)]),
        _s("4a", "System", "The system updates the televisit request as \"rejected\".",
           "W", "televisit request", "System updates request status in persistent storage",
           [_e("system", FU), _e("updates", AV), _e("the televisit request", DO)]),
        _s("4b", "System", "The system requests the Messanger to process the patient's notification.",
           "X", "notification (patient)", "System sends rejection notification to patient via Messenger",
           [_e("system", FU), _e("requests", AV), _e("the Messanger", FU)]),
    ]))

    # FP8: View received televisit details (E=1, R=1, X=1)
    records.append(build_fp(system, "FP8", gt, [
        _s("1", "User", "The user clicks the \"View televisit request details\" button relating to the televisit request of his (or her).",
           "E", "televisit request identifier", "Triggering entry — user provides request identifier",
           [_e("user", FU), _e("clicks", AV), _e("the \"View televisit request details\" button", UI), _e("the televisit request", DO)]),
        _s("2a", "System", "The system searches the televisit request details.",
           "R", "televisit request details", "System reads request details from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the televisit request details", DO)]),
        _s("2b", "System", "The system displays a page with the televisit request details.",
           "X", "televisit request details", "System displays request details to user",
           [_e("system", FU), _e("displays", AV), _e("the televisit request details", DO)]),
        _s("3", "User", "The user browses the results.",
           None, "", "Browsing — no data movement",
           [_e("user", FU), _e("browses", AV)]),
    ]))

    # FP9: Accept televisit request (E=1, W=1, X=2)
    records.append(build_fp(system, "FP9", gt, [
        _s("1", "Actor", "The actor clicks \"Accept televisit request\" button relating to the televisit request of his (or her) interest.",
           "E", "televisit request identifier", "Triggering entry — user selects request to accept",
           [_e("actor", FU), _e("clicks", AV), _e("\"Accept televisit request\" button", UI), _e("the televisit request", DO)]),
        _s("2a", "System", "The system saves the televisit request as \"accepted\".",
           "W", "televisit request", "System updates request status in persistent storage",
           [_e("system", FU), _e("saves", AV), _e("the televisit request", DO)]),
        _s("2b", "System", "The system requests the Messanger component to notify the patient.",
           "X", "notification (patient)", "System sends acceptance notification to patient via Messenger",
           [_e("system", FU), _e("requests", AV), _e("the Messanger component", FU)]),
        _s("EXC-1", "System", "The system displays an error message stating that not all required fields have been provided.",
           "X", "error message", "System displays validation error to user",
           [_e("system", FU), _e("displays", AV), _e("an error message", DO)]),
    ]))

    # FP10: Edit accepted televisit details (E=1, W=1, X=2)
    records.append(build_fp(system, "FP10", gt, [
        _s("1", "User", "The user selects \"Edit accepted televist details\".",
           None, "", "UI navigation",
           [_e("user", FU), _e("selects", AV), _e("\"Edit accepted televist details\"", UI)]),
        _s("2", "System", "The system shows a form with the televisit request fields.",
           None, "", "Form display — data already loaded from precondition",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "User", "The user modifies the fields of his (her) interest and presses \"Save\".",
           "E", "modified televisit data", "User provides modified data and submits",
           [_e("user", FU), _e("modifies", AV), _e("the fields", DO)]),
        _s("4a", "System", "The system saves the changes.",
           "W", "televisit request", "System writes updated request to persistent storage",
           [_e("system", FU), _e("saves", AV), _e("the changes", DO)]),
        _s("4b", "System", "The system requests the Messanger component to notify the patient.",
           "X", "notification (patient)", "System sends notification of changes to patient",
           [_e("system", FU), _e("requests", AV), _e("the Messanger component", FU)]),
        _s("EXC-1", "System", "The system shows an error message stating that not all required fields have been provided.",
           "X", "error message", "System displays validation error to user",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP11: Accept televisit event (E=1, W=3, X=1)
    records.append(build_fp(system, "FP11", gt, [
        _s("1", "Actor", "The actor clicks the \"Accept televisit event\" button relating to the televisit event of his (or her) interest.",
           "E", "televisit event identifier", "Triggering entry — user selects event to accept",
           [_e("actor", FU), _e("clicks", AV), _e("the \"Accept televisit event\" button", UI), _e("the televisit event", DO)]),
        _s("2a", "System", "The system saves an event in the doctor's agenda.",
           "W", "doctor agenda event", "System writes event to doctor's agenda",
           [_e("system", FU), _e("saves", AV), _e("an event", DO)]),
        _s("2b", "System", "The system saves an event in the patient's agenda.",
           "W", "patient agenda event", "System writes event to patient's agenda",
           [_e("system", FU), _e("saves", AV), _e("an event", DO)]),
        _s("2c", "System", "The system updates the televisit request as \"confirmed\".",
           "W", "televisit request", "System updates request status to confirmed",
           [_e("system", FU), _e("updates", AV), _e("the televisit request", DO)]),
        _s("2d", "System", "The system sends a link to all the participants via email.",
           "X", "event link (email)", "System sends meeting link to participants",
           [_e("system", FU), _e("sends", AV), _e("a link", DO)]),
    ]))

    # FP12: Reject televisit event (E=1, W=1, X=1)
    records.append(build_fp(system, "FP12", gt, [
        _s("1", "Actor", "The actor clicks \"Reject televisit event\" relating to the televisit event of his (or her) interest.",
           "E", "televisit event identifier", "Triggering entry — user selects event to reject",
           [_e("actor", FU), _e("clicks", AV), _e("the televisit event", DO)]),
        _s("2a", "System", "The system saves the televisit request as \"rejected\".",
           "W", "televisit request", "System updates request status to rejected",
           [_e("system", FU), _e("saves", AV), _e("the televisit request", DO)]),
        _s("2b", "System", "The system requests the Messanger to process the doctor's notification.",
           "X", "notification (doctor)", "System sends rejection notification to doctor via Messenger",
           [_e("system", FU), _e("requests", AV), _e("the Messanger", FU)]),
    ]))

    # FP13: View televisit event details (E=2, R=2, X=2)
    records.append(build_fp(system, "FP13", gt, [
        _s("1", "Actor", "The actor clicks the \"View televisit events\" button to view all his (her) televisit events for today on the agenda.",
           "E", "today's events request", "Triggering entry — user requests today's events",
           [_e("actor", FU), _e("clicks", AV), _e("the \"View televisit events\" button", UI)]),
        _s("2a", "System", "The system searches all the televisit events.",
           "R", "televisit events", "System reads today's events from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("all the televisit events", DO)]),
        _s("2b", "System", "The system displays today's events.",
           "X", "televisit events list", "System displays today's event list to user",
           [_e("system", FU), _e("displays", AV), _e("today's events", DO)]),
        _s("3", "Actor", "The actor clicks the \"View televisit event details\" button relating to the televisit event of his (or her) interest.",
           "E", "televisit event identifier", "User provides event identifier to view details",
           [_e("actor", FU), _e("clicks", AV), _e("the \"View televisit event details\" button", UI), _e("the televisit event", DO)]),
        _s("4a", "System", "The system searches the televisit event details.",
           "R", "televisit event details", "System reads event details from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the televisit event details", DO)]),
        _s("4b", "System", "The system displays the televisit event details.",
           "X", "televisit event details", "System displays event details to user",
           [_e("system", FU), _e("displays", AV), _e("the televisit event details", DO)]),
        _s("5", "Actor", "The actor browses the results.",
           None, "", "Browsing — no data movement",
           [_e("actor", FU), _e("browses", AV)]),
    ]))

    # FP14: List teleconsultations (E=1, R=1, X=1)
    records.append(build_fp(system, "FP14", gt, [
        _s("1", "Actor", "The actor clicks the \"List teleconsultations\" button to view all the teleconsultations.",
           "E", "teleconsultation list request", "Triggering entry — user requests teleconsultation list",
           [_e("actor", FU), _e("clicks", AV), _e("the \"List teleconsultations\" button", UI)]),
        _s("2a", "System", "The system searches the teleconsultations.",
           "R", "teleconsultations", "System reads teleconsultations from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the teleconsultations", DO)]),
        _s("2b", "System", "The system displays the teleconsultations.",
           "X", "teleconsultation list", "System displays teleconsultation list to user",
           [_e("system", FU), _e("displays", AV), _e("the teleconsultations", DO)]),
        _s("3", "Actor", "The actor browses the results.",
           None, "", "Browsing — no data movement",
           [_e("actor", FU), _e("browses", AV)]),
    ]))

    # FP15: Add teleconsultation (E=1, W=2, X=2)
    records.append(build_fp(system, "FP15", gt, [
        _s("1", "Actor", "The actor selects \"New teleconsultation\".",
           None, "", "UI navigation",
           [_e("actor", FU), _e("selects", AV), _e("\"New teleconsultation\"", UI)]),
        _s("2", "System", "The system shows a page with the teleconsultation fields.",
           "X", "teleconsultation form", "System displays form structure with field definitions",
           [_e("system", FU), _e("shows", AV), _e("the teleconsultation fields", DO)]),
        _s("3", "Actor", "The actor fills in the teleconsultation fields and presses \"Save\".",
           "E", "teleconsultation data", "User provides teleconsultation data and submits",
           [_e("actor", FU), _e("fills in", AV), _e("the teleconsultation fields", DO)]),
        _s("4a", "System", "The system saves the teleconsultation as \"requested\".",
           "W", "teleconsultation", "System writes teleconsultation to persistent storage",
           [_e("system", FU), _e("saves", AV), _e("the teleconsultation", DO)]),
        _s("4b", "System", "The system inserts the teleconsultation on the actor's agenda.",
           "W", "actor agenda event", "System writes event to actor's agenda",
           [_e("system", FU), _e("inserts", AV), _e("the teleconsultation", DO)]),
        _s("4c", "System", "The system requests the Messanger component to notify all the participants.",
           "X", "notification (participants)", "System sends notification to all participants",
           [_e("system", FU), _e("requests", AV), _e("the Messanger component", FU)]),
    ]))

    # FP16: View teleconsultation details (E=1, R=1, X=1)
    records.append(build_fp(system, "FP16", gt, [
        _s("1", "User", "The user clicks the \"View teleconsultation details\" button relating to the teleconsultation of his (or her) interest.",
           "E", "teleconsultation identifier", "Triggering entry — user provides teleconsultation identifier",
           [_e("user", FU), _e("clicks", AV), _e("the \"View teleconsultation details\" button", UI), _e("the teleconsultation", DO)]),
        _s("2a", "System", "The system searches the teleconsultation details.",
           "R", "teleconsultation details", "System reads teleconsultation from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the teleconsultation details", DO)]),
        _s("2b", "System", "The system displays the teleconsultation details.",
           "X", "teleconsultation details", "System displays details to user",
           [_e("system", FU), _e("displays", AV), _e("the teleconsultation details", DO)]),
        _s("3", "Actor", "The actor browses the results.",
           None, "", "Browsing — no data movement",
           [_e("actor", FU), _e("browses", AV)]),
    ]))

    # FP17: Edit teleconsultation details (E=1, W=1, X=2)
    records.append(build_fp(system, "FP17", gt, [
        _s("1", "User", "The user selects \"Edit teleconsultation details\".",
           None, "", "UI navigation",
           [_e("user", FU), _e("selects", AV), _e("\"Edit teleconsultation details\"", UI)]),
        _s("2", "System", "The system shows a form with the teleconsultation fields.",
           None, "", "Form display — data already loaded",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Actor", "The actor changes in the teleconsultation fields of his (her) interest and presses \"Save\".",
           "E", "modified teleconsultation data", "User provides modified data and submits",
           [_e("actor", FU), _e("changes", AV), _e("the teleconsultation fields", DO)]),
        _s("4", "System", "The system saves the changes to the teleconsultation.",
           "W", "teleconsultation", "System writes updated teleconsultation to persistent storage",
           [_e("system", FU), _e("saves", AV), _e("the changes", DO)]),
        _s("ALT-1", "System", "If the date-time has changed, the system requests the Messanger to process the notification for all the participants.",
           "X", "notification (participants)", "System sends update notification via Messenger",
           [_e("system", FU), _e("requests", AV), _e("the Messanger", FU)]),
        _s("EXC-1", "System", "The system shows an error message stating that only some of the required fields have been provided.",
           "X", "error message", "System displays validation error to user",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP18: Remove teleconsultation (E=1, W=1)
    records.append(build_fp(system, "FP18", gt, [
        _s("1", "Actor", "The actor clicks the \"Remove teleconsultation\" button relating to the teleconsultation of his (or her) interest.",
           "E", "teleconsultation identifier", "Triggering entry — user provides identifier to delete",
           [_e("actor", FU), _e("clicks", AV), _e("the \"Remove teleconsultation\" button", UI), _e("the teleconsultation", DO)]),
        _s("2", "System", "The system asks the user to confirm the operation.",
           None, "", "Confirmation dialog",
           [_e("system", FU)]),
        _s("3", "Actor", "The actor clicks the \"Confirm\" button.",
           None, "", "User confirms",
           [_e("actor", FU), _e("clicks", AV)]),
        _s("4", "System", "The system removes the teleconsultation.",
           "W", "teleconsultation", "System removes teleconsultation from persistent storage",
           [_e("system", FU), _e("removes", AV), _e("the teleconsultation", DO)]),
    ]))

    # FP19: Accept teleconsultation invitation (E=1, W=2)
    records.append(build_fp(system, "FP19", gt, [
        _s("1", "Actor", "The actor clicks the \"Accept teleconsultation invitation\" relating to the teleconsultation of his (her) interest.",
           "E", "teleconsultation identifier", "Triggering entry — user selects invitation to accept",
           [_e("actor", FU), _e("clicks", AV), _e("the \"Accept teleconsultation invitation\"", UI), _e("the teleconsultation", DO)]),
        _s("2a", "System", "The system updates the teleconsultation invitation as \"accepted\".",
           "W", "teleconsultation invitation", "System updates invitation status",
           [_e("system", FU), _e("updates", AV), _e("the teleconsultation invitation", DO)]),
        _s("2b", "System", "The system saves the teleconsultation event on the actor's agenda.",
           "W", "actor agenda event", "System writes event to actor's agenda",
           [_e("system", FU), _e("saves", AV), _e("the teleconsultation event", DO)]),
    ]))

    # FP20: Reject teleconsultation invitation (E=1, W=1, X=1)
    records.append(build_fp(system, "FP20", gt, [
        _s("1", "Actor", "The actor clicks the \"Reject teleconsultation invitation\" relating to the teleconsultation invitation of his (her) interest.",
           "E", "teleconsultation identifier", "Triggering entry — user selects invitation to reject",
           [_e("actor", FU), _e("clicks", AV), _e("the teleconsultation invitation", DO)]),
        _s("2", "System", "The system asks the user to confirm the operation.",
           None, "", "Confirmation dialog",
           [_e("system", FU)]),
        _s("3", "Actor", "The actor clicks the \"Confirm\" button.",
           None, "", "User confirms",
           [_e("actor", FU), _e("clicks", AV)]),
        _s("4a", "System", "The system removes the teleconsultation.",
           "W", "teleconsultation", "System removes teleconsultation from persistent storage",
           [_e("system", FU), _e("removes", AV), _e("the teleconsultation", DO)]),
        _s("4b", "System", "The system requests the Messanger component to notify the sender.",
           "X", "notification (sender)", "System sends rejection notification via Messenger",
           [_e("system", FU), _e("requests", AV), _e("the Messanger component", FU)]),
    ]))

    # FP21: View teleconsultation event details (E=2, R=2, X=2)
    records.append(build_fp(system, "FP21", gt, [
        _s("1", "Actor", "The actor clicks \"View all teleconsultations\" to search all today's teleconsultations.",
           "E", "today's teleconsultations request", "Triggering entry — user requests today's teleconsultations",
           [_e("actor", FU), _e("clicks", AV)]),
        _s("2a", "System", "The system searches all the teleconsultation for the given date.",
           "R", "teleconsultations", "System reads today's teleconsultations from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("all the teleconsultation", DO)]),
        _s("2b", "System", "The system displays all the teleconsultation for the given date.",
           "X", "teleconsultation list", "System displays today's teleconsultation list",
           [_e("system", FU), _e("displays", AV), _e("all the teleconsultation", DO)]),
        _s("3", "Actor", "The actor clicks \"View teleconsultation event details\" relating to the teleconsultation event of his (or her) interest.",
           "E", "teleconsultation event identifier", "User provides event identifier",
           [_e("actor", FU), _e("clicks", AV), _e("the teleconsultation event", DO)]),
        _s("4a", "System", "The system searches the teleconsultation event details.",
           "R", "teleconsultation event details", "System reads event details from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the teleconsultation event details", DO)]),
        _s("4b", "System", "The system displays the teleconsultation event details.",
           "X", "teleconsultation event details", "System displays event details to user",
           [_e("system", FU), _e("displays", AV), _e("the teleconsultation event details", DO)]),
        _s("5", "Actor", "The actor browses the results.",
           None, "", "Browsing — no data movement",
           [_e("actor", FU), _e("browses", AV)]),
    ]))

    # FP22: Start televisit/teleconsultation (E=1, X=1)
    records.append(build_fp(system, "FP22", gt, [
        _s("1", "Actor", "The actor clicks the \"Start televisit/teleconsultation\" button relating the televisit/teleconsultation of his (or her) interest.",
           "E", "televisit/teleconsultation identifier", "Triggering entry — user provides event identifier to start",
           [_e("actor", FU), _e("clicks", AV), _e("the \"Start televisit/teleconsultation\" button", UI)]),
        _s("2", "System", "The system sends the request to the XXX system to start the meeting.",
           "X", "meeting start request", "System sends start request to external meeting system",
           [_e("system", FU), _e("sends", AV), _e("the request", DO)]),
    ]))

    # FP23: Close televisit/teleconsultation (E=1, W=1, X=2)
    records.append(build_fp(system, "FP23", gt, [
        _s("1", "Actor", "The actor wants to close the meeting.",
           None, "", "User intent — no data movement yet",
           [_e("actor", FU)]),
        _s("2", "System", "The system shows a form with the outcome field for the televist/teleconsultation.",
           None, "", "Form display — showing outcome options",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Actor", "The actor selects the outcome and presses \"Save\".",
           "E", "event outcome", "User provides outcome selection and submits",
           [_e("actor", FU), _e("selects", AV), _e("the outcome", DO)]),
        _s("4a", "System", "The system saves the \"executed\" event.",
           "W", "televisit/teleconsultation event", "System saves event with executed status",
           [_e("system", FU), _e("saves", AV), _e("the \"executed\" event", DO)]),
        _s("4b", "System", "The system displays the refreshed agenda.",
           "X", "refreshed agenda", "System displays updated agenda to user",
           [_e("system", FU), _e("displays", AV), _e("the refreshed agenda", DO)]),
        _s("EXC-1", "System", "The system displays an error message.",
           "X", "error message", "System displays error to user",
           [_e("system", FU), _e("displays", AV), _e("an error message", DO)]),
    ]))

    save_system(system, records)
    return records


# ── U-CURE ─────────────────────────────────────────────────────────────


def annotate_ucure():
    system = "U-CURE"
    gt = load_gt(system)
    records = []

    # FP1: Loading clinical data (E=4, W=1, X=5)
    records.append(build_fp(system, "FP1", gt, [
        _s("1", "ML engineer", "The Machine learning engineer clicks the suboption \"Load dataset\".",
           None, "", "UI navigation",
           [_e("Machine learning engineer", FU), _e("clicks", AV), _e("the suboption \"Load dataset\"", UI)]),
        _s("2", "System", "The system displays a form containing the dataset fields.",
           "X", "dataset form", "System displays form structure with dataset fields",
           [_e("system", FU), _e("displays", AV), _e("a form containing the dataset fields", DO)]),
        _s("3", "ML engineer", "The Machine learning engineer fills in the dataset name and other details and presses \"Save\".",
           "E", "dataset details", "User provides dataset configuration and submits",
           [_e("Machine learning engineer", FU), _e("fills in", AV), _e("the dataset name and other details", DO)]),
        _s("4", "System", "The system saves the new project data.",
           "W", "project data", "System writes project data to persistent storage",
           [_e("system", FU), _e("saves", AV), _e("the new project data", DO)]),
        _s("5", "System", "The system requests the ML Engine to process the dataset.",
           "X", "dataset (to ML Engine)", "System sends dataset to ML Engine for processing",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU), _e("the dataset", DO)]),
        _s("6", "ML engine", "The ML engine replies with the EDA.",
           "E", "EDA results", "ML Engine returns exploratory data analysis results",
           [_e("ML engine", FU), _e("replies", AV), _e("the EDA", DO)]),
        _s("7", "System", "The system displays the EDA details (categorical, continuous, etc.).",
           "X", "EDA details", "System displays EDA results to user",
           [_e("system", FU), _e("displays", AV), _e("the EDA details", DO)]),
        _s("8", "ML engineer", "The Machine learning engineer fills in the variable of his (or her) interest and presses 'Transform'.",
           "E", "variable selection + transform request", "User provides variables and requests transformation",
           [_e("Machine learning engineer", FU), _e("fills in", AV), _e("the variable", DO)]),
        _s("9", "System", "The system requests the ML Engine to process the transformation.",
           "X", "transformation request (to ML Engine)", "System sends transformation request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU)]),
        _s("10", "ML engine", "The ML engine replies with the transformation.",
           "E", "transformation results", "ML Engine returns transformation results",
           [_e("ML engine", FU), _e("replies", AV), _e("the transformation", DO)]),
        _s("11", "System", "The system displays the transformation details.",
           "X", "transformation details", "System displays transformation results to user",
           [_e("system", FU), _e("displays", AV), _e("the transformation details", DO)]),
    ]))

    # FP2: List model projects (E=1, R=1, X=2)
    records.append(build_fp(system, "FP2", gt, [
        _s("1", "ML engineer", "The Machine learning engineer presses the \"List all projects\" link to list all projects.",
           "E", "list projects request", "Triggering entry — user requests project list",
           [_e("Machine learning engineer", FU), _e("presses", AV), _e("the \"List all projects\" link", UI)]),
        _s("2", "System", "The system searches for all projects.",
           "R", "projects", "System reads projects from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("all projects", DO)]),
        _s("3", "System", "The system displays the project list.",
           "X", "project list", "System displays project list to user",
           [_e("system", FU), _e("displays", AV), _e("the project list", DO)]),
        _s("4", "ML engineer", "The Machine learning engineer navigates the results.",
           None, "", "Browsing — no data movement",
           [_e("Machine learning engineer", FU), _e("navigates", AV)]),
        _s("EXC-3", "System", "The system displays an error message stating that no data have been retrieved.",
           "X", "error message", "System displays error to user",
           [_e("system", FU), _e("displays", AV), _e("an error message", DO)]),
    ]))

    # FP3: Open project (E=1, R=1, X=2)
    records.append(build_fp(system, "FP3", gt, [
        _s("1", "ML engineer", "The Machine learning engineer clicks \"Open\" relating the project of his (or) interest.",
           "E", "project identifier", "Triggering entry — user selects project to open",
           [_e("Machine learning engineer", FU), _e("clicks", AV), _e("\"Open\"", UI), _e("the project", DO)]),
        _s("2", "System", "The system searches the project details.",
           "R", "project details", "System reads project details from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the project details", DO)]),
        _s("3", "System", "The system displays the project details.",
           "X", "project details", "System displays project details to user",
           [_e("system", FU), _e("displays", AV), _e("the project details", DO)]),
        _s("4", "ML engineer", "The Machine learning views the project details.",
           None, "", "Viewing — no data movement",
           [_e("Machine learning", FU)]),
        _s("EXC-3", "System", "The system displays an error message.",
           "X", "error message", "System displays error to user",
           [_e("system", FU), _e("displays", AV), _e("an error message", DO)]),
    ]))

    # FP4: Univariate and bivariate analysis (E=4, X=4)
    records.append(build_fp(system, "FP4", gt, [
        _s("1", "ML engineer", "The Machine learning engineer clicks \"Analyze\" relating to the variables to analyze.",
           "E", "variables to analyze", "User provides variable selection for analysis",
           [_e("Machine learning engineer", FU), _e("clicks", AV), _e("\"Analyze\"", UI), _e("the variables", DO)]),
        _s("2", "System", "The system requests the ML Engine to process the analysis.",
           "X", "analysis request (to ML Engine)", "System sends analysis request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU)]),
        _s("3", "ML engine", "The ML engine replies with the analysis details.",
           "E", "analysis results", "ML Engine returns analysis results",
           [_e("ML engine", FU), _e("replies", AV), _e("the analysis details", DO)]),
        _s("4", "System", "The system displays the analysis details.",
           "X", "analysis details", "System displays analysis to user",
           [_e("system", FU), _e("displays", AV), _e("the analysis details", DO)]),
        _s("5-6", "ML engineer", "The Machine learning engineer fills in the variables of his (or her) interest and presses 'T-test'.",
           "E", "t-test variable selection", "User selects variables and requests t-test",
           [_e("Machine learning engineer", FU), _e("fills in", AV), _e("the variables", DO)]),
        _s("6b", "System", "The system requests the ML Engine component to process the t-test.",
           "X", "t-test request (to ML Engine)", "System sends t-test request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine component", FU)]),
        _s("7", "ML engine", "The ML engine replies with the t-test details.",
           "E", "t-test results", "ML Engine returns t-test results",
           [_e("ML engine", FU), _e("replies", AV), _e("the t-test details", DO)]),
        _s("8", "System", "The system displays the t-test details.",
           "X", "t-test details", "System displays t-test results to user",
           [_e("system", FU), _e("displays", AV), _e("the t-test details", DO)]),
    ]))

    # FP5: Comparison of categorical features (E=2, X=2)
    records.append(build_fp(system, "FP5", gt, [
        _s("1", "ML engineer", "The Machine learning engineer clicks \"Compare categorical features\" relating the categorical variables.",
           "E", "categorical variables", "User provides categorical variables for comparison",
           [_e("Machine learning engineer", FU), _e("clicks", AV), _e("\"Compare categorical features\"", UI), _e("the categorical variables", DO)]),
        _s("2", "System", "The system requests the ML Engine to process the features comparison.",
           "X", "comparison request (to ML Engine)", "System sends comparison request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU)]),
        _s("3", "ML engine", "The ML engine replies with the features comparison.",
           "E", "comparison results", "ML Engine returns comparison results",
           [_e("ML engine", FU), _e("replies", AV), _e("the features comparison", DO)]),
        _s("4", "System", "The system displays the features comparison details.",
           "X", "comparison details", "System displays comparison results to user",
           [_e("system", FU), _e("displays", AV), _e("the features comparison details", DO)]),
    ]))

    # FP6: Bivariate analysis of continuous features (E=2, X=2)
    records.append(build_fp(system, "FP6", gt, [
        _s("1", "ML engineer", "The Machine learning engineer clicks \"Continuous feature analysis\" relating the continuous features of his interest.",
           "E", "continuous features", "User provides continuous features for analysis",
           [_e("Machine learning engineer", FU), _e("clicks", AV), _e("the continuous features", DO)]),
        _s("2", "System", "The system sends the features to analyze.",
           "X", "features (to ML Engine)", "System sends features to ML Engine",
           [_e("system", FU), _e("sends", AV), _e("the features", DO)]),
        _s("3", "ML engine", "The ML engine replies with the analysis.",
           "E", "analysis results", "ML Engine returns analysis results",
           [_e("ML engine", FU), _e("replies", AV), _e("the analysis", DO)]),
        _s("4", "System", "The system displays the analysis.",
           "X", "analysis results", "System displays analysis to user",
           [_e("system", FU), _e("displays", AV), _e("the analysis", DO)]),
    ]))

    # FP7: Multivariate analysis (E=2, X=2)
    records.append(build_fp(system, "FP7", gt, [
        _s("1", "ML engineer", "The Machine learning engineer presses \"Multivariate analysis\".",
           None, "", "UI navigation",
           [_e("Machine learning engineer", FU), _e("presses", AV), _e("\"Multivariate analysis\"", UI)]),
        _s("2", "System", "The system displays a form to select the features.",
           None, "", "Blank form — no domain data output",
           [_e("system", FU), _e("displays", AV)]),
        _s("3", "ML engineer", "The Machine learning engineer selects the features of his (or her) interest and presses \"Execute\".",
           "E", "feature selection", "User provides feature selection and submits",
           [_e("Machine learning engineer", FU), _e("selects", AV), _e("the features", DO)]),
        _s("4", "System", "The system requests the ML Engine to process the Multivariate analysis of the selected features.",
           "X", "multivariate analysis request (to ML Engine)", "System sends analysis request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU)]),
        _s("5", "ML engine", "The ML engine replies with the multivariate analysis.",
           "E", "multivariate analysis results", "ML Engine returns analysis results",
           [_e("ML engine", FU), _e("replies", AV), _e("the multivariate analysis", DO)]),
        _s("6", "System", "The system displays the analysis.",
           "X", "analysis results", "System displays analysis to user",
           [_e("system", FU), _e("displays", AV), _e("the analysis", DO)]),
    ]))

    # FP8: SVC model training (E=2, W=1, X=2)
    records.append(build_fp(system, "FP8", gt, [
        _s("1", "ML engineer", "The Machine learning engineer clicks \"Train\" link.",
           None, "", "UI navigation",
           [_e("Machine learning engineer", FU), _e("clicks", AV), _e("\"Train\" link", UI)]),
        _s("2", "System", "The system shows a form containing the following fields: training parameters, dataset.",
           None, "", "Blank form — no domain data output",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "ML engineer", "The Machine learning engineer fills in the model parameters and presses \"Execute\".",
           "E", "training parameters", "User provides model training parameters and submits",
           [_e("Machine learning engineer", FU), _e("fills in", AV), _e("the model parameters", DO)]),
        _s("4", "System", "The system requests the ML Engine to train the model.",
           "X", "training request (to ML Engine)", "System sends training request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU)]),
        _s("5", "ML engine", "The ML engine replies with the training results.",
           "E", "training results", "ML Engine returns training results",
           [_e("ML engine", FU), _e("replies", AV), _e("the training results", DO)]),
        _s("6", "System", "The system updates the new model status.",
           "W", "model status", "System writes updated model status to persistent storage",
           [_e("system", FU), _e("updates", AV), _e("the new model status", DO)]),
        _s("7", "System", "The system displays the training details (precision, recall and F1-score).",
           "X", "training metrics", "System displays training metrics to user",
           [_e("system", FU), _e("displays", AV), _e("the training details", DO)]),
    ]))

    # FP9: Model test (E=2, X=2)
    records.append(build_fp(system, "FP9", gt, [
        _s("1", "ML engineer", "The Machine learning engineer clicks the \"Test\" link.",
           None, "", "UI navigation",
           [_e("Machine learning engineer", FU), _e("clicks", AV), _e("the \"Test\" link", UI)]),
        _s("2", "System", "The system displays a form containing the training parameters and dataset fields.",
           None, "", "Form display with pre-populated parameters — no new data movement",
           [_e("system", FU), _e("displays", AV)]),
        _s("3", "ML engineer", "The Machine learning engineer fills in the parameters for the model, and presses \"Execute\".",
           "E", "test parameters", "User provides test parameters and submits",
           [_e("Machine learning engineer", FU), _e("fills in", AV), _e("the parameters", DO)]),
        _s("4", "System", "The system requests the ML Engine to test the SVC model.",
           "X", "test request (to ML Engine)", "System sends test request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU)]),
        _s("5", "ML engine", "The ML engine replies with the test results for the model.",
           "E", "test results", "ML Engine returns test results",
           [_e("ML engine", FU), _e("replies", AV), _e("the test results", DO)]),
        _s("6", "System", "The system displays the test details (precision, recall and F1-score) for the model.",
           "X", "test metrics", "System displays test metrics to user",
           [_e("system", FU), _e("displays", AV), _e("the test details", DO)]),
    ]))

    # FP10: Patient classification (E=2, R=1, X=3)
    records.append(build_fp(system, "FP10", gt, [
        _s("1", "Doctor", "The doctor clicks \"Diagnosis\" relating to the patient of his (or her) interest.",
           "E", "patient identifier", "Triggering entry — doctor selects patient for classification",
           [_e("doctor", FU), _e("clicks", AV), _e("\"Diagnosis\"", UI), _e("the patient", DO)]),
        _s("2", "System", "The system displays the patient data.",
           "X", "patient data", "System displays patient data context to doctor (implied step)",
           [_e("system", FU), _e("displays", AV), _e("the patient data", DO)]),
        _s("3", "System", "The system searches for the SVC model parameters.",
           "R", "SVC model parameters", "System reads model parameters from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the SVC model parameters", DO)]),
        _s("4", "System", "The system requests the ML Engine to classify the patient.",
           "X", "classification request (to ML Engine)", "System sends classification request to ML Engine",
           [_e("system", FU), _e("requests", AV), _e("the ML Engine", FU)]),
        _s("5", "ML engine", "The ML engine replies with the patient classification.",
           "E", "classification result", "ML Engine returns classification result",
           [_e("ML engine", FU), _e("replies", AV), _e("the patient classification", DO)]),
        _s("6", "System", "The system displays the patient classification.",
           "X", "patient classification", "System displays classification result to doctor",
           [_e("system", FU), _e("displays", AV), _e("the patient classification", DO)]),
    ]))

    save_system(system, records)
    return records


# ── ALBERGATE ──────────────────────────────────────────────────────────
# 62 FPs — hotel management system (MIS domain).
# ALBERGATE GT only has total CFP (no per-movement breakdown).
# Movement labels derived from expert COSMIC analysis of each UC.


def _alb_search(system, fp_id, gt, actor, entity_name, steps_text):
    """Standard search pattern: X(form)+E(criteria)+R(search)+X(display) = 4 CFP."""
    return build_fp(system, fp_id, gt, [
        _s("1", actor, steps_text[0], None, "", "UI navigation",
           [_e(actor.lower(), FU), _e("selects", AV)]),
        _s("2", "System", steps_text[1], "X", f"{entity_name} search form", "Form display with filter fields",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", actor, steps_text[2], "E", f"{entity_name} search criteria", "User provides search criteria",
           [_e(actor.lower(), FU), _e("fills in", AV)]),
        _s("4", "System", steps_text[3], "R", entity_name, f"System reads {entity_name} from persistent storage",
           [_e("system", FU), _e("searches", AV), _e(entity_name, DO)]),
        _s("5", "System", steps_text[4], "X", f"{entity_name} list", f"System displays {entity_name} list",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", actor, steps_text[5] if len(steps_text) > 5 else "The user browses the results.",
           None, "", "Browsing — no data movement",
           [_e(actor.lower(), FU), _e("browses", AV)]),
    ])


def _alb_view(system, fp_id, gt, actor, entity_name, main_text, exc_text, extra_reads=0, extra_displays=0):
    """Standard view pattern: E(click)+R(search)+X(display)+X(error) = 3-4+ CFP."""
    def _text_ents(text, base_ents):
        """Only include entities that actually appear in the text."""
        return [e for e in base_ents if e["text"].lower() in text.lower()]

    trigger_ents = _text_ents(main_text[0],
        [_e(actor.lower(), FU), _e("hotel manager", FU), _e("clicks", AV),
         _e("presses", AV), _e(entity_name, DO)])
    steps = [
        _s("1", actor, main_text[0], "E", f"{entity_name} identifier", "Triggering entry — user provides identifier",
           trigger_ents if trigger_ents else [_e(actor.lower(), FU)]),
    ]
    r_count = 1 + extra_reads
    x_count = 1 + extra_displays
    for i in range(r_count):
        sid = f"2{'abcdef'[i]}" if r_count > 1 else "2"
        r_text = main_text[1] if i == 0 else f"The system searches for additional {entity_name} data."
        r_verb = "searches" if "searches" in r_text.lower() else "shows"
        steps.append(_s(sid, "System", r_text,
                       "R", entity_name, f"System reads {entity_name} from persistent storage",
                       [_e("system", FU), _e(r_verb, AV)]))
    for i in range(x_count):
        sid = f"3{'abcdef'[i]}" if x_count > 1 else "3"
        x_text = main_text[2] if i == 0 else f"The system displays additional {entity_name} data."
        x_verb = "shows" if "shows" in x_text.lower() else "displays"
        steps.append(_s(sid, "System", x_text,
                       "X", entity_name, f"System displays {entity_name} to user",
                       [_e("system", FU), _e(x_verb, AV)]))
    if exc_text:
        exc_verb = "shows" if "shows" in exc_text.lower() else "displays"
        steps.append(_s("EXC-1", "System", exc_text, "X", "error message", "System displays error to user",
                       [_e("system", FU), _e(exc_verb, AV), _e("an error message", DO)]))
    return build_fp(system, fp_id, gt, steps)


def _alb_edit(system, fp_id, gt, actor, entity_name, has_error=False, extra_w=0, has_read=False):
    """Standard edit pattern: E(data)+W(save)+X(confirm) = 3 CFP."""
    steps = [
        _s("1", actor, f"The {actor.lower()} clicks the \"Edit\" option.", None, "", "UI navigation",
           [_e(actor.lower(), FU), _e("clicks", AV)]),
        _s("2", "System", f"The system shows a form with the {entity_name} fields.", None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
    ]
    if has_read:
        steps.insert(1, _s("1b", "System", f"The system checks the {entity_name} status.",
                          "R", entity_name, f"System reads {entity_name} status",
                          [_e("system", FU), _e("checks", AV)]))
    steps.append(_s("3", actor, f"The {actor.lower()} modifies the fields and presses \"Save\".",
                    "E", f"modified {entity_name} data", "User provides modified data and submits",
                    [_e(actor.lower(), FU), _e("modifies", AV)]))
    steps.append(_s("4", "System", f"The system validates and records the new {entity_name} details.",
                    "W", entity_name, f"System writes {entity_name} to persistent storage",
                    [_e("system", FU), _e("records", AV)]))
    for i in range(extra_w):
        steps.append(_s(f"4{'bcdef'[i]}", "System", f"The system updates additional {entity_name} data.",
                       "W", entity_name, f"System writes additional data",
                       [_e("system", FU), _e("updates", AV)]))
    steps.append(_s("5", "System", "The system shows a confirmation message.",
                    "X", "confirmation", "System displays confirmation to user",
                    [_e("system", FU), _e("shows", AV)]))
    if has_error:
        steps.append(_s("EXC-1", "System", "The system shows an error message.",
                       "X", "error message", "System displays error to user",
                       [_e("system", FU), _e("shows", AV), _e("an error message", DO)]))
    return build_fp(system, fp_id, gt, steps)


def _alb_insert(system, fp_id, gt, actor, entity_name):
    """Standard insert pattern: E(data)+W(save)+X(confirm) = 3 CFP."""
    return build_fp(system, fp_id, gt, [
        _s("1", actor, f"The {actor.lower()} clicks the \"New\" button.", None, "", "UI navigation",
           [_e(actor.lower(), FU), _e("clicks", AV)]),
        _s("2", "System", f"The system shows a form with the {entity_name} fields.", None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3-4", actor, f"The {actor.lower()} fills in the fields and submits.",
           "E", f"{entity_name} data", "User provides data and submits",
           [_e(actor.lower(), FU), _e("fills in", AV)]),
        _s("5", "System", f"The system validates and records the new {entity_name}.",
           "W", entity_name, f"System writes {entity_name} to persistent storage",
           [_e("system", FU), _e("records", AV)]),
        _s("6", "System", "The system shows a confirmation message.",
           "X", "confirmation", "System displays confirmation to user",
           [_e("system", FU), _e("shows", AV)]),
    ])


def _alb_delete(system, fp_id, gt, actor, entity_name, extra_w=0, has_confirm_x=True):
    """Standard delete pattern: E(id)+W(remove)+X(confirm) = 3 CFP."""
    steps = [
        _s("1", actor, f"The {actor.lower()} clicks the \"Delete\" button for the {entity_name}.",
           "E", f"{entity_name} identifier", "Triggering entry — user provides identifier to delete",
           [_e(actor.lower(), FU), _e("clicks", AV), _e(entity_name, DO)]),
        _s("2", "System", "The system asks the user to confirm.", None, "", "Confirmation dialog",
           [_e("system", FU)]),
        _s("3", actor, f"The {actor.lower()} confirms.", None, "", "User confirms",
           [_e(actor.lower(), FU), _e("confirms", AV)]),
        _s("4", "System", f"The system removes the {entity_name}.",
           "W", entity_name, f"System removes {entity_name} from persistent storage",
           [_e("system", FU), _e("removes", AV), _e(entity_name, DO)]),
    ]
    for i in range(extra_w):
        steps.append(_s(f"4{'bcdef'[i]}", "System", f"The system updates related data.",
                       "W", f"related {entity_name} data", "System updates additional data",
                       [_e("system", FU), _e("updates", AV)]))
    if has_confirm_x:
        steps.append(_s("5", "System", "The system shows a confirmation message.",
                       "X", "confirmation", "System displays confirmation to user",
                       [_e("system", FU), _e("shows", AV)]))
    return build_fp(system, fp_id, gt, steps)


def annotate_albergate():
    system = "ALBERGATE"
    gt = load_gt(system)
    recs = []

    ps = lambda fid: _read_steps(system, fid)

    # --- Room management (UC-GES-01.x) ---

    # FP1: Search rooms (4) = X(form)+E+R+X
    st = ps("FP1")
    recs.append(build_fp(system, "FP1", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], None, "", "UI navigation",
           [_e("hotel manager", FU), _e("selects", AV), _e("\"Rooms\"", UI)]),
        _s("2", "System", st[1]["action_text"], "X", "room search form", "Form display with filter fields",
           [_e("system", FU), _e("shows", AV), _e("a form to search rooms", DO)]),
        _s("3", "Hotel manager", st[2]["action_text"], "E", "room search criteria", "User provides filter criteria",
           [_e("user", FU), _e("fills in", AV), _e("the room code", DO)]),
        _s("4", "System", st[3]["action_text"], "R", "rooms", "System reads rooms from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("rooms", DO)]),
        _s("5", "System", st[4]["action_text"], "X", "rooms list", "System displays room list",
           [_e("system", FU), _e("displays", AV), _e("the list of rooms", DO)]),
        _s("6", "Hotel Manager", st[5]["action_text"] if len(st) > 5 else "The user browses.", None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
    ]))

    # FP2: View room details (4) = E+R+X+X(error)
    st = ps("FP2")
    recs.append(build_fp(system, "FP2", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], "E", "room identifier", "User provides room identifier",
           [_e("hotel manager", FU), _e("clicks", AV), _e("the \"View room details\" button", UI)]),
        _s("2", "System", st[1]["action_text"], "R", "room details", "System reads room from persistent storage",
           [_e("system", FU), _e("searches", AV), _e("the room", DO)]),
        _s("3", "System", st[2]["action_text"], "X", "room details", "System displays room details",
           [_e("system", FU), _e("shows", AV), _e("the room details", DO)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP3: Edit room details (6) = R(check)+X(form with data)+E+W+X(confirm)+X(error)
    recs.append(build_fp(system, "FP3", gt, [
        _s("1", "Hotel manager", "The hotel manager clicks the Edit room details button.", None, "", "UI navigation",
           [_e("hotel manager", FU), _e("clicks", AV)]),
        _s("2", "System", "The system checks if the room is already occupied, looking up the Booked Room File.",
           "R", "Booked Room File", "System reads booked room status",
           [_e("system", FU), _e("checks", AV), _e("the Booked Room File", PS)]),
        _s("3", "System", "The system displays a page with the Room's description and other details.",
           "X", "room details for editing", "System displays current room data for editing",
           [_e("system", FU), _e("displays", AV), _e("the Room's description", DO)]),
        _s("4", "Hotel manager", "The hotel manager modifies the room details fields and presses Save.",
           "E", "modified room data", "User provides modified data and submits",
           [_e("hotel manager", FU), _e("modifies", AV), _e("the room details fields", DO)]),
        _s("5", "System", "The system checks and records the new room details.",
           "W", "room details", "System writes room details to persistent storage",
           [_e("system", FU), _e("records", AV)]),
        _s("6", "System", "The system displays a message indicating that the operation was successful.",
           "X", "confirmation", "System displays confirmation to user",
           [_e("system", FU), _e("displays", AV)]),
        _s("EXC-1", "System", "The system shows an error message stating that the provided data are not valid.",
           "X", "error message", "System displays validation error",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP4: Insert new room (3)
    recs.append(_alb_insert(system, "FP4", gt, "Hotel manager", "room"))

    # FP5: Search services (4)
    st = ps("FP5")
    recs.append(build_fp(system, "FP5", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], None, "", "UI navigation",
           [_e("hotel manager", FU), _e("clicks", AV), _e("\"Services\"", UI)]),
        _s("2", "System", st[1]["action_text"], "X", "service search form", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Hotel manager", st[2]["action_text"], "E", "service search criteria", "User provides criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "System", st[3]["action_text"], "R", "services", "System reads services",
           [_e("system", FU), _e("searches", AV), _e("services", DO)]),
        _s("5", "System", st[4]["action_text"], "X", "services list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", "Hotel Manager", st[5]["action_text"] if len(st) > 5 else "The user browses.", None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
    ]))

    # FP6: View service details (4) = E+R+X+X(error)
    st = ps("FP6")
    recs.append(build_fp(system, "FP6", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], "E", "service identifier", "User provides identifier",
           [_e("hotel manager", FU), _e("presses", AV)]),
        _s("2", "System", st[1]["action_text"], "R", "service details", "System reads service",
           [_e("system", FU), _e("searches", AV)]),
        _s("3", "System", st[2]["action_text"], "X", "service details", "System displays details",
           [_e("system", FU), _e("shows", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP7: Edit service details (3)
    recs.append(_alb_edit(system, "FP7", gt, "Hotel manager", "service"))

    # FP8: Insert new service (3)
    recs.append(_alb_insert(system, "FP8", gt, "Hotel manager", "service"))

    # --- Customer management (UC-GES-03.x) ---

    # FP9: Search customers (4)
    st = ps("FP9")
    recs.append(build_fp(system, "FP9", gt, [
        _s("1", "User", st[0]["action_text"], None, "", "UI navigation",
           [_e("user", FU), _e("clicks", AV), _e("Customers", UI)]),
        _s("2", "System", st[1]["action_text"], "X", "customer search form", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "User", st[2]["action_text"], "E", "customer search criteria", "User provides criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "System", st[3]["action_text"], "R", "customers", "System reads customers",
           [_e("system", FU), _e("searches", AV), _e("customers", DO)]),
        _s("5", "System", st[4]["action_text"], "X", "customers list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", "User", st[5]["action_text"] if len(st) > 5 else "The user browses.", None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
    ]))

    # FP10: View customer details (4) = E+R+X+X(error)
    st = ps("FP10")
    recs.append(build_fp(system, "FP10", gt, [
        _s("1", "User", st[0]["action_text"], "E", "customer identifier", "User provides identifier",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "R", "customer details", "System reads customer",
           [_e("system", FU), _e("searches", AV)]),
        _s("3", "System", st[2]["action_text"], "X", "customer details", "System displays details",
           [_e("system", FU), _e("displays", AV), _e("the customer details", DO)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP11: Edit customer details (3)
    recs.append(_alb_edit(system, "FP11", gt, "User", "customer"))

    # FP12: Insert new customer (3)
    recs.append(_alb_insert(system, "FP12", gt, "User", "customer"))

    # --- Season configuration (UC-GES-04.x) ---

    # FP13: Search seasons config (4) = E+R+X+X(error)
    st = ps("FP13")
    recs.append(build_fp(system, "FP13", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], "E", "seasons config request", "Triggering entry",
           [_e("actor", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "R", "configured seasons", "System reads seasons",
           [_e("system", FU), _e("searches", AV)]),
        _s("3", "System", st[2]["action_text"], "X", "seasons list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("4", "Hotel manager", st[3]["action_text"], None, "", "Browsing",
           [_e("user", FU), _e("browses", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP14: View season config details (4) = E+R+X+X(error)
    st = ps("FP14")
    recs.append(build_fp(system, "FP14", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], "E", "season config identifier", "User provides identifier",
           [_e("hotel manager", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "R", "season config details", "System reads config",
           [_e("system", FU), _e("searches", AV)]),
        _s("3", "System", st[2]["action_text"], "X", "season config details", "System displays details",
           [_e("system", FU), _e("shows", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP15: Edit season config (3)
    recs.append(_alb_edit(system, "FP15", gt, "Hotel manager", "season configuration"))

    # FP16: Remove season config (3)
    recs.append(_alb_delete(system, "FP16", gt, "Hotel manager", "season configuration"))

    # FP17: Insert season config (3) — mapped to same Insert pattern, different entity name
    recs.append(_alb_insert(system, "FP17", gt, "Hotel manager", "season configuration"))

    # FP18: View rooms/services config (4) = E+R+X+X(error)
    st = ps("FP18")
    recs.append(build_fp(system, "FP18", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], "E", "rooms/services config request", "Triggering entry",
           [_e("actor", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "R", "rooms/services configuration", "System reads config",
           [_e("system", FU), _e("searches", AV)]),
        _s("3", "System", st[2]["action_text"], "X", "configuration details", "System displays config",
           [_e("system", FU), _e("displays", AV)]),
        _s("4", "Hotel manager", st[3]["action_text"], None, "", "Reviewing",
           [_e("user", FU)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP19: Edit rooms/services config (3)
    recs.append(_alb_edit(system, "FP19", gt, "Hotel manager", "rooms/services configuration"))

    # --- Room services (UC-GES-05.x) ---

    # FP20: View room services (4) = E+R+X+X(error)
    st = ps("FP20")
    recs.append(build_fp(system, "FP20", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], "E", "room identifier for services", "User provides room",
           [_e("hotel manager", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "R", "room services", "System reads assigned services",
           [_e("system", FU), _e("searches", AV)]),
        _s("3", "System", st[2]["action_text"], "X", "room services list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("4", "Hotel Manager", st[3]["action_text"], None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP21: Insert room service (6) = E(search)+R+X+E(select)+W+X(confirm)
    st = ps("FP21")
    recs.append(build_fp(system, "FP21", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], None, "", "UI navigation",
           [_e("hotel manager", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Hotel manager", st[2]["action_text"], "E", "service search criteria", "User provides search criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "System", st[3]["action_text"], "R", "services", "System reads available services",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "available services list", "System displays services",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", "Hotel Manager", st[5]["action_text"], None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
        _s("7", "Hotel manager", st[6]["action_text"], "E", "service selection", "User selects service",
           [_e("hotel manager", FU), _e("fills in", AV)]),
        _s("8", "System", st[7]["action_text"], None, "", "UI refresh — no domain data movement",
           [_e("system", FU)]),
        _s("9", "Hotel Manager", st[8]["action_text"], None, "", "Submit trigger",
           [_e("Hotel Manager", FU), _e("clicks", AV)]),
        _s("10", "System", st[9]["action_text"], "W", "room services", "System writes service assignment",
           [_e("system", FU), _e("saves", AV)]),
        _s("11", "System", st[10]["action_text"], "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("displays", AV)]),
    ]))

    # FP22: Delete room service (3) = E+W+X(confirm)
    recs.append(_alb_delete(system, "FP22", gt, "Hotel manager", "room service"))

    # --- Reservation (UC-PRE-xx) ---

    # FP23: Book a room (27) — complex multi-step booking
    st = ps("FP23")
    recs.append(build_fp(system, "FP23", gt, [
        _s("1", "Operator", st[0]["action_text"], None, "", "UI navigation",
           [_e("user", FU), _e("selects", AV)]),
        _s("2", "System", st[1]["action_text"], None, "", "Form display",
           [_e("system", FU), _e("displays", AV)]),
        _s("3", "Operator", st[2]["action_text"], None, "", "Filter input (no submit)",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "Operator", st[3]["action_text"], "E", "room search criteria", "User submits search criteria",
           [_e("user", FU), _e("clicks", AV)]),
        _s("5", "System", st[4]["action_text"], None, "", "Internal validation",
           [_e("system", FU)]),
        _s("6", "System", st[5]["action_text"], "R", "rooms by type", "System reads rooms",
           [_e("system", FU), _e("searches", AV)]),
        _s("7", "System", st[6]["action_text"], "R", "Commission file", "System reads commissions",
           [_e("system", FU), _e("checks", AV)]),
        _s("8", "System", st[7]["action_text"], "R", "Booked rooms", "System reads booked rooms",
           [_e("system", FU), _e("searches", AV)]),
        _s("9", "System", st[8]["action_text"], "X", "available rooms list", "System displays available rooms",
           [_e("system", FU), _e("lists", AV)]),
        _s("10", "Operator", st[9]["action_text"], None, "", "Browsing",
           [_e("user", FU), _e("browses", AV)]),
        _s("11", "Operator", st[10]["action_text"], "E", "selected rooms", "User selects rooms",
           [_e("actor", FU), _e("clicks", AV)]),
        _s("12a", "System", st[11]["action_text"].split(" and ")[0] if " and " in st[11]["action_text"] else st[11]["action_text"],
           "R", "room details", "System reads room details",
           [_e("system", FU), _e("reads", AV)]),
        _s("12b", "System", "The system searches for the requestable room services.",
           "R", "requestable services", "System reads services",
           [_e("system", FU), _e("searches", AV)]),
        _s("13", "System", st[12]["action_text"], "X", "rooms + services display", "System displays rooms and services",
           [_e("system", FU), _e("lists", AV)]),
        _s("14", "Operator", st[13]["action_text"], "E", "reservation selection", "User selects reservation details",
           [_e("user", FU), _e("presses", AV)]),
        _s("15a", "System", "The system searches for supplements in Configured season file.",
           "R", "season supplements", "System reads season supplements",
           [_e("system", FU), _e("searches", AV)]),
        _s("15b", "System", "The system lists the reservation details and shows a blank formatted screen.",
           "X", "reservation details + customer form", "System displays reservation and form",
           [_e("system", FU), _e("lists", AV)]),
        _s("16", "Operator", st[15]["action_text"], "E", "fiscal code", "User provides customer fiscal code",
           [_e("user", FU), _e("fills in", AV), _e("the fiscal code", DO)]),
        _s("17", "System", st[16]["action_text"], "R", "customer", "System reads customer by fiscal code",
           [_e("system", FU), _e("searches", AV)]),
        _s("18", "System", st[17]["action_text"], "X", "customer details", "System displays customer data",
           [_e("system", FU), _e("displays", AV), _e("the customer details", DO)]),
        _s("19", "User", st[18]["action_text"], None, "", "UI action — select",
           [_e("user", FU), _e("clicks", AV)]),
        _s("20", "System", st[19]["action_text"], "X", "customer details in form", "System fills form with customer data",
           [_e("system", FU), _e("fills", AV)]),
        _s("21", "Operator", st[20]["action_text"], None, "", "Optional input",
           [_e("user", FU)]),
        _s("22", "Operator", st[21]["action_text"], None, "", "Input (no submit)",
           [_e("user", FU), _e("fills in", AV)]),
        _s("23", "System", st[22]["action_text"], "W", "customer details", "System updates customer",
           [_e("system", FU), _e("updates", AV)]),
        _s("24a", "System", "The system searches for the season configuration.",
           "R", "season configuration", "System reads season config",
           [_e("system", FU), _e("searches", AV)]),
        _s("24b", "System", "The system stores the reservation header.",
           "W", "reservation header", "System writes reservation header",
           [_e("system", FU), _e("stores", AV)]),
        _s("24c", "System", "The system stores the booked rooms.",
           "W", "booked rooms", "System writes booked rooms",
           [_e("system", FU), _e("stores", AV)]),
        _s("24d", "System", "The system saves the requested services.",
           "W", "requested services", "System writes requested services",
           [_e("system", FU), _e("saves", AV)]),
        _s("24e", "System", "The system saves the children's ages.",
           "W", "children ages", "System writes children ages",
           [_e("system", FU), _e("saves", AV)]),
        _s("25", "System", st[24]["action_text"], "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
        # Alt flow: no customer found → E(customer data) + W(new customer)
        _s("ALT-17a1", "User", "The user fills in the customer fields and submits.",
           "E", "new customer data", "User provides new customer data (alt flow: customer not found)",
           [_e("user", FU), _e("fills in", AV)]),
        _s("ALT-18a1", "System", "The system records the new customer.",
           "W", "new customer", "System writes new customer (alt flow)",
           [_e("system", FU), _e("records", AV)]),
        # Alt flow: children's ages
        _s("ALT-7a1", "System", "The system presents drop-down lists to specify every child's age.",
           "X", "age selection form", "System displays age input form (alt flow: children present)",
           [_e("system", FU), _e("presents", AV)]),
        # Exception
        _s("EXC-9a1", "System", "The system shows an error message stating that no results have been found.",
           "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP24: Blocked rooms management (5) = E(timer)+R(config)+R(reservations)+W(clear)+W(save)
    recs.append(build_fp(system, "FP24", gt, [
        _s("1", "System timer", "The system timer ticks and requests the system to check blocked rooms.",
           "E", "timer event", "Triggering entry — timer initiates daily check",
           [_e("system timer", FU), _e("ticks", AV)]),
        _s("2a", "System", "The system searches for the maximum block time in configuration.",
           "R", "rooms/services configuration", "System reads max block duration",
           [_e("system", FU), _e("searches", AV)]),
        _s("2b", "System", "The system looks for reservations blocked beyond the limit.",
           "R", "overdue reservations", "System reads reservation data",
           [_e("system", FU), _e("looks", AV)]),
        _s("3", "System", "The system empties the Blocked Rooms Warnings file.",
           "W", "blocked rooms warnings (clear)", "System clears warnings file",
           [_e("system", FU), _e("empties", AV)]),
        _s("4", "System", "The system saves the reservations' IDs in the Blocked Rooms Warnings file.",
           "W", "blocked rooms warnings", "System writes new warnings",
           [_e("system", FU), _e("saves", AV)]),
    ]))

    # FP25: Search blocked rooms (4) = E+R+X+X(error)
    st = ps("FP25")
    recs.append(build_fp(system, "FP25", gt, [
        _s("1", "Hotel Manager", st[0]["action_text"], "E", "blocked rooms request", "Triggering entry",
           [_e("hotel manager", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "R", "blocked reservations", "System reads warnings",
           [_e("system", FU), _e("searches", AV)]),
        _s("3", "System", st[2]["action_text"], "X", "blocked rooms list", "System displays list",
           [_e("system", FU), _e("shows", AV)]),
        _s("4", "Hotel Manager", st[3]["action_text"], None, "", "Browsing",
           [_e("user", FU), _e("browses", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP26: Change reservation status (3) = E+W+X
    recs.append(_alb_edit(system, "FP26", gt, "User", "reservation status"))

    # FP27: View reservation details (10) = E+R×4+X×4+X(error)
    recs.append(_alb_view(system, "FP27", gt, "User", "reservation",
        ["The user clicks the \"Details\" button.",
         "The system searches for the reservation header, booked rooms, services, and ages.",
         "The system shows the reservation details, rooms, services, and ages."],
        "The system shows an error message.", extra_reads=3, extra_displays=3))

    # FP28: Cancel reservation (6) = E+W×4+X
    recs.append(_alb_delete(system, "FP28", gt, "User", "reservation", extra_w=3))

    # FP29: Edit reservation (23) — complex
    st = ps("FP29")
    recs.append(build_fp(system, "FP29", gt, [
        _s("1", "Operator", st[0]["action_text"], None, "", "UI navigation",
           [_e("user", FU), _e("selects", AV)]),
        _s("2", "System", st[1]["action_text"], None, "", "Form display",
           [_e("system", FU), _e("displays", AV)]),
        _s("3", "Operator", st[2]["action_text"], None, "", "Filter input",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "Operator", st[3]["action_text"], "E", "search criteria", "User submits search",
           [_e("user", FU), _e("clicks", AV)]),
        _s("5", "System", st[4]["action_text"], None, "", "Validation",
           [_e("system", FU)]),
        _s("6", "System", st[5]["action_text"], "R", "rooms", "System reads rooms",
           [_e("system", FU), _e("searches", AV)]),
        _s("7", "System", st[6]["action_text"], "R", "Commission file", "System reads commissions",
           [_e("system", FU), _e("checks", AV)]),
        _s("8", "System", st[7]["action_text"], "R", "Booked rooms", "System reads bookings",
           [_e("system", FU), _e("searches", AV)]),
        _s("9", "System", st[8]["action_text"], "X", "available rooms", "System displays rooms",
           [_e("system", FU), _e("lists", AV)]),
        _s("10", "Operator", st[9]["action_text"], "E", "selected rooms", "User selects rooms",
           [_e("actor", FU), _e("presses", AV)]),
        _s("11a", "System", "The system reads room details.",
           "R", "room details", "System reads room details",
           [_e("system", FU), _e("reads", AV)]),
        _s("11b", "System", "The system searches requestable services.",
           "R", "requestable services", "System reads services",
           [_e("system", FU), _e("searches", AV)]),
        _s("12", "System", st[11]["action_text"], "X", "rooms + services", "System displays data",
           [_e("system", FU), _e("lists", AV)]),
        _s("13", "Operator", st[12]["action_text"], "E", "reservation confirmation", "User confirms reservation",
           [_e("user", FU), _e("presses", AV)]),
        _s("14a", "System", "The system updates customer details.",
           "W", "customer", "System updates customer",
           [_e("system", FU), _e("updates", AV)]),
        _s("14b", "System", "The system removes previous Requested Services.",
           "W", "old requested services", "System removes old services",
           [_e("system", FU), _e("removes", AV)]),
        _s("14c", "System", "The system removes previous Booked Rooms.",
           "W", "old booked rooms", "System removes old rooms",
           [_e("system", FU), _e("removes", AV)]),
        _s("14d", "System", "The system removes previous Children's Ages.",
           "W", "old children ages", "System removes old ages",
           [_e("system", FU), _e("removes", AV)]),
        _s("14e", "System", "The system removes the previous Reservation header.",
           "W", "old reservation header", "System removes old header",
           [_e("system", FU), _e("removes", AV)]),
        _s("14f", "System", "The system searches for the season configuration.",
           "R", "season configuration", "System reads season config",
           [_e("system", FU), _e("searches", AV)]),
        _s("14g", "System", "The system stores the Reservation header.",
           "W", "reservation header", "System writes new header",
           [_e("system", FU), _e("stores", AV)]),
        _s("14h", "System", "The system stores the Booked Rooms.",
           "W", "booked rooms", "System writes new rooms",
           [_e("system", FU), _e("stores", AV)]),
        _s("14i", "System", "The system saves Requested Services.",
           "W", "requested services", "System writes new services",
           [_e("system", FU), _e("saves", AV)]),
        _s("14j", "System", "The system stores Children's Ages.",
           "W", "children ages", "System writes new ages",
           [_e("system", FU), _e("stores", AV)]),
        _s("15", "System", st[14]["action_text"] if len(st) > 14 else "The system displays a confirmation message.",
           "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("displays", AV)]),
        # Alt: children ages dropdown
        _s("ALT-11a1", "System", "The system presents drop-down lists for children's ages.",
           "X", "age dropdown", "System displays age input (alt flow)",
           [_e("system", FU), _e("presents", AV)]),
        # Exception
        _s("EXC-1", "System", "The system shows an error message.",
           "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP30: Change reservation services (15) — complex
    st = ps("FP30")
    recs.append(build_fp(system, "FP30", gt, [
        _s("1", "Operator", st[0]["action_text"], "E", "reservation identifier", "User provides reservation id",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2a", "System", "The system searches for room details.",
           "R", "room details", "System reads room details",
           [_e("system", FU), _e("searches", AV)]),
        _s("2b", "System", "The system searches for available services.",
           "R", "available services", "System reads available services",
           [_e("system", FU), _e("searches", AV)]),
        _s("2c", "System", "The system searches for already selected services.",
           "R", "selected services", "System reads current services",
           [_e("system", FU), _e("searches", AV)]),
        _s("2d", "System", "The system searches for extra services.",
           "R", "extra services", "System reads extra services",
           [_e("system", FU), _e("searches", AV)]),
        _s("2e", "System", "The system lists room details and services.",
           "X", "room details display", "System displays room details",
           [_e("system", FU), _e("lists", AV)]),
        _s("2f", "System", "The system lists selected services.",
           "X", "selected services display", "System displays selected services",
           [_e("system", FU), _e("lists", AV)]),
        _s("2g", "System", "The system lists requestable services.",
           "X", "requestable services display", "System displays requestable services",
           [_e("system", FU), _e("lists", AV)]),
        _s("2h", "System", "The system lists extra services.",
           "X", "extra services display", "System displays extra services",
           [_e("system", FU), _e("lists", AV)]),
        _s("3", "Operator", st[2]["action_text"], "E", "service selections", "User provides service choices",
           [_e("user", FU), _e("chooses", AV)]),
        _s("4a", "System", "The system stores the newly requested services.",
           "W", "requested services", "System writes new services",
           [_e("system", FU), _e("stores", AV)]),
        _s("4b", "System", "The system removes previous room services.",
           "W", "old room services", "System removes old services",
           [_e("system", FU), _e("removes", AV)]),
        _s("4c", "System", "The system updates the reservation header.",
           "W", "reservation header", "System updates reservation total",
           [_e("system", FU), _e("updates", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("displays", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP31: List customer's reservations (4) = X(form)+E+R+X
    st = ps("FP31")
    recs.append(build_fp(system, "FP31", gt, [
        _s("1", "User", st[0]["action_text"], None, "", "UI navigation",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "X", "reservation filter form", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "User", st[2]["action_text"], "E", "reservation search criteria", "User provides criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "System", st[3]["action_text"], "R", "reservations", "System reads reservations",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "reservation list", "System displays list",
           [_e("system", FU), _e("lists", AV)]),
        _s("6", "User", st[5]["action_text"], None, "", "Browsing",
           [_e("user", FU), _e("browses", AV)]),
    ]))

    # FP32: Search agencies (4) = X(form)+E+R+X
    st = ps("FP32")
    recs.append(build_fp(system, "FP32", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], None, "", "UI navigation",
           [_e("hotel manager", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "X", "agency search form", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Hotel manager", st[2]["action_text"], "E", "agency search criteria", "User provides criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "System", st[3]["action_text"], "R", "agencies", "System reads agencies",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "agency list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", "Hotel Manager", st[5]["action_text"], None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
    ]))

    # FP33: View agency details (6) = E+R(agency)+R(rooms)+X(agency)+X(rooms)+X(error)
    recs.append(_alb_view(system, "FP33", gt, "Hotel manager", "agency",
        ["The hotel manager clicks the view button.",
         "The system searches for the agency and commissioned rooms.",
         "The system shows the agency details and lists the commissioned rooms."],
        "The system shows an error message.", extra_reads=1, extra_displays=1))

    # FP34: Edit agency details (3)
    recs.append(_alb_edit(system, "FP34", gt, "Hotel manager", "agency"))

    # FP35: Insert agency (3)
    recs.append(_alb_insert(system, "FP35", gt, "Hotel manager", "agency"))

    # FP36: Add commissioned room (8) = E(dates)+R(commission)+R(booked)+R(room details)+X(list)+E(selection)+W+X(confirm)
    st = ps("FP36")
    recs.append(build_fp(system, "FP36", gt, [
        _s("1", "Hotel Manager", st[0]["action_text"], None, "", "UI navigation",
           [_e("hotel manager", FU), _e("selects", AV)]),
        _s("2", "System", st[1]["action_text"], None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Hotel Manager", st[2]["action_text"], "E", "date interval", "User provides date range",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4a", "System", "The system searches for uncommissioned rooms.",
           "R", "Commission file", "System reads commissions",
           [_e("system", FU), _e("searches", AV)]),
        _s("4b", "System", "The system searches for unbooked rooms.",
           "R", "Booked Room file", "System reads bookings",
           [_e("system", FU), _e("searches", AV)]),
        _s("4c", "System", "The system searches for room details.",
           "R", "Room file", "System reads room details",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "available rooms list", "System displays rooms",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", "Hotel manager", st[5]["action_text"], "E", "room selection", "User selects rooms",
           [_e("hotel manager", FU), _e("selects", AV)]),
        _s("7", "System", st[6]["action_text"], "W", "commissioned rooms", "System writes to Commission file",
           [_e("system", FU), _e("adds", AV)]),
        _s("8", "System", st[7]["action_text"], "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
    ]))

    # FP37: Remove commissioned room (3)
    recs.append(_alb_delete(system, "FP37", gt, "Hotel manager", "commissioned room"))

    # FP38: Search orders (4) = X(form)+E+R+X
    st = ps("FP38")
    recs.append(build_fp(system, "FP38", gt, [
        _s("1", "User", st[0]["action_text"], None, "", "UI navigation",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], "X", "order search form", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "User", st[2]["action_text"], "E", "order search criteria", "User provides criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "System", st[3]["action_text"], "R", "orders", "System reads orders",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "orders list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", "User", st[5]["action_text"], None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
    ]))

    # FP39: View order details (6) = E+R(order)+R(foods)+X(order)+X(foods)+X(error)
    recs.append(_alb_view(system, "FP39", gt, "User", "order",
        ["The user clicks the view button.",
         "The system searches for the order and ordered foods.",
         "The system displays the order details and lists the ordered foods."],
        "The system shows an error message.", extra_reads=1, extra_displays=1))

    # FP40: Place order (8) = R(tables)+X(tables)+E(order)+R(verify table)+W(table status)+W(order)+X(confirm)+X(error)
    st = ps("FP40")
    recs.append(build_fp(system, "FP40", gt, [
        _s("1", "Waiter", st[0]["action_text"], None, "", "UI navigation",
           [_e("waiter", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3a", "System", st[2]["action_text"].split(" and ")[0] if " and " in st[2]["action_text"] else st[2]["action_text"],
           "R", "free tables", "System reads Tables file for free tables",
           [_e("system", FU), _e("looks", AV)]),
        _s("3b", "System", "The system displays the list of free tables.",
           "X", "free tables list", "System displays free tables",
           [_e("system", FU), _e("displays", AV)]),
        _s("4", "Waiter", st[3]["action_text"], "E", "order data", "User provides order details",
           [_e("waiter", FU), _e("selects", AV)]),
        _s("5a", "System", "The system verifies the table is not occupied.",
           "R", "table verification", "System re-reads table status for verification",
           [_e("system", FU), _e("verifies", AV)]),
        _s("5b", "System", "The system updates the table status to Occupied.",
           "W", "table status", "System writes table status",
           [_e("system", FU), _e("updates", AV)]),
        _s("5c", "System", "The system inserts the new order.",
           "W", "order", "System writes new order",
           [_e("system", FU), _e("inserts", AV)]),
        _s("6", "System", st[5]["action_text"] if len(st) > 5 else "The system shows a confirmation.",
           "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP41: Add food to meal (9) = E(search)+R(foods)+R(categories)+X(categories+foods)+E(selection)+W(food)+W(order)+X(confirm)+X(error)
    st = ps("FP41")
    recs.append(build_fp(system, "FP41", gt, [
        _s("1", "Waiter", st[0]["action_text"], None, "", "UI navigation",
           [_e("waiter", FU), _e("clicks", AV)]),
        _s("2", "System", st[1]["action_text"], None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Waiter", st[2]["action_text"], "E", "food search criteria", "User provides criteria",
           [_e("waiter", FU), _e("fills in", AV)]),
        _s("4a", "System", "The system searches for foods.",
           "R", "foods", "System reads foods",
           [_e("system", FU), _e("searches", AV)]),
        _s("4b", "System", "The system searches for categories.",
           "R", "categories", "System reads categories",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "categories + food list", "System displays categories and foods",
           [_e("system", FU), _e("lists", AV)]),
        _s("6-7", "Waiter", "The waiter selects the food and submits.",
           "E", "food selection", "User selects food and submits",
           [_e("waiter", FU), _e("selects", AV)]),
        _s("8a", "System", "The system adds the ordered food.",
           "W", "ordered food", "System writes food",
           [_e("system", FU), _e("adds", AV)]),
        _s("8b", "System", "The system updates the order.",
           "W", "order", "System updates order total",
           [_e("system", FU), _e("updates", AV)]),
        _s("9", "System", st[8]["action_text"], "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP42: Remove food from meal (4) = E+W(food)+W(order total)+X
    recs.append(_alb_delete(system, "FP42", gt, "Waiter", "ordered food", extra_w=1))

    # FP43: Cancel order (5) = E+W(food)+W(order)+W(table)+X
    recs.append(_alb_delete(system, "FP43", gt, "User", "order", extra_w=2))

    # FP44: Prepare bill (3) = E+W+X
    recs.append(_alb_edit(system, "FP44", gt, "Waiter", "order billing"))

    # FP45: Print bill (6) = E+R(order)+R(food)+X(order print)+X(food print)+X(error)
    recs.append(build_fp(system, "FP45", gt, [
        _s("1", "Waiter", "The waiter clicks the \"Print bill\" button for the current order.",
           "E", "order identifier", "Triggering entry — user provides order identifier",
           [_e("waiter", FU), _e("clicks", AV)]),
        _s("2a", "System", "The system searches the order details.",
           "R", "order details", "System reads order",
           [_e("system", FU), _e("searches", AV)]),
        _s("2b", "System", "The system searches for the ordered food.",
           "R", "ordered food", "System reads ordered food",
           [_e("system", FU), _e("searches", AV)]),
        _s("3a", "System", "The system prints the order details.",
           "X", "order details (print)", "System outputs order to printer",
           [_e("system", FU), _e("prints", AV)]),
        _s("3b", "System", "The system prints the ordered food details.",
           "X", "food details (print)", "System outputs food to printer",
           [_e("system", FU), _e("prints", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP46: Pay bill (4) = E+W(order status)+W(table status)+X
    recs.append(_alb_delete(system, "FP46", gt, "User", "order payment", extra_w=1))

    # FP47: Search menu categories (4) = X(form)+E+R+X
    st = ps("FP47")
    recs.append(build_fp(system, "FP47", gt, [
        _s("1", "Hotel manager", st[0]["action_text"], None, "", "UI navigation",
           [_e("hotel manager", FU), _e("selects", AV)]),
        _s("2", "System", st[1]["action_text"], "X", "category search form", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "Hotel manager", st[2]["action_text"], "E", "category search criteria", "User provides criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4", "System", st[3]["action_text"], "R", "menu categories", "System reads categories",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", st[4]["action_text"], "X", "category list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("6", "Hotel Manager", st[5]["action_text"], None, "", "Browsing",
           [_e("Hotel Manager", FU), _e("browses", AV)]),
    ]))

    # FP48: View menu category details (6) = E+R(category)+R(foods)+X(category)+X(foods)+X(error)
    recs.append(_alb_view(system, "FP48", gt, "Hotel manager", "menu category",
        ["The hotel manager clicks the view button.",
         "The system searches for the category and associated foods.",
         "The system shows the category details and lists the foods."],
        "The system shows an error message.", extra_reads=1, extra_displays=1))

    # FP49: Edit menu category (3)
    recs.append(_alb_edit(system, "FP49", gt, "Hotel manager", "menu category"))

    # FP50: Insert menu category (3)
    recs.append(_alb_insert(system, "FP50", gt, "Hotel manager", "menu category"))

    # FP51: Remove menu category (4) = E+W(category)+W(foods)+X
    recs.append(_alb_delete(system, "FP51", gt, "Hotel manager", "menu category", extra_w=1))

    # FP52: Edit food (3)
    recs.append(_alb_edit(system, "FP52", gt, "Hotel manager", "food"))

    # FP53: Remove food (3)
    recs.append(_alb_delete(system, "FP53", gt, "Hotel manager", "food"))

    # FP54: Insert food (3)
    recs.append(_alb_insert(system, "FP54", gt, "Hotel manager", "food"))

    # FP55: Check-in (16) — complex
    recs.append(build_fp(system, "FP55", gt, [
        _s("1", "Hotel manager", "The user selects check-in.", None, "", "UI navigation",
           [_e("user", FU), _e("selects", AV)]),
        _s("2", "System", "The system displays reservation details.", "X", "reservation details", "System displays details",
           [_e("system", FU), _e("displays", AV)]),
        _s("3", "Hotel manager", "The user selects the room and executes the search.",
           "E", "room selection", "User provides room selection",
           [_e("user", FU), _e("selects", AV)]),
        _s("4a", "System", "The system searches for room guests.",
           "R", "room guests", "System reads guests",
           [_e("system", FU), _e("searches", AV)]),
        _s("4b", "System", "The system searches for each guest's details.",
           "R", "guest details", "System reads guest details",
           [_e("system", FU), _e("searches", AV)]),
        _s("5", "System", "The system lists the guests.", "X", "guests list", "System displays guests",
           [_e("system", FU), _e("lists", AV)]),
        _s("6", "Hotel manager", "The user clicks Add guest.", None, "", "UI action",
           [_e("user", FU), _e("clicks", AV)]),
        _s("7", "System", "The system shows a customer search form.", None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("8", "User", "The user fills in the filters and searches.",
           "E", "customer search criteria", "User provides search criteria",
           [_e("user", FU), _e("fills in", AV)]),
        _s("9", "System", "The system searches for customers.",
           "R", "customers", "System reads customers",
           [_e("system", FU), _e("searches", AV)]),
        _s("10", "System", "The system displays the customer list.", "X", "customer list", "System displays list",
           [_e("system", FU), _e("displays", AV)]),
        _s("11", "User", "The user clicks Select.", None, "", "UI action",
           [_e("user", FU), _e("clicks", AV)]),
        _s("12", "System", "The system displays the customer form.", "X", "customer form", "System displays form",
           [_e("system", FU), _e("displays", AV)]),
        _s("13-15", "Hotel manager", "The user enters details, selects room, and confirms.", None, "", "UI actions",
           [_e("user", FU)]),
        _s("17", "System", "The system updates the customer details.",
           "W", "customer", "System writes customer details",
           [_e("system", FU), _e("updates", AV)]),
        _s("18a", "System", "The system updates the booked room details.",
           "W", "booked room", "System updates booked room",
           [_e("system", FU), _e("updates", AV)]),
        _s("18b", "System", "The system inserts the new guest.",
           "W", "room guest", "System writes new guest",
           [_e("system", FU), _e("inserts", AV)]),
        _s("18c", "System", "The system changes the reservation status to Checked-in.",
           "W", "reservation status", "System updates reservation status",
           [_e("system", FU), _e("changes", AV)]),
        _s("19", "System", "The system shows a confirmation message.", "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
        # Alt: new customer
        _s("ALT-1", "User", "The user fills in customer fields and submits.",
           "E", "new customer data", "User provides new customer data (alt flow)",
           [_e("user", FU), _e("fills in", AV)]),
        _s("ALT-4", "System", "The system records the new customer.",
           "W", "new customer", "System writes new customer (alt flow)",
           [_e("system", FU), _e("records", AV)]),
    ]))

    # FP56: View current room services (6) = E+R(services)+R(service details)+X(services)+X(details)+X(error)
    st = ps("FP56")
    recs.append(build_fp(system, "FP56", gt, [
        _s("1", "User", st[0]["action_text"], None, "", "UI navigation",
           [_e("user", FU), _e("selects", AV)]),
        _s("2", "System", st[1]["action_text"], None, "", "Form display",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "User", st[2]["action_text"], "E", "service filter criteria", "User provides filter",
           [_e("user", FU), _e("fills in", AV)]),
        _s("4a", "System", "The system searches for the requested services.",
           "R", "requested services", "System reads requested services",
           [_e("system", FU), _e("searches", AV)]),
        _s("4b", "System", "The system looks up the service details.",
           "R", "service details", "System reads service details",
           [_e("system", FU), _e("looks up", AV)]),
        _s("5a", "System", "The system lists the currently requested services.",
           "X", "requested services list", "System displays services",
           [_e("system", FU), _e("lists", AV)]),
        _s("5b", "System", "The system lists the service details.",
           "X", "service details", "System displays service details",
           [_e("system", FU), _e("lists", AV)]),
        _s("6", "User", st[5]["action_text"], None, "", "Browsing",
           [_e("user", FU), _e("browses", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP57: Add services to current room (9) = E(room)+R(services)+R(service details)+X(services)+X(details)+E(selection)+W(services)+W(reservation)+X(confirm)
    st = ps("FP57")
    recs.append(build_fp(system, "FP57", gt, [
        _s("1", "User", st[0]["action_text"], "E", "room code", "User provides room code",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2a", "System", "The system searches for assigned services.",
           "R", "assigned services", "System reads assigned services",
           [_e("system", FU), _e("searches", AV)]),
        _s("2b", "System", "The system looks up service details.",
           "R", "service details", "System reads service details",
           [_e("system", FU), _e("looks up", AV)]),
        _s("3a", "System", "The system displays each service's details.",
           "X", "service details", "System displays service details",
           [_e("system", FU), _e("displays", AV)]),
        _s("3b", "System", "The system displays the list of assigned services.",
           "X", "assigned services list", "System displays assigned services",
           [_e("system", FU), _e("displays", AV)]),
        _s("4", "User", st[3]["action_text"], "E", "service selections", "User provides service selections",
           [_e("user", FU), _e("enters", AV)]),
        _s("5a", "System", "The system adds the selected services.",
           "W", "requested services", "System writes new services",
           [_e("system", FU), _e("adds", AV)]),
        _s("5b", "System", "The system updates the Reservation total amount.",
           "W", "reservation total", "System updates reservation",
           [_e("system", FU), _e("updates", AV)]),
        _s("6", "System", st[5]["action_text"], "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
    ]))

    # FP58: Telephone bill (8) = E(room)+R(phone bill)+X(bill form)+E(phone calls)+R(config)+W(phone bill)+W(reservation)+X(confirm)
    recs.append(build_fp(system, "FP58", gt, [
        _s("1", "User", "The user clicks the Telephone bill button.",
           "E", "room code", "User provides room code",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2a", "System", "The system looks up the telephone bill for the given room.",
           "R", "telephone bill", "System reads phone bill",
           [_e("system", FU), _e("looks up", AV)]),
        _s("2b", "System", "The system shows the Telephone bill details.",
           "X", "telephone bill details", "System displays bill details",
           [_e("system", FU), _e("shows", AV)]),
        _s("3", "User", "The user fills in the number of phone calls and submits.",
           "E", "phone calls count", "User provides phone call count",
           [_e("user", FU), _e("fills in", AV)]),
        _s("5a", "System", "The system searches for rooms and services configuration.",
           "R", "phone call cost", "System reads config for phone cost",
           [_e("system", FU), _e("searches", AV)]),
        _s("5b", "System", "The system inserts or updates the telephone bill.",
           "W", "telephone bill", "System writes phone bill",
           [_e("system", FU), _e("inserts", AV)]),
        _s("5c", "System", "The system updates the reservation total amount.",
           "W", "reservation total", "System updates reservation",
           [_e("system", FU), _e("updates", AV)]),
        _s("6", "System", "The system shows a confirmation message.",
           "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
    ]))

    # FP59: Calculate total bill (8) = E+R(reservation)+R(services)+R(phone bill)+X(total)+X(services)+X(phone bill)+X(error)
    recs.append(build_fp(system, "FP59", gt, [
        _s("1", "User", "The user clicks the View bill link.",
           "E", "reservation identifier", "User provides reservation identifier",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2a", "System", "The system looks up the reservation details and services.",
           "R", "reservation details", "System reads reservation",
           [_e("system", FU), _e("looks up", AV)]),
        _s("2b", "System", "The system searches for the requested services.",
           "R", "requested services", "System reads services",
           [_e("system", FU), _e("searches", AV)]),
        _s("2c", "System", "The system searches for the Telephone bill.",
           "R", "telephone bill", "System reads phone bill",
           [_e("system", FU), _e("searches", AV)]),
        _s("3a", "System", "The system shows the total amount.",
           "X", "total amount", "System displays total",
           [_e("system", FU), _e("shows", AV)]),
        _s("3b", "System", "The system shows the list of requested services.",
           "X", "services list", "System displays services",
           [_e("system", FU), _e("shows", AV)]),
        _s("3c", "System", "The system shows the telephone bill of each room.",
           "X", "telephone bill per room", "System displays phone bills",
           [_e("system", FU), _e("shows", AV)]),
        _s("4", "User", "The user presses the close button.", None, "", "UI action",
           [_e("user", FU), _e("presses", AV)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP60: Room change (21) — complex
    recs.append(build_fp(system, "FP60", gt, [
        _s("1", "User", "The user clicks the Change room link.",
           "E", "customer identifier", "User provides customer identifier",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2", "System", "The system searches for rooms of the given type.",
           "R", "rooms by type", "System reads rooms",
           [_e("system", FU), _e("searches", AV)]),
        _s("3a", "System", "The system checks the Commission file.",
           "R", "Commission file", "System reads commissions",
           [_e("system", FU), _e("checks", AV)]),
        _s("3b", "System", "The system checks the Booked Room file.",
           "R", "Booked Room file", "System reads bookings",
           [_e("system", FU), _e("checks", AV)]),
        _s("4", "System", "The system lists the available rooms.",
           "X", "available rooms", "System displays rooms",
           [_e("system", FU), _e("lists", AV)]),
        _s("5", "User", "The user selects a new room and clicks search services.",
           "E", "new room selection", "User provides room selection",
           [_e("user", FU), _e("selects", AV)]),
        _s("6a", "System", "The system searches for assigned services of the new room.",
           "R", "new room services", "System reads new room services",
           [_e("system", FU), _e("searches", AV)]),
        _s("6b", "System", "The system looks up requested services for the old room.",
           "R", "old room services", "System reads old room services",
           [_e("system", FU), _e("looks up", AV)]),
        _s("7a", "System", "The system shows the selected room details.",
           "X", "new room details", "System displays new room",
           [_e("system", FU), _e("shows", AV)]),
        _s("7b", "System", "The system lists the old room's requested services.",
           "X", "old services list", "System displays old services",
           [_e("system", FU), _e("lists", AV)]),
        _s("7c", "System", "The system lists the new room's assigned services.",
           "X", "new services list", "System displays new services",
           [_e("system", FU), _e("lists", AV)]),
        _s("8", "User", "The user optionally enters requestable services.", None, "", "Optional input",
           [_e("user", FU)]),
        _s("9a", "System", "The system searches for customer details.",
           "R", "customer details", "System reads customer",
           [_e("system", FU), _e("searches", AV)]),
        _s("9b", "System", "The system changes the room code in Customers file.",
           "W", "customer room code", "System updates customer room",
           [_e("system", FU), _e("changes", AV)]),
        _s("9c", "System", "The system updates the booked rooms.",
           "W", "booked rooms", "System updates booked rooms",
           [_e("system", FU), _e("updates", AV)]),
        _s("9d", "System", "The system deletes old requested services.",
           "W", "old requested services", "System removes old services",
           [_e("system", FU), _e("deletes", AV)]),
        _s("9e", "System", "The system inserts new requested services.",
           "W", "new requested services", "System writes new services",
           [_e("system", FU), _e("inserts", AV)]),
        _s("9f", "System", "The system changes the room code in Telephone Bill file.",
           "W", "telephone bill room code", "System updates phone bill room",
           [_e("system", FU), _e("changes", AV)]),
        _s("9g", "System", "The system updates the reservation total amount.",
           "W", "reservation total", "System updates reservation",
           [_e("system", FU), _e("updates", AV)]),
        _s("9h", "System", "The system updates the room code in Room Guest file.",
           "W", "room guest code", "System updates room guest",
           [_e("system", FU), _e("updates", AV)]),
        _s("10", "System", "The system shows a confirmation message.",
           "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
        _s("EXC-1", "System", "The system shows an error message.",
           "X", "error message", "System displays error for invalid room change",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    # FP61: Checkout (4) = E+W(reservation)+W(customer room)+X
    recs.append(build_fp(system, "FP61", gt, [
        _s("1", "User", "The user clicks the Checkout button.",
           "E", "reservation identifier", "User provides reservation identifier",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2a", "System", "The system changes the reservation status.",
           "W", "reservation status", "System updates reservation",
           [_e("system", FU), _e("changes", AV)]),
        _s("2b", "System", "The system updates the room reference in Customer file.",
           "W", "customer room reference", "System updates customer",
           [_e("system", FU), _e("updates", AV)]),
        _s("3", "System", "The system shows a confirmation message.",
           "X", "confirmation", "System displays confirmation",
           [_e("system", FU), _e("shows", AV)]),
    ]))

    # FP62: Print total bill (3) = E+X(bill)+X(error)
    recs.append(build_fp(system, "FP62", gt, [
        _s("1", "User", "The user clicks the Print bill button.",
           "E", "reservation identifier", "User provides reservation identifier",
           [_e("user", FU), _e("clicks", AV)]),
        _s("2", "System", "The system displays the formatted bill.",
           "X", "formatted bill", "System outputs bill for printing",
           [_e("system", FU), _e("displays", AV)]),
        _s("3", "User", "The user gives the printed bill to the customer.", None, "", "Physical action",
           [_e("user", FU)]),
        _s("EXC-1", "System", "The system shows an error message.", "X", "error message", "Error display",
           [_e("system", FU), _e("shows", AV), _e("an error message", DO)]),
    ]))

    save_system(system, recs)
    return recs


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building manual NER annotations...")
    annotate_ricecooker()
    annotate_automatic_line_switching()
    annotate_k01720()
    annotate_k01719()
    annotate_k01726()
    annotate_ucure()
    annotate_albergate()
    print("\nAll 7 systems annotated.")
