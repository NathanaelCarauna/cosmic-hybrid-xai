#!/usr/bin/env python3
"""Constrói o dataset C-REG (validação externa out-of-sample, GT canônico COSMIC).

Fonte: COSMIC Consortium, "Course Registration (C-Reg) System Case Study" v2.0.1
(2018), medido pelo COSMIC Measurement Practices Committee sob ISO/IEC 19761.
19 processos funcionais em formato de caso de uso; CFP total 102 (Tabela 3 do
case study). Texto dos requisitos transcrito fielmente das seções 2.1, 3.1 e 4.1
(passos lettered a/b/c... preservados como passos do cenário principal).

Gera: datasets/cosmic-casestudies/c-reg.csv (fp_id, name, cfp_gt, text).
"""
from __future__ import annotations

import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# (id, nome, CFP canônico, [passos verbatim do requisito])
FPS = [
    ("CREG-01", "Add a Professor", 4, [
        "The Registrar enters the following details for the Professor: ID, name and address, social security number, Department, qualifications and contact details and presses Save.",
        "C-Reg validates the data to ensure the proper formats and checks whether a Professor of that ID already exists.",
        "If the entered data are valid, C-Reg creates a record for the new Professor.",
        "Alternatively, if data entered is not valid, C-Reg displays one or more error messages.",
    ]),
    ("CREG-02", "Enquire on a Professor", 4, [
        "The Registrar enters the Professor ID.",
        "C-Reg searches for a Professor with the specified ID.",
        "C-Reg displays the Professor's name and address and other details.",
        "Alternatively, if a Professor with the specified ID is not found, C-Reg displays an error message.",
    ]),
    ("CREG-03", "Modify a Professor", 3, [
        "The Registrar may change one or more of the displayed Professor data items, then presses Save.",
        "C-Reg updates the professor data.",
        "Alternatively, if data entered is not valid, C-Reg displays an error message.",
    ]),
    ("CREG-04", "Delete a Professor", 5, [
        "The Registrar retrieves the Professor data and presses Delete.",
        "C-Reg enquires the Course Catalog whether the Professor has any Course Offerings that he has committed to teach.",
        "C-Reg displays a message asking the Registrar to confirm the deletion.",
        "If the Registrar selects yes, the Professor data is deleted from C-Reg.",
        "If the Professor is committed to teach any Course Offerings, deletion is not allowed and C-Reg displays an error message.",
    ]),
    ("CREG-05", "Enquire on Course Offerings (Professor)", 7, [
        "The Professor enters his ID.",
        "C-Reg obtains the Professor's qualifications and Department, and sends these to the Course Catalog.",
        "C-Reg obtains from the Course Catalog and displays the list of Course Offerings for the Professor's Department that he is qualified to teach and that are unavailable.",
        "The scheduled month and room for each Course Offering are also shown.",
        "Alternatively, if there are no Course Offerings that the Professor may commit to teach, C-Reg displays an error message.",
    ]),
    ("CREG-06", "Create Course Offering commitments", 6, [
        "The Professor selects the Course Offerings that he will commit to teach by adding his ID to the selections.",
        "C-Reg returns each selected Course Offering the Professor has committed to teach to the Course Catalog.",
        "The Course Catalog checks if the selected Course Offerings conflict on date or location and returns a message to C-Reg containing the count of conflicting Course Offerings and a pair of IDs for each conflict.",
        "If any Course Offerings conflict on date or location, C-Reg indicates the IDs of the conflicting pairs on the display with an error message.",
        "C-Reg sends the details of any changed Course Offerings back to the Course Catalog.",
    ]),
    ("CREG-07", "Modify Course Offering commitments", 7, [
        "The Professor modifies the Course Offerings that he commits to teach by adding or removing his ID from the Course Offerings.",
        "C-Reg returns each modified Course Offering to the Course Catalog.",
        "The Course Catalog checks for conflicts and returns a message to C-Reg with the count of conflicting Course Offerings.",
        "If any Course Offerings conflict, C-Reg displays the conflicting pairs with an error message.",
        "C-Reg sends data for each modified Course Offering to all Students via a broadcast e-mail.",
    ]),
    ("CREG-08", "Delete Course Offering commitments", 4, [
        "The Professor re-enters his ID.",
        "C-Reg displays a message asking the Professor to confirm that he wants to delete all his commitments.",
        "If the Professor selects yes, C-Reg sends the Course Offering data to the Course Catalog, which changes their status to unavailable.",
        "C-Reg sends the ID of each deleted Course Offering to all Students via a broadcast e-mail.",
    ]),
    ("CREG-09", "Add a Student's details", 4, [
        "The Registrar enters the following details for the student: name, date of birth, social security number, status, and graduation date.",
        "C-Reg validates the data to ensure the proper formats.",
        "If the data are valid, C-Reg creates a new student record and assigns a unique system-generated ID, which is displayed.",
        "Alternatively, if invalid data are entered, C-Reg displays an error message.",
    ]),
    ("CREG-10", "Enquire on a Student's details", 4, [
        "The Registrar enters a Student ID.",
        "C-Reg searches for a Student with the specified ID.",
        "If the ID is valid, C-Reg shows the Student's name, date of birth, status and graduation date.",
        "Alternatively, if the Student ID is not found, C-Reg displays an error message.",
    ]),
    ("CREG-11", "Modify a Student's details", 3, [
        "The Registrar modifies one or more of the displayed student data items, then presses Save.",
        "C-Reg validates the data and if correct updates the Student details.",
        "Alternatively, if invalid data are entered, C-Reg displays an error message.",
    ]),
    ("CREG-12", "Delete a Student's details", 4, [
        "The Registrar retrieves the Student's details and presses Delete.",
        "C-Reg checks whether the Student has a Student Schedule.",
        "If the Student does not have a Student Schedule, C-Reg displays a message asking the Registrar to confirm the deletion, and the student is deleted from C-Reg.",
        "Alternatively, if the Student has a Student Schedule, deletion is not allowed and C-Reg displays an error message.",
    ]),
    ("CREG-13", "Enquire on Course Offerings (Student)", 6, [
        "The Student enters his ID.",
        "C-Reg retrieves the details of all Course Offerings, their availability status, and the ID of a pre-requisite Course for each Course Offering from the Course Catalog.",
        "C-Reg displays these Course Offering details and the entered Student ID.",
    ]),
    ("CREG-14", "Create a Student Schedule", 6, [
        "The Student selects at most six available Course Offerings from the displayed list for his Schedule and selects Submit, which results in his ID and the selections being entered.",
        "C-Reg checks whether the Student has achieved the necessary pre-requisite for each Course Offering selected for enrollment.",
        "If the check is OK, C-Reg accepts each selected Course Offering as an item in the Student's Schedule with the status enrolled.",
        "Alternatively, if the Student has not satisfied the necessary pre-requisite, an error message is displayed.",
        "C-Reg stores the Student Schedule items.",
        "C-Reg sends the IDs of the added Course Offerings to the Course Catalog so that it can update the number of Students enrolled for each Course Offering.",
    ]),
    ("CREG-15", "Modify a Student Schedule", 8, [
        "C-Reg retrieves the Student's current Schedule items, both enrolled and selected, and adds this status to each of the displayed Course Offerings that are available.",
        "The Student adds and/or deletes Course Offerings to/from the selections in his current Schedule and selects Submit.",
        "C-Reg checks whether the Student has achieved the necessary prerequisite for each Course Offering now selected for enrollment.",
        "If the check is OK, C-Reg updates the selected Course Offering in the Schedule with the status enrolled.",
        "Alternatively, if the Student has not satisfied the necessary prerequisite, an error message is displayed.",
        "C-Reg stores the Student Schedule items.",
        "C-Reg sends the IDs of the modified Course Offerings to the Course Catalog so that it can update the number of Students enrolled for each Course Offering.",
    ]),
    ("CREG-16", "Delete a Student Schedule", 6, [
        "The Student enters his ID.",
        "C-Reg retrieves and displays his Student Schedule items, his list of enrolled and selected Course Offerings.",
        "C-Reg prompts the Student to verify the deletion; the Student verifies the deletion.",
        "C-Reg deletes the Schedule items and issues a confirmation message.",
        "C-Reg sends the IDs of the deleted Course Offerings to the Course Catalog so that it can update the number of students enrolled for each course.",
        "C-Reg issues an error message in case the Student's Schedule is not found.",
    ]),
    ("CREG-17", "Monitor Course Offering Enrolment progress", 6, [
        "When a Registrar wishes to check progress on Course Offering enrollment, he selects the sub-option Monitor Course Offering Enrollment.",
        "C-Reg requests from the Course Catalog and receives all Course Offerings including their current status, the name of the Department that owns the Course, the number of students enrolled, and the maximum number of students that may be enrolled.",
        "C-Reg sorts the received Course Offerings by Department name and produces a report showing, for each Department and for the whole College, the actual and maximum number of Students enrolled and the enrollment percentage.",
    ]),
    ("CREG-18", "Monitor Student Schedule Enrolment progress", 5, [
        "When a Registrar wishes to check progress on Student Schedule enrollment, he selects the sub-option Monitor Student Schedules.",
        "C-Reg retrieves the Student Schedule items for each Student and the names of all Students that have not enrolled for or selected any Course Offerings.",
        "C-Reg displays the name of every Student that has no Student Schedule items and the overall percentage of enrollments in Student Schedules for the College.",
    ]),
    ("CREG-19", "Close Registration", 10, [
        "When a Registrar wishes to close all registration processes, he selects the sub-option Close Registration.",
        "C-Reg obtains the Course Offerings from the Course Catalog that are available or full.",
        "For each Course Offering that has less than 3 students enrolled, C-Reg sets the Course Offering status to cancelled; for all other Course Offerings, C-Reg sets the status to closed.",
        "C-Reg returns the Course Offering data to the Course Catalog so that it can update Course Offering statuses in its files.",
        "For each combination of enrolled Student and closed Course Offering, C-Reg sends the Student ID and the Course Offering ID to the Billing System.",
        "C-Reg prepares and issues e-mails announcing cancelled Course Offerings to be sent to each affected Student via the E-mail System.",
    ]),
]


def to_uc_text(name: str, steps: list[str]) -> str:
    lines = [f"ID\nUC\t{name}", f"Description\t{name}.", "MAIN SCENARIO"]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}\t\t{s}")
    return "\n".join(lines) + "\n"


def main() -> None:
    out = BASE / "datasets" / "cosmic-casestudies"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    with (out / "c-reg.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fp_id", "name", "cfp_gt", "text"])
        for fid, name, cfp, steps in FPS:
            w.writerow([fid, name, cfp, to_uc_text(name, steps)])
            total += cfp
    assert len(FPS) == 19, len(FPS)
    assert total == 102, total  # confere com a Tabela 3 do case study
    print(f"C-REG dataset: {len(FPS)} processos funcionais, CFP total {total} (confere com 102)")
    print(f"-> {out}/c-reg.csv")


if __name__ == "__main__":
    main()
