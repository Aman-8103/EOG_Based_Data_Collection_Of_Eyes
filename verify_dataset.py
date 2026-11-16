#!/usr/bin/env python3
"""
iSync — Automated Verification & Testing Suite (verify_dataset.py)

Simulates the complete research workflow using mock EOG streams:
1. S01: A01 (REJECT), A02 (REJECT), A03 (SELECT)
2. S02: A01 (REJECT), A02 (SELECT)
3. Duplicate Protection Test: S01_A04 (Attempt SELECT without replace -> BLOCKED)
4. Replacement Test: S01_A04 (SELECT with replace=True -> S01_A03 replaced by S01_A04)

Verifies:
- S01 timestamp starts at 0.000, S02 timestamp starts at 0.000.
- S01 sample index starts at 1, S02 sample index starts at 1.
- Raw CSVs and PNGs for ALL attempts exist in 'New Data/'.
- Final dataset 'LabelData/Eog Data.csv' has single header and only selected attempts.
- Trial ledger 'LabelData/trials_metadata.csv' correctly records attempts.
"""

import os
import time
import pandas as pd  # pyrefly: ignore [missing-import]

# Import functions from collector
from collector import (
    EOGAcquisitionEngine,
    get_next_attempt_id,
    save_raw_attempt,
    generate_attempt_plot,
    commit_selected_attempt,
    record_trial_metadata,
    assign_timeline_labels,
    FINAL_CSV_PATH,
    METADATA_CSV_PATH,
    NEW_DATA_DIR
)

def run_test_simulation():
    print("=" * 65)
    print(" Starting iSync EOG Automated Verification & Test Suite ")
    print("=" * 65)

    # Clean up existing test final csv and metadata for a clean test run
    if os.path.exists(FINAL_CSV_PATH):
        os.remove(FINAL_CSV_PATH)
        print(f"[Setup] Cleared {FINAL_CSV_PATH} for clean test run.")
    if os.path.exists(METADATA_CSV_PATH):
        os.remove(METADATA_CSV_PATH)
        print(f"[Setup] Cleared {METADATA_CSV_PATH} for clean test run.")

    # Clean test mock attempts from New Data (S01_A*, S02_A*)
    for pattern in ["S01_A*", "S02_A*"]:
        import glob
        for f in glob.glob(os.path.join(NEW_DATA_DIR, f"{pattern}")):
            os.remove(f)

    engine = EOGAcquisitionEngine(force_mock=True)

    # -------------------------------------------------------------
    # Test Workflow: Subject S01
    # A01 -> REJECT
    # A02 -> REJECT
    # A03 -> SELECT
    # -------------------------------------------------------------
    print("\n--- Testing Subject S01 ---")
    s01_attempts = [
        ("A01", "REJECT"),
        ("A02", "REJECT"),
        ("A03", "SELECT")
    ]

    for att_suffix, decision in s01_attempts:
        att_id = get_next_attempt_id("S01")
        print(f"\n[Trial] Recording S01 Attempt: {att_id} (Duration: 2.0s fast mock test)...")
        # For fast unit testing, record 2 seconds
        engine.start_recording(duration=2.0, sequence_id="Sequence_A")
        while engine.get_status()["recording"]:
            time.sleep(0.1)

        data = engine.get_data()
        csv_path, df = save_raw_attempt("S01", att_id, data)
        png_path = generate_attempt_plot("S01", att_id, df, "Sequence_A")

        assert os.path.exists(csv_path), f"Raw CSV missing: {csv_path}"
        assert os.path.exists(png_path), f"Inspection PNG missing: {png_path}"
        assert df["Timestamp"].iloc[0] == 0.0, f"Timestamp does not start at 0: {df['Timestamp'].iloc[0]}"
        assert df["Sample_Index"].iloc[0] == 1, f"Sample index does not start at 1: {df['Sample_Index'].iloc[0]}"

        if decision == "SELECT":
            ok, msg = commit_selected_attempt("S01", att_id, df, "Sequence_A")
            assert ok, f"Failed to commit selected: {msg}"
            print(f"  -> Decision: {decision} ({msg})")
        else:
            record_trial_metadata("S01", att_id, "Sequence_A", "REJECTED", len(df), df["Timestamp"].iloc[-1], False)
            print(f"  -> Decision: {decision} (Archived in New Data)")

    # -------------------------------------------------------------
    # Test Workflow: Subject S02
    # Verify attempt numbering restarts at A01!
    # A01 -> REJECT
    # A02 -> SELECT
    # -------------------------------------------------------------
    print("\n--- Testing Subject S02 (Per-Subject Attempt Reset) ---")
    s02_first_att = get_next_attempt_id("S02")
    assert s02_first_att == "S02_A01", f"Attempt numbering did NOT restart for S02: {s02_first_att}"
    print(f"  [PASS] S02 attempt numbering restarted correctly at {s02_first_att}.")

    s02_attempts = [
        ("A01", "REJECT"),
        ("A02", "SELECT")
    ]

    for att_suffix, decision in s02_attempts:
        att_id = get_next_attempt_id("S02")
        print(f"\n[Trial] Recording S02 Attempt: {att_id}...")
        engine.start_recording(duration=2.0, sequence_id="Sequence_A")
        while engine.get_status()["recording"]:
            time.sleep(0.1)

        data = engine.get_data()
        csv_path, df = save_raw_attempt("S02", att_id, data)
        png_path = generate_attempt_plot("S02", att_id, df, "Sequence_A")

        assert os.path.exists(csv_path), f"Raw CSV missing: {csv_path}"
        assert os.path.exists(png_path), f"Inspection PNG missing: {png_path}"
        assert df["Timestamp"].iloc[0] == 0.0, f"Timestamp does not start at 0: {df['Timestamp'].iloc[0]}"
        assert df["Sample_Index"].iloc[0] == 1, f"Sample index does not start at 1: {df['Sample_Index'].iloc[0]}"

        if decision == "SELECT":
            ok, msg = commit_selected_attempt("S02", att_id, df, "Sequence_A")
            assert ok, f"Failed to commit selected: {msg}"
            print(f"  -> Decision: {decision} ({msg})")
        else:
            record_trial_metadata("S02", att_id, "Sequence_A", "REJECTED", len(df), df["Timestamp"].iloc[-1], False)
            print(f"  -> Decision: {decision} (Archived in New Data)")

    # -------------------------------------------------------------
    # Verification of Final CSV Dataset (LabelData/Eog Data.csv)
    # -------------------------------------------------------------
    print("\n--- Verifying Curated Dataset (LabelData/Eog Data.csv) ---")
    assert os.path.exists(FINAL_CSV_PATH), "Final dataset does not exist!"
    df_final = pd.read_csv(FINAL_CSV_PATH)

    subjects = df_final["Subject_ID"].unique()
    attempts = df_final["Attempt_ID"].unique()

    print(f"  Subjects present: {list(subjects)}")
    print(f"  Attempts present: {list(attempts)}")

    # Check that S01 and S02 are present, and only S01_A03 and S02_A02 exist
    assert set(subjects) == {"S01", "S02"}, f"Unexpected subjects: {subjects}"
    assert set(attempts) == {"S01_A03", "S02_A02"}, f"Final dataset has wrong attempts: {attempts}"
    assert "S01_A01" not in attempts, "Rejected attempt S01_A01 leaked into final dataset!"
    assert "S01_A02" not in attempts, "Rejected attempt S01_A02 leaked into final dataset!"
    assert "S02_A01" not in attempts, "Rejected attempt S02_A01 leaked into final dataset!"

    # Verify S01 and S02 both restart timestamp at 0.000
    s01_subset = df_final[df_final["Subject_ID"] == "S01"]
    s02_subset = df_final[df_final["Subject_ID"] == "S02"]

    assert s01_subset["Timestamp"].iloc[0] == 0.0, "S01 timestamp did not start at 0"
    assert s02_subset["Timestamp"].iloc[0] == 0.0, "S02 timestamp did not start at 0"
    assert s01_subset["Sample_Index"].iloc[0] == 1, "S01 Sample_Index did not start at 1"
    assert s02_subset["Sample_Index"].iloc[0] == 1, "S02 Sample_Index did not start at 1"

    # Verify labels in final dataset
    valid_labels = {"REST", "LEFT", "RIGHT", "UP", "DOWN"}
    present_labels = set(df_final["Label"].unique())
    assert present_labels.issubset(valid_labels), f"Invalid labels found: {present_labels}"
    print(f"  [PASS] Labels in final dataset valid: {present_labels}")

    # Comprehensive 20-second timeline mapping verification
    test_times = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 17.5]
    expected_labels = ["REST", "RIGHT", "REST", "LEFT", "REST", "UP", "REST", "DOWN", "REST"]
    dummy_df = pd.DataFrame({"Timestamp": test_times, "EOG": [500] * len(test_times)})
    labeled_dummy = assign_timeline_labels(dummy_df, "Sequence_A")
    assert list(labeled_dummy["Label"]) == expected_labels, f"Timeline label mismatch: {list(labeled_dummy['Label'])} vs {expected_labels}"
    print(f"  [PASS] Full 20-second stimulus timeline correctly maps all 5 states: {set(expected_labels)}")

    # Verify header occurs exactly once
    with open(FINAL_CSV_PATH, 'r') as f:
        content = f.read()
    header_count = content.count("Subject_ID,Attempt_ID,Sample_Index,Timestamp,EOG,Label")
    assert header_count == 1, f"Header appears {header_count} times, expected exactly 1!"
    print("  [PASS] Final CSV header appears exactly once.")

    # -------------------------------------------------------------
    # Test Duplicate Protection
    # -------------------------------------------------------------
    print("\n--- Testing Duplicate Subject Protection ---")
    att_s01_new = get_next_attempt_id("S01")  # Should be S01_A04
    engine.start_recording(duration=1.0, sequence_id="Sequence_A")
    while engine.get_status()["recording"]:
        time.sleep(0.1)
    d = engine.get_data()
    _, df_new_s01 = save_raw_attempt("S01", att_s01_new, d)
    generate_attempt_plot("S01", att_s01_new, df_new_s01, "Sequence_A")

    # Attempt to commit without replace -> MUST FAIL
    ok, msg = commit_selected_attempt("S01", att_s01_new, df_new_s01, replace=False)
    assert not ok, "Duplicate commit succeeded when it should have been blocked!"
    print(f"  [PASS] Duplicate commit correctly blocked: {msg}")

    # Commit with explicit replace -> MUST SUCCEED and REPLACE S01_A03 with S01_A04
    ok, msg = commit_selected_attempt("S01", att_s01_new, df_new_s01, replace=True)
    assert ok, f"Explicit replacement failed: {msg}"
    df_after_replace = pd.read_csv(FINAL_CSV_PATH)
    attempts_after = set(df_after_replace["Attempt_ID"].unique())
    assert "S01_A04" in attempts_after, "Replaced attempt S01_A04 missing!"
    assert "S01_A03" not in attempts_after, "Old attempt S01_A03 was not removed upon replacement!"
    print(f"  [PASS] Replacement succeeded cleanly. New dataset attempts: {attempts_after}")

    # -------------------------------------------------------------
    # Verify New Data Directory Integrity
    # -------------------------------------------------------------
    print("\n--- Verifying Raw Storage in New Data/ ---")
    expected_files = [
        "S01_A01.csv", "S01_A01.png",
        "S01_A02.csv", "S01_A02.png",
        "S01_A03.csv", "S01_A03.png",
        "S01_A04.csv", "S01_A04.png",
        "S02_A01.csv", "S02_A01.png",
        "S02_A02.csv", "S02_A02.png"
    ]
    for ef in expected_files:
        p = os.path.join(NEW_DATA_DIR, ef)
        assert os.path.exists(p), f"Expected raw file missing in New Data: {ef}"
    print(f"  [PASS] All {len(expected_files)} raw CSV and PNG files preserved in New Data/.")

    # -------------------------------------------------------------
    # Verify Trials Metadata Ledger
    # -------------------------------------------------------------
    print("\n--- Verifying LabelData/trials_metadata.csv ---")
    assert os.path.exists(METADATA_CSV_PATH), "trials_metadata.csv missing!"
    df_meta = pd.read_csv(METADATA_CSV_PATH)
    print(f"  Total logged trials: {len(df_meta)}")
    assert len(df_meta) >= 5, "Metadata ledger missing trial records"
    print("  [PASS] Metadata ledger contains all attempt logs.")

    print("\n" + "=" * 65)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 65)

if __name__ == "__main__":
    run_test_simulation()
