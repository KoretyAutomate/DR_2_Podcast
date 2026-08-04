"""Remediate the 6 sleep-week episodes using the new accuracy machinery.

Each episode's own accuracy_audit.md already contains FAIL + exact corrections
(the historical bug was only that they were never enforced). For each episode:
  1. back up the current script_final.md
  2. if the audit FAILs / has HIGH drift / fabricated citations -> run the
     Smart-Model correction pass (applies the audit's corrections, fixes
     mislabeled Q&A blocks, drops duplicate closings)
  3. deterministic normalize speaker labels + dedup repeated blocks
  4. validate with the deterministic gates
  5. re-render audio from the corrected script

Backup: scripts/script_final.pre_remediation_2026-07-05.md per episode.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/korety/Project/DR_2_Podcast")
import dr2_podcast.pipeline as P
from dr2_podcast.pipeline import (
    SUPPORTED_LANGUAGES,
    _run_script_correction,
    _run_audio_pipeline,
    _audit_requires_correction,
)
from dr2_podcast.pipeline_script import _deduplicate_script
from dr2_podcast.pipeline_validators import (
    validate_citations,
    validate_script_structure,
    normalize_speaker_labels,
)

CFG = SUPPORTED_LANGUAGES["ja"]
P.language = "ja"
P.language_config = CFG
ROOT = Path("/home/korety/Project/DR_2_Podcast/research_outputs")

EPISODES = {
    "Mon-概観": "Ep016_睡眠は身体に良いのか_概観",
    "Tue-寿命": "Ep017_睡眠時間と寿命_U字曲線",
    "Wed-脳": "Ep018_睡眠と脳_グリンパティック系と認知症",
    "Thu-代謝": "Ep019_睡眠と代謝_血糖と食欲",
    "Fri-ホルモン": "Ep020_睡眠とホルモン_テストステロンと成長ホルモン",
    "Sat-行動": "Ep021_睡眠を妨げる習慣_カフェインと光と室温",
}

summary = {}
for day, d in EPISODES.items():
    print(f"\n{'=' * 70}\n=== {day}  ({d}) ===\n{'=' * 70}")
    base = ROOT / d
    sf = base / "scripts" / "script_final.md"
    try:
        orig = sf.read_text(encoding="utf-8")
        (base / "scripts" / "script_final.pre_remediation_2026-07-05.md").write_text(orig, encoding="utf-8")
        audit = (base / "research" / "accuracy_audit.md").read_text(encoding="utf-8")
        sot = (base / "research" / "source_of_truth.md").read_text(encoding="utf-8")

        det = validate_citations(orig, sot_text=sot)
        needs = _audit_requires_correction(audit) or bool(det)
        print(f"  audit_trigger={_audit_requires_correction(audit)} fabricated_citations={det}")

        script = orig
        if needs:
            aud = audit + (("\n\n## Deterministic Citation Issues\n" + "\n".join(f"- {i}" for i in det)) if det else "")
            print("  running correction pass ...")
            corrected = _run_script_correction(aud, orig, "ja", CFG, base)
            script = corrected if corrected else orig
            if corrected is None:
                print("  !! correction rejected — keeping original for this episode")
        else:
            print("  no correction needed (audit clean, no fabricated citations)")

        # deterministic cleanup
        script, fixed = normalize_speaker_labels(script)
        script = _deduplicate_script(script, CFG)
        print(f"  normalized {fixed} labels + deduped")

        r = validate_script_structure(script, sot_text=sot)
        print(f"  post-fix validate: pass={r['pass']} residual_issues={r['issues']}")

        sf.write_text(script, encoding="utf-8")
        print("  re-rendering audio ...")
        audio, dur = _run_audio_pipeline(script, base, CFG)
        print(f"  {day} DONE: audio={audio} dur={dur}")
        summary[day] = {
            "corrected": needs and script is not orig,
            "residual_issues": r["issues"],
            "duration_min": round(dur, 2) if dur else None,
            "audio": str(audio) if audio else None,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        summary[day] = {"error": f"{type(e).__name__}: {e}"}

print(f"\n{'=' * 70}\n=== REMEDIATION SUMMARY ===\n{'=' * 70}")
for day, s in summary.items():
    print(f"  {day}: {s}")
