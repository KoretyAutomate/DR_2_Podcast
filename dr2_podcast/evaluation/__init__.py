"""Self-learning evaluation loop for DR_2_Podcast pipeline."""

import logging

logger = logging.getLogger(__name__)


def run_self_evaluation(output_dir) -> None:
    """Run the full post-run evaluation once: scorecard -> lessons -> Telegram
    report -> threshold review. Non-blocking — failures are logged, never raised.

    This is the single entry point; each run directory must be evaluated
    exactly once (a second call re-sends Telegram and duplicates lessons in
    lessons_pending.json, inflating the promotion threshold).
    """
    try:
        from dr2_podcast.evaluation.scorecard import generate_scorecard
        from dr2_podcast.evaluation.lesson_generator import generate_lessons
        from dr2_podcast.evaluation.telegram_report import send_run_report
        from dr2_podcast.evaluation.lesson_reviewer import check_threshold, run_review

        scorecard = generate_scorecard(str(output_dir))
        lessons = generate_lessons(scorecard, str(output_dir))
        send_run_report(scorecard, lessons)

        if check_threshold():
            run_review()
    except Exception as eval_err:
        logger.warning("Self-evaluation error (non-blocking): %s", eval_err)
