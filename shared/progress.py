"""
shared/progress.py — Progress tracking and monitoring for pipeline execution.
"""

import time
import threading


# ---------------------------------------------------------------------------
# Phase markers for progress percentage
# ---------------------------------------------------------------------------
PHASE_MARKERS = [
    ("PHASE 0: RESEARCH FRAMING", "Research Framing", 5),
    ("PHASE 1: DEEP RESEARCH", "Deep Research", 15),
    ("PHASE 2: LEAD RESEARCHER REPORT", "Lead Researcher Report", 20),
    ("PHASE 2b: RESEARCH GATE CHECK", "Research Gate Check", 25),
    ("PHASE 2c: GAP-FILL RESEARCH", "Gap-Fill Research", 30),
    ("PHASE 3: ADVERSARIAL RESEARCH", "Adversarial Research", 40),
    ("PHASE 4a: SOURCE VALIDATION", "Source Validation", 45),
    ("PHASE 4b: SOURCE-OF-TRUTH SYNTAX", "Source-of-Truth Synthesis", 50),
    ("PHASE 5: PODCAST PLANNING", "Podcast Planning", 60),
    ("PHASE 6: PODCAST RECORDING", "Podcast Recording", 75),
    ("PHASE 7: POST-PROCESSING", "Post-Processing", 90),
    ("PHASE 8: ACCURACY CHECK", "Accuracy Check", 95),
    ("PHASE 9: BGM MERGING", "BGM Merging", 98),
]


# ---------------------------------------------------------------------------
# Task metadata template
# ---------------------------------------------------------------------------
TASK_METADATA = {
    'framing_task': {
        'name': 'Research Framing & Hypothesis',
        'phase': '0',
        'estimated_duration_min': 2,
        'description': 'Defining scope, questions, and evidence criteria',
        'agent': 'Research Framing Specialist',
        'dependencies': [],
        'crew': 0,
    },
    'research_task': {
        'name': 'Deep Research Execution',
        'phase': '1',
        'estimated_duration_min': 8,
        'description': 'Systematic evidence gathering and data collection',
        'agent': 'Principal Investigator',
        'dependencies': ['framing_task'],
        'crew': 1,
    },
    'report_task': {
        'name': 'Lead Researcher Report',
        'phase': '2',
        'estimated_duration_min': 3,
        'description': 'Synthesizing initial findings into a structured report',
        'agent': 'Principal Investigator',
        'dependencies': ['research_task'],
        'crew': 1,
    },
    'gap_analysis_task': {
        'name': 'Gap Analysis (Internal Gate)',
        'phase': '2b',
        'estimated_duration_min': 2,
        'description': 'Checking for missing critical information',
        'agent': 'Scientific Auditor',
        'dependencies': ['report_task'],
        'crew': 1,
    },
    'gap_fill_task': {
        'name': 'Gap-Fill Research (Conditional)',
        'phase': '2c',
        'estimated_duration_min': 4,
        'description': 'Targeted supplementary research if needed',
        'agent': 'Principal Investigator',
        'dependencies': ['gap_analysis_task'],
        'crew': 'conditional',
        'conditional': True,
    },
    'adversarial_task': {
        'name': 'Adversarial Research',
        'phase': '3',
        'estimated_duration_min': 8,
        'description': 'Counter-evidence gathering and challenge',
        'agent': 'Adversarial Researcher',
        'dependencies': ['gap_analysis_task'],
        'crew': 2,
    },
    'source_verification_task': {
        'name': 'Source Validation (Audit Step 1)',
        'phase': '4a',
        'estimated_duration_min': 4,
        'description': 'Validating citations and checking claim-to-source accuracy',
        'agent': 'Scientific Source Verifier',
        'dependencies': ['adversarial_task'],
        'crew': 2,
    },
    'audit_task': {
        'name': 'Source-of-Truth Syntax (Audit Step 2)',
        'phase': '4b',
        'estimated_duration_min': 4,
        'description': 'Synthesizing all valid evidence into authoritative document',
        'agent': 'Scientific Auditor',
        'dependencies': ['source_verification_task'],
        'crew': 2,
    },
    'planning_task': {
        'name': 'Podcast Planning',
        'phase': '5',
        'estimated_duration_min': 3,
        'description': 'Developing show notes, outline, and narrative arc',
        'agent': 'Podcast Producer',
        'dependencies': ['audit_task'],
        'crew': 3,
    },
    'recording_task': {
        'name': 'Podcast Recording',
        'phase': '6',
        'estimated_duration_min': 6,
        'description': 'Script writing and conversation generation',
        'agent': 'Podcast Producer',
        'dependencies': ['planning_task'],
        'crew': 3,
    },
    'post_process_task': {
        'name': 'Post-Processing',
        'phase': '7',
        'estimated_duration_min': 5,
        'description': 'BGM merging, translation (if applicable), and polishing',
        'agent': 'Podcast Personality',
        'dependencies': ['recording_task'],
        'crew': 3,
    },
}


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Real-time progress tracking for CrewAI task execution."""

    def __init__(self, task_metadata: dict = None):
        self.task_metadata = task_metadata or TASK_METADATA
        self.task_names = list(self.task_metadata.keys())
        self.current_task_index = 0
        self.total_phases = len(
            [m for m in self.task_metadata.values() if not m.get('conditional', False)]
        )
        self.start_time = None
        self.task_start_time = None
        self.completed_tasks = []
        self.gate_passed = True

    def start_workflow(self):
        self.start_time = time.time()
        print(f"\n{'='*70}")
        print("WORKFLOW EXECUTION STARTED")
        print(f"{'='*70}\n")

    def task_started(self, task_index: int):
        if task_index >= len(self.task_names):
            return
        task_name = self.task_names[task_index]
        self.current_task_index = task_index
        self.task_start_time = time.time()
        metadata = self.task_metadata[task_name]

        print(f"\n{'='*70}")
        print(f"PHASE {metadata['phase']}/{self.total_phases}: {metadata['name'].upper()}")
        print(f"{'='*70}")
        print(f"Agent: {metadata['agent']}")
        print(f"Description: {metadata['description']}")
        print(f"Estimated Duration: {metadata['estimated_duration_min']} minutes")
        if metadata['dependencies']:
            deps_str = ', '.join([
                self.task_metadata[d]['name']
                for d in metadata['dependencies']
                if d in self.task_metadata
            ])
            print(f"Dependencies: {deps_str}")
        print("-" * 70)

    def task_completed(self, task_index: int):
        if task_index >= len(self.task_names):
            return
        task_name = self.task_names[task_index]
        elapsed_task = time.time() - self.task_start_time
        self.completed_tasks.append({'name': task_name, 'duration': elapsed_task})

        progress_pct = (len(self.completed_tasks) / self.total_phases) * 100
        elapsed_total = time.time() - self.start_time
        avg_time_per_task = elapsed_total / len(self.completed_tasks)
        remaining_tasks = self.total_phases - len(self.completed_tasks)
        estimated_remaining = avg_time_per_task * remaining_tasks

        metadata = self.task_metadata[task_name]

        print(f"\n{'='*70}")
        print(f"\u2713 PHASE {metadata['phase']}/{self.total_phases} COMPLETED")
        print(f"{'='*70}")
        print(f"Task Duration: {elapsed_task/60:.1f} minutes ({elapsed_task:.0f} seconds)")
        print(f"Total Elapsed: {elapsed_total/60:.1f} minutes")
        print(f"Progress: {progress_pct:.1f}% complete ({len(self.completed_tasks)}/{self.total_phases} tasks)")
        print(f"Estimated Remaining: {estimated_remaining/60:.1f} minutes")
        print(f"{'='*70}\n")

    def workflow_completed(self):
        total_time = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(" " * 22 + "WORKFLOW COMPLETED")
        print(f"{'='*70}")
        print(f"\nTotal Execution Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Tasks Completed: {len(self.completed_tasks)}/{self.total_phases}")

        print(f"\n{'Task Performance Summary':^70}")
        print("-" * 70)
        for i, task_info in enumerate(self.completed_tasks, 1):
            task_name = task_info['name']
            duration = task_info['duration']
            estimated = self.task_metadata[task_name]['estimated_duration_min'] * 60
            variance = ((duration - estimated) / estimated) * 100 if estimated > 0 else 0
            print(
                f"{i}. {self.task_metadata[task_name]['name']:<40} "
                f"{duration/60:>6.1f} min (est: {estimated/60:.1f} min, {variance:+.0f}%)"
            )
        print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CrewMonitor (background thread)
# ---------------------------------------------------------------------------
class CrewMonitor(threading.Thread):
    """Background thread that monitors crew execution progress."""

    def __init__(self, task_list, progress_tracker: ProgressTracker):
        super().__init__(daemon=True)
        self.task_list = task_list
        self.progress_tracker = progress_tracker
        self.running = True
        self.last_completed = -1

    def run(self):
        while self.running:
            try:
                completed_count = 0
                for task in self.task_list:
                    if hasattr(task, 'output') and task.output is not None:
                        completed_count += 1
                    else:
                        break
                if completed_count > self.last_completed:
                    if self.last_completed >= 0:
                        self.progress_tracker.task_completed(self.last_completed)
                    if completed_count < len(self.task_list):
                        self.progress_tracker.task_started(completed_count)
                    self.last_completed = completed_count
                time.sleep(3)
            except Exception:
                pass

    def stop(self):
        self.running = False


def display_workflow_plan(task_metadata: dict, topic_name: str, language_config: dict, output_dir):
    """Display detailed workflow plan before execution."""
    print("\n" + "=" * 70)
    print(" " * 20 + "PODCAST GENERATION WORKFLOW")
    print("=" * 70)
    print(f"\nTopic: {topic_name}")
    print(f"Language: {language_config['name']}")
    print(f"Output Directory: {output_dir}")
    print("\n" + "-" * 70)
    print(f"{'PHASE':<6} {'TASK NAME':<40} {'EST TIME':<12} {'AGENT':<25}")
    print("-" * 70)

    total_duration = 0
    for task_name, metadata in task_metadata.items():
        phase = metadata['phase']
        name = metadata['name']
        duration = metadata['estimated_duration_min']
        agent = metadata['agent']
        is_conditional = metadata.get('conditional', False)

        if not is_conditional:
            total_duration += duration

        conditional_marker = " [CONDITIONAL]" if is_conditional else ""
        print(f"{phase:<6} {name:<40} {duration:>3} min{'':<6} {agent:<25}{conditional_marker}")
        print(f"{'':6} \u2514\u2500 {metadata['description']}")
        if metadata['dependencies']:
            deps_str = ', '.join([
                f"Phase {task_metadata[d]['phase']}"
                for d in metadata['dependencies']
                if d in task_metadata
            ])
            print(f"{'':6}    Dependencies: {deps_str}")
        print()

    print("-" * 70)
    print(f"TOTAL ESTIMATED TIME: {total_duration} minutes (~{total_duration // 60}h {total_duration % 60}m)")
    print(f"  (+ up to 4 min if gap-fill triggers)")
    print("=" * 70 + "\n")
