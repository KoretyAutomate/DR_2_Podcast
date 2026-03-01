"""Entry point for `python -m dr2_podcast`.

Delegates to `python -m dr2_podcast.pipeline`, which runs the full pipeline.
"""
import runpy
runpy.run_module("dr2_podcast.pipeline", run_name="__main__", alter_sys=True)
