"""Run a new experiment, or use --replay to verify the published results."""
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--audit-reasons' in args:
        if '--replay' in args:
            raise ValueError('理由审计与历史复算是两个命令，请分别执行')
        from extensions.audit_recorded_reasons import main as audit
        args.remove('--audit-reasons')
        return audit(args)
    if '--replay' in args:
        from scripts.replay import main as replay
        args.remove('--replay')
        return replay(args)
    from src import pipeline
    from extensions.guarded_agent import get_plan
    # Inject only for new runs; keep historical src hashes valid for replay.
    previous_planner = pipeline.get_plan
    pipeline.get_plan = get_plan
    try:
        return pipeline.main(args)
    finally:
        pipeline.get_plan = previous_planner


if __name__ == '__main__':
    raise SystemExit(main())
