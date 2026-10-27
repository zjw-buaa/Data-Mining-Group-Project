"""Run a new experiment, or use --replay to verify the published results."""
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--replay' in args:
        from scripts.replay import main as replay
        args.remove('--replay')
        return replay(args)
    from src.pipeline import main as pipeline
    return pipeline(args)


if __name__ == '__main__':
    raise SystemExit(main())
