"""Remove length-truncated records so they can be retried; supports dry-run selection."""

import argparse
import json

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
PILOT = REPO / "logs" / "interface_confound_pilot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--dataset", default="*")
    ap.add_argument("--interface", default="*")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pattern = f"{args.judge}_{args.dataset}_{args.interface}.jsonl"
    total_dropped = 0
    for path in sorted(PILOT.glob(pattern)):
        rows = [json.loads(l) for l in open(path)]
        keep = [r for r in rows if r.get("finish_reason") != "length"]
        dropped = len(rows) - len(keep)
        if dropped:
            total_dropped += dropped
            print(f"{path.name}: drop {dropped} truncated -> {len(keep)} kept")
            if not args.dry_run:
                with open(path, "w") as f:
                    for r in keep:
                        f.write(json.dumps(r) + "\n")
    print(
        f"{'[dry-run] would drop' if args.dry_run else 'dropped'} {total_dropped} truncated rows total"
    )


if __name__ == "__main__":
    main()
