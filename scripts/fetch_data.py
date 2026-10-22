"""Download a 3D vessel model from vascularmodel.com by its name.

Examples:
    python scripts/fetch_data.py --list --anatomy cerebral
    python scripts/fetch_data.py --model 0248_H_AOCERE_CAS
    python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --list-variants
    python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --variant 10Left100Right
    python scripts/fetch_data.py --model 0050_H_CERE_H
"""

from __future__ import annotations

import argparse

from stroke_nav.data.fetch import DEFAULT_RAW_DIR, fetch_model, list_variants, load_index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="VMR model name, e.g. 0248_H_AOCERE_CAS")
    p.add_argument("--variant", help="surface-model variant inside the zip (0248 stenosis variants)")
    p.add_argument("--dest", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--list", action="store_true", help="list models from the VMR index that have centerlines")
    p.add_argument("--anatomy", help="filter --list by anatomy substring (e.g. cerebral, aorta)")
    p.add_argument("--disease", help="filter --list by disease substring (e.g. stenosis, aneurysm)")
    p.add_argument("--list-variants", action="store_true", help="list surface-model variants of --model")
    p.add_argument("--no-build", action="store_true", help="only download; skip building the centerline graph")
    p.add_argument("--root", default=None, help="root centerline path for graph building (default: aorta or longest)")
    args = p.parse_args()

    if args.list:
        shown = 0
        for row in load_index():
            if row.get("Paths", "0").strip() != "1" or row.get("Models", "0").strip() != "1":
                continue
            if args.anatomy and args.anatomy.lower() not in row.get("Anatomy", "").lower():
                continue
            if args.disease and args.disease.lower() not in row.get("Disease", "").lower():
                continue
            print(f"{row['Name']:<28} {row.get('Anatomy', ''):<34} {row.get('Disease', '')}")
            shown += 1
        print(f"\n{shown} models with centerlines and surface models.")
        return

    if not args.model:
        p.error("--model is required (or use --list)")
    if args.list_variants:
        for v in list_variants(args.model):
            print(v)
        return
    fetch_model(args.model, variant=args.variant, dest=args.dest)
    if not args.no_build:
        from stroke_nav.data.centerline import build_model

        build_model(args.model, variant=args.variant, raw_dir=args.dest, root=args.root)


if __name__ == "__main__":
    main()
