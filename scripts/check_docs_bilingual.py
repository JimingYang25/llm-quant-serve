#!/usr/bin/env python3
"""Check that each translated document still carries the same content as its source.

The invariant this enforces is narrow but decisive: **a translation must not drop or
invent a figure.** Every numeric token in the source document has to appear in its
counterpart. Structure is checked the same way — section, heading, code-fence and
table-row counts — because a dropped paragraph or table row shows up there at a
glance, whereas prose can be reworded legitimately.

The language switcher under each H1 is checked too: a bilingual doc that nobody can
navigate out of is only half bilingual.

Stdlib only, so it runs in any environment — it does not need the llmquant env.

Usage:
    python scripts/check_docs_bilingual.py           # report; exit 1 on any violation
    python scripts/check_docs_bilingual.py -v        # also print per-section line counts

Exit status:
    0  every pair consistent
    1  at least one violation (missing figures, structure mismatch, missing switcher)
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# (source, translation) — the source side is the primary language of that document.
PAIRS: list[tuple[str, str]] = [
    ('README.md', 'README.zh-CN.md'),
    ('docs/report.md', 'docs/report.zh-CN.md'),
    ('docs/measurement_spec.md', 'docs/measurement_spec.zh-CN.md'),
    ('docs/glossary.md', 'docs/glossary.zh-CN.md'),
    ('docs/how_to_run.md', 'docs/how_to_run.zh-CN.md'),
    ('docs/ROADMAP.md', 'docs/ROADMAP.en.md'),
]

# A number, but not a digit run glued to a preceding word or hyphen: `batch-1`,
# `decision-10`, `wikitext-103` and `layers.10` are identifiers, not measurements.
NUM = re.compile(r'(?<![\w.\-])[-+]?\d[\d,]*(?:\.\d+)?')
HEADING = re.compile(r'^#{1,6} ')


def figures(text: str) -> list[str]:
    """Numeric tokens, with the unicode minus folded to ASCII and commas stripped."""
    text = text.replace('\u2212', '-')
    return [m.group(0).replace(',', '') for m in NUM.finditer(text)]


def structure(text: str) -> dict[str, int]:
    lines = text.split('\n')
    fences = sum(1 for l in lines if l.strip().startswith('```'))
    return {
        'lines': len(lines),
        'headings': sum(1 for l in lines if HEADING.match(l)),
        'code_blocks': fences // 2,
        'table_rows': sum(1 for l in lines if l.strip().startswith('|')),
    }


def has_switcher(text: str, counterpart_name: str) -> bool:
    """The counterpart's filename must appear in a link near the top of the document."""
    head = '\n'.join(text.split('\n')[:6])
    return counterpart_name in head


def sections(text: str) -> list[tuple[str, int]]:
    """(heading, non-blank body line count) for each section."""
    out: list[tuple[str, int]] = []
    title, count = '<preamble>', 0
    for l in text.split('\n'):
        if HEADING.match(l) and l.startswith('#'):
            out.append((title, count))
            title, count = l.strip(), 0
        elif l.strip():
            count += 1
    out.append((title, count))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='print per-section line counts for each pair')
    args = ap.parse_args()

    violations = 0
    for source_rel, trans_rel in PAIRS:
        p_src, p_tr = REPO / source_rel, REPO / trans_rel
        print(f'=== {source_rel}  <->  {trans_rel}')

        if not p_tr.exists():
            print(f'    MISSING: {trans_rel}\n')
            violations += 1
            continue

        src = p_src.read_text(encoding='utf-8')
        tr = p_tr.read_text(encoding='utf-8')

        src_figs, tr_figs = figures(src), figures(tr)
        tr_set = set(tr_figs)
        missing = sorted({f for f in src_figs if f not in tr_set})
        str_src, str_tr = structure(src), structure(tr)

        bad = False

        if missing:
            bad = True
            print(f'    DROPPED FIGURES ({len(missing)}): {missing[:30]}')
        if str_src['headings'] != str_tr['headings']:
            bad = True
            print(f"    HEADING COUNT {str_src['headings']} -> {str_tr['headings']}")
        if str_src['code_blocks'] != str_tr['code_blocks']:
            bad = True
            print(f"    CODE BLOCK COUNT {str_src['code_blocks']} -> {str_tr['code_blocks']}")
        if str_src['table_rows'] != str_tr['table_rows']:
            bad = True
            print(f"    TABLE ROW COUNT {str_src['table_rows']} -> {str_tr['table_rows']}")

        a, b = sections(src), sections(tr)
        if len(a) != len(b):
            bad = True
            print(f'    SECTION COUNT {len(a)} -> {len(b)}')

        # A switcher is only required by the file that links *to* its counterpart.
        if not has_switcher(tr, pathlib.Path(source_rel).name):
            bad = True
            print(f'    NO LANGUAGE SWITCHER under the H1 of {trans_rel}')
        if not has_switcher(src, pathlib.Path(trans_rel).name):
            bad = True
            print(f'    NO LANGUAGE SWITCHER under the H1 of {source_rel}')

        if bad:
            violations += 1
            print('    FAIL\n')
        else:
            print(f"    ok  {str_src['lines']} -> {str_tr['lines']} lines, "
                  f"{len(src_figs)} -> {len(tr_figs)} figures, "
                  f"{str_src['headings']} headings, "
                  f"{str_src['code_blocks']} code blocks, "
                  f"{str_src['table_rows']} table rows\n")

        if args.verbose:
            for (ta, na), (tb, nb) in zip(a, b):
                ratio = nb / na if na else 1.0
                warn = '   <== shrunk' if na >= 4 and ratio < 0.55 else ''
                print(f'      {na:4d} -> {nb:4d}  ({ratio:4.2f})  {ta[:58]}{warn}')
            print()

    if violations:
        print(f'FAILED: {violations} of {len(PAIRS)} document pairs are inconsistent.')
        return 1
    print(f'All {len(PAIRS)} document pairs are consistent.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
