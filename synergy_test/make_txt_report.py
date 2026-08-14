#!/usr/bin/env python3
"""Render REPORT.md to a plain-text REPORT.txt with aligned tables."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "REPORT.md")
DST = os.path.join(HERE, "REPORT.txt")

WIDTH = 100


def strip_inline(s):
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    s = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', s)
    s = s.replace('\\*', '*').replace('|', '|')
    return s


def is_table_row(line):
    return line.strip().startswith('|') and line.strip().endswith('|')


def is_sep_row(line):
    return bool(re.fullmatch(r'\|[\s:\-|]+\|', line.strip()))


def render_table(rows):
    cells = [[strip_inline(c.strip()) for c in r.strip().strip('|').split('|')]
             for r in rows]
    ncol = max(len(r) for r in cells)
    for r in cells:
        r += [''] * (ncol - len(r))
    w = [max(len(r[i]) for r in cells) for i in range(ncol)]
    out = []
    header = '  '.join(cells[0][i].ljust(w[i]) for i in range(ncol)).rstrip()
    out.append(header)
    out.append('-' * len(header))
    for r in cells[1:]:
        out.append('  '.join(r[i].ljust(w[i]) for i in range(ncol)).rstrip())
    return out


def wrap(text, indent=''):
    words, line, out = text.split(), '', []
    for wd in words:
        if len(line) + len(wd) + 1 > WIDTH - len(indent):
            out.append(indent + line)
            line = wd
        else:
            line = (line + ' ' + wd).strip()
    if line:
        out.append(indent + line)
    return out


raw_lines = open(SRC, encoding='utf-8').read().split('\n')


def special(l):
    return (not l.strip() or l.startswith('#') or is_table_row(l)
            or l.strip().startswith(('```', '---', '***', '>'))
            or re.match(r'^(\s*)([-*]|\d+\.)\s+', l))


# join markdown soft-wrapped continuation lines so wrap()/indent works
lines, k, code = [], 0, False
while k < len(raw_lines):
    l = raw_lines[k]
    if l.strip().startswith('```'):
        code = not code
    if code or special(l):
        if re.match(r'^(\s*)([-*]|\d+\.)\s+', l) and not code:
            buf = l
            k += 1
            while k < len(raw_lines) and raw_lines[k].strip() and not special(raw_lines[k]):
                buf += ' ' + raw_lines[k].strip()
                k += 1
            lines.append(buf)
            continue
        lines.append(l)
        k += 1
        continue
    buf = l
    k += 1
    while k < len(raw_lines) and raw_lines[k].strip() and not special(raw_lines[k]):
        buf += ' ' + raw_lines[k].strip()
        k += 1
    lines.append(buf)

out, i, in_code = [], 0, False
while i < len(lines):
    ln = lines[i]

    if ln.strip().startswith('```'):
        in_code = not in_code
        i += 1
        continue
    if in_code:
        out.append('    ' + ln)
        i += 1
        continue

    if is_table_row(ln):
        block = []
        while i < len(lines) and is_table_row(lines[i]):
            if not is_sep_row(lines[i]):
                block.append(lines[i])
            i += 1
        out.append('')
        out.extend(render_table(block))
        out.append('')
        continue

    if ln.startswith('#'):
        lvl = len(ln) - len(ln.lstrip('#'))
        title = strip_inline(ln.lstrip('#').strip())
        out.append('')
        if lvl == 1:
            out.append('=' * WIDTH)
            out.append(title.upper())
            out.append('=' * WIDTH)
        elif lvl == 2:
            out.append('=' * WIDTH)
            out.append(title)
            out.append('=' * WIDTH)
        else:
            out.append(title)
            out.append('-' * len(title))
        out.append('')
        i += 1
        continue

    if ln.strip() in ('---', '***'):
        out.append('')
        out.append('.' * WIDTH)
        out.append('')
        i += 1
        continue

    if not ln.strip():
        out.append('')
        i += 1
        continue

    s = strip_inline(ln)
    m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', s)
    if m:
        pad, bullet, rest = m.groups()
        first = f"{pad}{'-' if bullet in '-*' else bullet} "
        wrapped = wrap(rest, '')
        out.append(first + wrapped[0])
        for extra in wrapped[1:]:
            out.append(' ' * len(first) + extra)
        i += 1
        continue

    if s.strip().startswith('>'):
        out.extend(wrap(s.strip().lstrip('> ').strip(), '  | '))
        i += 1
        continue

    out.extend(wrap(s.strip()))
    i += 1

# collapse >2 blank lines
txt, blank = [], 0
for ln in out:
    if ln.strip() == '':
        blank += 1
        if blank > 2:
            continue
    else:
        blank = 0
    txt.append(ln)

open(DST, 'w', encoding='utf-8').write('\n'.join(txt).strip() + '\n')
print(f"[+] Wrote {DST} ({len(txt)} lines)")
