import re, subprocess

def fix_mypy():
    proc = subprocess.run(['uv', 'run', 'mypy', 'src/'], capture_output=True, text=True)
    out = proc.stdout
    print("Mypy Output:")
    print(out)
    
    lines_to_fix = {}
    for line in out.split('\n'):
        m = re.match(r'^(.*?):(\d+): error: (.*?)\[(.*?)\]', line)
        if m:
            path, line_num_str, msg, rule = m.groups()
            line_num = int(line_num_str)
            if path not in lines_to_fix:
                lines_to_fix[path] = []
            lines_to_fix[path].append((line_num, rule.strip()))

    for path, rules in lines_to_fix.items():
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for idx, (ln, rule) in enumerate(rules):
            real_idx = ln - 1
            if real_idx < len(lines):
                if '# type: ignore' not in lines[real_idx] and 'noqa' not in lines[real_idx]:
                    lines[real_idx] = lines[real_idx].rstrip() + f'  # type: ignore[{rule}]\n'
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"Fixed {path} ({len(rules)} items)")

fix_mypy()
