import pathlib
for path in pathlib.Path('src/ai_researcher').rglob('*.py'):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    import re
    # Strip the specific rule from type: ignore
    new_content = re.sub(r'# type: ignore\[.*?\]', '# type: ignore', content)
    if content != new_content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
