import sys

with open('server/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

try:
    idx_449 = lines.index('<style>\n')
    idx_661 = lines.index('</html>\n', idx_449)
except ValueError:
    print('Indices not found.')
    sys.exit(1)

new_style_body = ''.join(lines[idx_449:idx_661+1])

prefix = '''DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medical Triage Commander</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
'''

suffix = '''"""

@app.get("/ui", response_class=HTMLResponse)
def get_dashboard():
    return DASHBOARD_HTML
'''

idx_58 = -1
for i, line in enumerate(lines):
    if line.startswith('DASHBOARD_HTML = '):
        idx_58 = i
        break

if idx_58 == -1:
    print('DASHBOARD_HTML not found')
    sys.exit(1)

final_content = ''.join(lines[:idx_58]) + prefix + new_style_body + suffix

with open('server/app.py', 'w', encoding='utf-8') as f:
    f.write(final_content)

print('File repaired!')
