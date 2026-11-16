import json

def patch_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 4
    c4 = nb['cells'][4]['source']
    new_c4 = []
    for line in c4:
        if line.startswith('import os'):
            new_c4.append('import time as _stdlib_time\n')
            new_c4.append('time = _stdlib_time\n')
            new_c4.append(line)
        elif 'start_at = time.time() + 2.0' in line:
            new_c4.append('                start_at = _stdlib_time.time() + 2.0\n')
        elif 'httpd = None' in line:
            new_c4.append(line)
            new_c4.append('\n')
            new_c4.append('if "httpd" in globals() and httpd is not None:\n')
            new_c4.append('    try:\n')
            new_c4.append('        httpd.shutdown()\n')
            new_c4.append('        httpd.server_close()\n')
            new_c4.append('    except Exception:\n')
            new_c4.append('        pass\n')
            new_c4.append('    _stdlib_time.sleep(0.3)\n')
        else:
            new_c4.append(line)
    nb['cells'][4]['source'] = new_c4

    # 2. Update Cell 6
    c6 = nb['cells'][6]['source']
    new_c6 = []
    for line in c6:
        if line == 'import time\n':
            new_c6.append('import urllib.error\n')
            new_c6.append('import time as _stdlib_time\n')
            new_c6.append('time = _stdlib_time\n')
        elif 'except Exception as exc:\n' in line and len(new_c6) > 0 and 'sync_response = json.loads' in new_c6[-1]:
            new_c6.append('except urllib.error.HTTPError as exc:\n')
            new_c6.append('    err_body = exc.read().decode("utf-8", errors="replace")\n')
            new_c6.append('    try:\n')
            new_c6.append('        err_json = json.loads(err_body)\n')
            new_c6.append('        err_msg = err_json.get("message", err_body)\n')
            new_c6.append('    except Exception:\n')
            new_c6.append('        err_msg = err_body\n')
            new_c6.append('    raise RuntimeError(f"iSync server rejected visual start (HTTP {exc.code}): {err_msg}")\n')
            new_c6.append('except Exception as exc:\n')
        elif 'remaining = start_at - time.time()' in line:
            new_c6.append('    remaining = start_at - _stdlib_time.time()\n')
        else:
            new_c6.append(line)
    nb['cells'][6]['source'] = new_c6

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'Updated {path} successfully!')

if __name__ == '__main__':
    patch_notebook('Demo/EOG_optimized.ipynb')
