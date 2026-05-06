
import subprocess
from pathlib import Path
from datetime import datetime

def run_cmd(cmd, cwd=None, show_output=True):
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if show_output:
        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)
    return result

def git_commit_push(project_dir, message):
    project_dir = Path(project_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_message = f"{message} | {timestamp}"

    run_cmd("git add .", cwd=project_dir)
    status = run_cmd("git status --short", cwd=project_dir, show_output=False)

    if not status.stdout.strip():
        print("No changes to commit.")
        return

    run_cmd(f'git commit -m "{final_message}"', cwd=project_dir)
    run_cmd("git push -u origin main", cwd=project_dir)
    print("GitHub sync completed.")
