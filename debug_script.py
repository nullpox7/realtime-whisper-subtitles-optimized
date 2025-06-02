#!/usr/bin/env python3
"""
Docker ??????????????
realtime-whisper-subtitles-optimized ???????
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, capture_output=True):
    """???????"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode, "", ""
    except Exception as e:
        return 1, "", str(e)

def check_container_status():
    """????????????"""
    print("=== ?????????? ===")
    
    # ????????
    ret, stdout, stderr = run_command("docker ps -a")
    if ret == 0:
        print("?????:")
        print(stdout)
    else:
        print(f"???: {stderr}")
    
    # ??????????
    container_name = "realtime-whisper-subtitles-optimized-whisper-subtitles-1"
    print(f"\n=== {container_name} ??? ===")
    
    ret, stdout, stderr = run_command(f"docker inspect {container_name}")
    if ret == 0:
        try:
            data = json.loads(stdout)
            container = data[0]
            state = container['State']
            config = container['Config']
            mounts = container['Mounts']
            
            print(f"??: {state['Status']}")
            print(f"?????: {state['RestartCount']}")
            print(f"?????: {state.get('ExitCode', 'N/A')}")
            print(f"???: {state.get('Error', 'None')}")
            
            print("\n????:")
            for env in config.get('Env', []):
                if 'LOG' in env or 'PATH' in env:
                    print(f"  {env}")
            
            print("\n????:")
            for mount in mounts:
                if 'logs' in mount['Destination']:
                    print(f"  {mount['Source']} -> {mount['Destination']}")
            
        except json.JSONDecodeError as e:
            print(f"JSON?????: {e}")
    else:
        print(f"???: {stderr}")

def check_logs():
    """???????"""
    print("\n=== ?????? ===")
    
    container_name = "realtime-whisper-subtitles-optimized-whisper-subtitles-1"
    
    # ?????
    print("????? (???20?):")
    ret, stdout, stderr = run_command(f"docker logs {container_name} --tail 20")
    if ret == 0:
        print(stdout)
        if stderr:
            print("STDERR:")
            print(stderr)
    else:
        print(f"???????: {stderr}")

def check_host_directories():
    """????????????????"""
    print("\n=== ?????????????? ===")
    
    # ???????????
    base_dirs = []
    
    # ???????????????
    possible_bases = [
        "./data",
        "../data", 
        "data",
        os.path.expanduser("~/realtime-whisper-subtitles-optimized/data"),
        os.path.expanduser("~/Codes/realtime-whisper-subtitles-optimized/data"),
    ]
    
    # Windows?????Windows??????
    if os.name == 'nt':
        possible_bases.extend([
            "C:\\Users\\kumo1\\Codes\\realtime-whisper-subtitles-optimized\\data",
            "C:\\realtime-whisper-subtitles-optimized\\data"
        ])
    
    # ????????????????
    detected_base = None
    for base in possible_bases:
        if os.path.exists(base):
            detected_base = base
            break
    
    if detected_base:
        print(f"??????????????: {detected_base}")
        subdirs = ["logs", "models", "outputs", "cache"]
        for subdir in subdirs:
            dir_path = os.path.join(detected_base, subdir)
            base_dirs.append(dir_path)
    else:
        print("??????????????????????????????????")
        base_dirs = ["./data/logs", "./data/models", "./data/outputs", "./data/cache"]
    
    for dir_path in base_dirs:
        print(f"\n?????: {dir_path}")
        if os.path.exists(dir_path):
            print("  ? ?????")
            # ??????
            try:
                files = os.listdir(dir_path)
                if files:
                    print(f"  ?????: {len(files)}")
                    for f in files[:5]:  # ???5????
                        print(f"    - {f}")
                    if len(files) > 5:
                        print(f"    ... ? {len(files) - 5} ????")
                else:
                    print("  ? ????????")
            except PermissionError:
                print("  ? ????????")
        else:
            print("  ? ??????")
            # ?????????
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ? ?????????????: {dir_path}")
            except Exception as e:
                print(f"  ? ??????????: {e}")

def fix_container():
    """??????????"""
    print("\n=== ???? ===")
    
    # 1. ???????????????????
    check_host_directories()
    
    # 2. ???????
    container_name = "realtime-whisper-subtitles-optimized-whisper-subtitles-1"
    print(f"\n????????: {container_name}")
    ret, stdout, stderr = run_command(f"docker stop {container_name}")
    if ret == 0:
        print("? ???????????")
    else:
        print(f"?? ????? (????????): {stderr}")
    
    # 3. docker-compose ????
    print("\ndocker-compose ?????...")
    
    # ?? docker-compose.yml ????????
    compose_files = ["docker-compose.yml", "../docker-compose.yml", "./docker-compose.yml"]
    compose_file = None
    for cf in compose_files:
        if os.path.exists(cf):
            compose_file = cf
            break
    
    if compose_file:
        print(f"???? compose ????: {compose_file}")
        ret, stdout, stderr = run_command("docker-compose restart whisper-subtitles")
        if ret == 0:
            print("? ????????????")
            print(stdout)
        else:
            print(f"? ??????: {stderr}")
            print("??????????...")
            ret2, stdout2, stderr2 = run_command(f"docker start {container_name}")
            if ret2 == 0:
                print("? ??????????????????")
            else:
                print(f"? ?????????: {stderr2}")
    else:
        print("?? docker-compose.yml ?????????????????????")
        ret, stdout, stderr = run_command(f"docker start {container_name}")
        if ret == 0:
            print("? ???????????")
        else:
            print(f"? ?????: {stderr}")

def create_fixed_dockerfile():
    """?????Dockerfile???"""
    print("\n=== ?????Dockerfile?? ===")
    
    dockerfile_content = '''FROM python:3.11-slim

# ????????????????
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    curl \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# ???????????
WORKDIR /app

# ????????????????
RUN useradd -m -u 1000 appuser

# ????????????????????????????
RUN mkdir -p /app/data/models /app/data/outputs /app/data/logs /app/data/cache /app/static /app/src /app/templates
RUN chown -R appuser:appuser /app

# ???????????
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ???????????????
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/

# ?????
RUN chown -R appuser:appuser /app

# ??????????????????
USER appuser

# ????
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_PATH=/app/data/logs

# ???????
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# ?????
EXPOSE 8000

# ??????
CMD ["python3", "-m", "uvicorn", "src.web_interface:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
'''
    
    with open("Dockerfile.fixed", "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    
    print("? ?????Dockerfile.fixed???????")
    print("???????????????????????:")
    print("docker build -f Dockerfile.fixed -t realtime-whisper-subtitles-optimized-fixed .")

def show_fix_instructions():
    """???????"""
    print("\n=== ???? ===")
    print("1. ???????:")
    print("   python debug_script.py fix")
    print()
    print("2. ???????:")
    print("   a) ????????????:")
    print("      python debug_script.py dirs")
    print("   b) ????????:")
    print("      docker-compose restart whisper-subtitles")
    print("   c) ?????:")
    print("      python debug_script.py logs")
    print()
    print("3. ?????????????:")
    print("   a) ???Dockerfile???:")
    print("      python debug_script.py dockerfile") 
    print("   b) ???????????:")
    print("      docker build -f Dockerfile.fixed -t whisper-fixed .")
    print("   c) docker-compose.yml?????????????")

def main():
    """?????"""
    print("Real-time Whisper Subtitles Optimized - ???????")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "status":
            check_container_status()
        elif action == "logs":
            check_logs()
        elif action == "dirs":
            check_host_directories()
        elif action == "fix":
            fix_container()
        elif action == "dockerfile":
            create_fixed_dockerfile()
        elif action == "help":
            show_fix_instructions()
        else:
            print(f"????????: {action}")
            print("??????????: status, logs, dirs, fix, dockerfile, help")
    else:
        # ????????
        check_container_status()
        check_logs()
        check_host_directories()
        
        # ?????
        show_fix_instructions()

if __name__ == "__main__":
    main()
