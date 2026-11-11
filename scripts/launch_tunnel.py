#!/usr/bin/env python3
"""
scripts/launch_tunnel.py: One-Click Public Tunnel Launcher for Kaggle / Colab
=============================================================================
Launches the Streamlit app and exposes it via a high-performance, passwordless
Cloudflare Tunnel (trycloudflare.com) to completely eliminate dynamic JavaScript
chunk import errors (TypeError: Failed to fetch dynamically imported module)
caused by Localtunnel's interstitial landing page.

Usage:
  python scripts/launch_tunnel.py
"""

import os
import sys
import time
import shutil
import urllib.request
import subprocess
import re
from pathlib import Path

PORT = 8501
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
BINARY_NAME = "cloudflared"


def setup_cloudflared() -> str:
    """Verifies or downloads the cloudflared binary."""
    # Check if already in PATH
    which_path = shutil.which("cloudflared")
    if which_path:
        return which_path

    # Check local binary
    local_bin = Path(BINARY_NAME)
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin.resolve())

    # Download on Linux (Kaggle / Colab)
    if sys.platform.startswith("linux"):
        print("[*] Downloading official Cloudflare Tunnel binary (cloudflared)...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, BINARY_NAME)
        os.chmod(BINARY_NAME, 0o755)
        print("[✓] Cloudflared ready.")
        return str(local_bin.resolve())
    elif sys.platform.startswith("win"):
        win_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        win_bin = Path("cloudflared.exe")
        if not win_bin.is_file():
            print("[*] Downloading Cloudflare Tunnel for Windows...")
            urllib.request.urlretrieve(win_url, "cloudflared.exe")
        return str(win_bin.resolve())
    else:
        return BINARY_NAME


def main():
    print("=" * 70)
    print("STREAMLIT CLOUD TUNNEL LAUNCHER (KAGGLE / COLAB / REMOTE)")
    print("=" * 70)

    # 1. Start Streamlit
    print(f"[*] Starting Streamlit application on port {PORT}...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]
    st_proc = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3)

    # 2. Setup Tunnel
    try:
        bin_path = setup_cloudflared()
    except Exception as e:
        print(f"[!] Could not download cloudflared: {e}")
        print("[*] Falling back to local IP check...")
        try:
            import urllib.request
            ip = urllib.request.urlopen("https://ipv4.icanhazip.com", timeout=3).read().decode().strip()
            print(f"[*] Container Public IP: {ip}")
            print(f"[*] If using localtunnel, enter '{ip}' into the loca.lt reminder page.")
        except Exception:
            pass
        return

    # 3. Launch Cloudflare Tunnel
    print("[*] Establishing Cloudflare Tunnel (trycloudflare.com)...")
    tunnel_cmd = [bin_path, "tunnel", "--url", f"http://127.0.0.1:{PORT}"]
    proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    url_found = None
    url_regex = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")

    for line in iter(proc.stdout.readline, ""):
        match = url_regex.search(line)
        if match:
            url_found = match.group(0)
            break

    if url_found:
        print("\n" + "=" * 70)
        print("🚀 STREAMLIT APP IS LIVE ONLINE!")
        print("=" * 70)
        print(f"🔗 Public URL : {url_found}")
        print("=" * 70)
        print("[✓] Zero passwords required.")
        print("[✓] Zero interstitial warning blockers.")
        print("[✓] Full WebSocket and dynamic JavaScript chunk compatibility.")
        print("=" * 70)
    else:
        print("[!] Could not parse tunnel URL automatically.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down tunnel and Streamlit server...")
        proc.terminate()
        st_proc.terminate()


if __name__ == "__main__":
    main()

