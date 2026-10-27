"""出门在外连回家：Lightsail 服务器中转 + SSH 反向隧道 + 设备密钥。

    板子（手机热点）──wss──▶ pet.your-domain.example（服务器上的 Nginx：https 证书，只转发 /ws）
                                  ▼ 服务器的 127.0.0.1:18600
                            SSH 反向隧道（这台电脑主动连出去，家里路由器不用开端口）
                                  ▼ 这台电脑的 127.0.0.2:8600

隧道故意接到 127.0.0.2 而不是 127.0.0.1：大脑看连接是从哪个本机地址进来的（scope["server"]），
就知道它是不是从公网绕进来的——这个没法靠伪造请求头骗过去（服务器上别的网站被黑了也一样）。
从隧道进来的：只能连 /ws，而且 hello 里要带设备密钥；记忆管理、导出备份这些 HTTP 接口一律拒绝。
"""
import hmac
import logging
import secrets
import subprocess
import threading
import time

import config

log = logging.getLogger("pet.relay")
TOKEN_FILE = config.DATA_DIR / "device_token.txt"


def via_relay(scope) -> bool:
    """这个连接是不是从隧道（公网）进来的。"""
    server = scope.get("server") or ("", 0)
    return server[0] == config.RELAY_LOCAL_HOST


def device_token() -> str:
    """设备密钥：第一次用时随机生成，存在 data/device_token.txt（烧进固件的也是它）。"""
    if not TOKEN_FILE.exists():
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    return TOKEN_FILE.read_text(encoding="utf-8-sig").strip()


def token_ok(token) -> bool:
    return isinstance(token, str) and hmac.compare_digest(token.encode(), device_token().encode())


def enabled() -> bool:
    return bool(config.RELAY_HOST) and config.RELAY_KEY is not None and config.RELAY_KEY.exists()


class Tunnel:
    """维持一条 SSH 反向隧道：断了就重连（等待时间从 5 秒慢慢加到 60 秒，连上稳定一会儿就重置）。"""

    def __init__(self):
        self.proc = None
        self.stopping = False

    def command(self):
        return ["ssh", "-N", "-i", str(config.RELAY_KEY),
                "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
                "-o", "StrictHostKeyChecking=accept-new",
                "-R", f"127.0.0.1:{config.RELAY_REMOTE_PORT}:{config.RELAY_LOCAL_HOST}:{config.PORT}",
                config.RELAY_HOST]

    def run(self):
        wait = 5
        while not self.stopping:
            started = time.monotonic()
            try:
                self.proc = subprocess.Popen(self.command(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                log.info("隧道已连：%s 的 %d 端口 → 本机 %s:%d", config.RELAY_HOST, config.RELAY_REMOTE_PORT,
                         config.RELAY_LOCAL_HOST, config.PORT)
                _, err = self.proc.communicate()
                if not self.stopping:
                    log.warning("隧道断了：%s", (err or b"").decode(errors="replace").strip()[-200:])
            except FileNotFoundError:
                log.error("找不到 ssh 命令（Windows 的“可选功能”里装 OpenSSH 客户端）")
                return
            if time.monotonic() - started > 60:
                wait = 5
            for _ in range(int(wait * 10)):
                if self.stopping:
                    return
                time.sleep(0.1)
            wait = min(wait * 2, 60)

    def start(self):
        if not enabled():
            log.info("没配中转服务器（PET_RELAY_HOST / PET_RELAY_KEY），出门连不回家")
            return
        threading.Thread(target=self.run, daemon=True, name="relay-tunnel").start()

    def stop(self):
        self.stopping = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


tunnel = Tunnel()
