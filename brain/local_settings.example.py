"""复制成 local_settings.py 再改：这台电脑自己的设置，覆盖 config.py 里的同名值（local_settings.py 不进 git）。"""
from pathlib import Path

# 出门在外连回家的中转服务器（relay.py）：SSH 登录名和私钥文件。不填就不开隧道，只能在局域网里用
RELAY_HOST = "ubuntu@your.server.ip"
RELAY_KEY = Path.home() / ".ssh" / "your_server_key.pem"
