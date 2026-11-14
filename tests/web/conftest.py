"""Fixtures for the browser tests of the game in web/ (Playwright for Python).

Global context: the game is a static site, so each test session serves a folder with the standard
library's http.server on a free port (no external server needed), and the tests drive the page in
headless Chromium like a player would. Run with:  uv run pytest tests/web -m web
Docs: https://playwright.dev/python/docs/test-runners (pytest-playwright fixtures: page, browser).
"""

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # repository root
WEB = ROOT / "web"  # the site that GitHub Pages publishes


class QuietHandler(SimpleHTTPRequestHandler):
    """Static file handler that does not print one log line per request."""

    def log_message(self, format, *args):  # noqa: A002 (signature of the base class)
        pass


@pytest.fixture(scope="session")
def serve():
    """Start an http.server for a directory on a free port; returns its base URL. Stopped at session end."""
    servers = []

    def _serve(directory: Path) -> str:
        handler = functools.partial(QuietHandler, directory=str(directory))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)  # port 0: the OS picks a free port
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield _serve
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(scope="session")
def site_url(serve):
    """The game served at the server root, like the local dev server (http://localhost:8123/)."""
    return serve(WEB) + "/"


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    """Headless Chromium renders WebGL with SwiftShader; opt in explicitly so Chromium does not log a
    deprecation warning about the automatic software fallback."""
    return {**browser_type_launch_args, "args": ["--enable-unsafe-swiftshader"]}


@pytest.fixture
def console(page):
    """Collect console errors/warnings and uncaught page errors of the test's page.

    Chromium's own "GL Driver Message ... GPU stall due to ReadPixels" performance notes are skipped:
    headless SwiftShader emits them when it presents the WebGL canvas, not because of page code
    (they do not appear in a GPU-backed browser).
    """
    messages = []

    def on_console(m):
        if m.type in ("error", "warning") and "GL Driver Message" not in m.text:
            messages.append(f"{m.type}: {m.text}")

    page.on("console", on_console)
    page.on("pageerror", lambda e: messages.append(f"pageerror: {e}"))
    return messages
