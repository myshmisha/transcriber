#!/usr/bin/env python

"""Utilities for managing a persistent YouTube login on headless hosts.

The script keeps a Chromium profile up to date using Playwright so that
yt-dlp can reuse fresh cookies without manual exports. It can:

  * launch an automated Google/YouTube login flow (headless automation)
  * export the current storage state as a yt-dlp compatible cookies file
  * show profile status and clear stale state when troubleshooting

Usage examples:

  python scripts/youtube_profile.py login --email me@example.com
  python scripts/youtube_profile.py export-cookies
  python scripts/youtube_profile.py status
  python scripts/youtube_profile.py clear

Sensitive inputs (password, OTP) are requested interactively so they do not
appear in shell history. This automation expects a standard Google login
experience; additional challenges (phone prompts, CAPTCHA) must be resolved
manually using the Google account security options.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional


try:
    from playwright.sync_api import (  # type: ignore
        Error as PlaywrightError,
        TimeoutError as PlaywrightTimeout,
        sync_playwright,
    )
except Exception:  # pragma: no cover - handled at runtime
    sync_playwright = None  # type: ignore
    PlaywrightError = PlaywrightTimeout = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent.parent
PROFILE_DIR = Path(os.environ.get("YT_BROWSER_PROFILE", BASE_DIR / "youtube_profile"))
COOKIE_FILE = Path(os.environ.get("YT_COOKIE_FILE", BASE_DIR / "yt_cookies.txt"))
YT_LOGIN_URL = "https://accounts.google.com/signin/v2/identifier?service=youtube"
CHALLENGE_SELECTORS: Iterable[str] = (
    "input[type='tel']",
    "input[type='number']",
    "input[name='totpPin']",
    "input[aria-label*='code']",
)


def _ensure_playwright() -> None:
    if sync_playwright is None:
        print(
            "Playwright is not installed. Run `pip install playwright` and `playwright install chromium`.",
            file=sys.stderr,
        )
        sys.exit(1)


def _launch_context(headless: bool = True):
    _ensure_playwright()

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    playwright = sync_playwright()
    p = playwright.__enter__()
    try:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=headless,
            viewport={"width": 1280, "height": 720},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-default-browser-check",
                "--no-first-run",
                "--no-sandbox",
                "--disable-gpu",
            ],
        )
    except PlaywrightError as error:  # type: ignore
        playwright.__exit__(type(error), error, error.__traceback__)
        raise

    return playwright, context


def _wait_for_selector(page, selector: str, timeout: int = 20000):
    return page.wait_for_selector(selector, state="visible", timeout=timeout)


def _click_if_present(page, selector: str, timeout: int = 5000) -> bool:
    locator = page.locator(selector).first
    try:
        locator.wait_for(state="visible", timeout=timeout)
        locator.click()
        return True
    except PlaywrightTimeout:  # type: ignore
        return False


def _fill_and_submit(page, selector: str, value: str, submit_selector: Optional[str] = None):
    _wait_for_selector(page, selector)
    page.fill(selector, value)
    if submit_selector:
        submit_clicked = False
        for candidate in (f"{submit_selector} button", submit_selector, "button[type='submit']"):
            if _click_if_present(page, candidate, timeout=8000):
                submit_clicked = True
                break
        if not submit_clicked:
            page.keyboard.press("Enter")
    else:
        page.keyboard.press("Enter")


def _handle_challenge(page, initial_code: Optional[str] = None) -> bool:
    code = initial_code
    for selector in CHALLENGE_SELECTORS:
        locator = page.locator(selector).first
        try:
            if locator.is_visible(timeout=1000):
                if not code:
                    code = input("Enter Google verification code: ").strip()
                locator.fill(code)
                page.keyboard.press("Enter")
                return True
        except PlaywrightTimeout:  # type: ignore
            continue
    return False


def command_login(email: Optional[str], password: Optional[str], otp: Optional[str], headless: bool) -> None:
    email = email or input("Google account email: ").strip()
    if not email:
        print("Email is required.")
        sys.exit(1)

    password = password or getpass.getpass("Password: ")
    if not password:
        print("Password is required.")
        sys.exit(1)

    playwright, context = _launch_context(headless=headless)
    page = context.new_page()
    try:
        page.goto(YT_LOGIN_URL, wait_until="domcontentloaded")

        if _click_if_present(page, "button:has-text('Accept all')") or _click_if_present(
            page, "button:has-text('I agree')"
        ):
            page.wait_for_timeout(500)

        _fill_and_submit(page, "input[type='email']", email, "#identifierNext")

        try:
            _wait_for_selector(page, "input[type='password']:not([aria-hidden='true'])")
        except PlaywrightTimeout:  # type: ignore
            if _handle_challenge(page, otp):
                _wait_for_selector(page, "input[type='password']:not([aria-hidden='true'])")
            else:
                raise

        _fill_and_submit(page, "input[type='password']:not([aria-hidden='true'])", password, "#passwordNext")

        if _handle_challenge(page, otp):
            page.wait_for_timeout(1000)

        page.wait_for_url("https://www.youtube.com/*", timeout=60000)
        context.storage_state(path=str(COOKIE_FILE))
        print(f"Login successful. Cookies exported to {COOKIE_FILE}.")
    except PlaywrightTimeout as error:  # type: ignore
        print("Timed out during login flow:", error, file=sys.stderr)
        sys.exit(1)
    except PlaywrightError as error:  # type: ignore
        print("Playwright error:", error, file=sys.stderr)
        sys.exit(1)
    finally:
        context.close()
        playwright.__exit__(None, None, None)


def command_export(headless: bool) -> None:
    playwright, context = _launch_context(headless=headless)
    try:
        context.storage_state(path=str(COOKIE_FILE))
        print(f"Cookies exported to {COOKIE_FILE}.")
    finally:
        context.close()
        playwright.__exit__(None, None, None)


def command_status() -> None:
    exists = PROFILE_DIR.exists()
    cookies = COOKIE_FILE.exists()
    print(json.dumps({
        "profile_dir": str(PROFILE_DIR),
        "profile_exists": exists,
        "cookie_file": str(COOKIE_FILE),
        "cookie_exists": cookies,
        "profile_size_bytes": _dir_size(PROFILE_DIR) if exists else 0,
        "cookie_mtime": _mtime(COOKIE_FILE) if cookies else None,
    }, indent=2))


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def _mtime(path: Path) -> Optional[float]:
    return path.stat().st_mtime if path.exists() else None


def command_clear(confirm: bool) -> None:
    if not confirm:
        answer = input(f"Delete {PROFILE_DIR} and {COOKIE_FILE}? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    if PROFILE_DIR.exists():
        shutil.rmtree(PROFILE_DIR, ignore_errors=True)
    if COOKIE_FILE.exists():
        COOKIE_FILE.unlink()
    print("Profile and cookies removed.")


def command_refresh(otp: Optional[str], headless: bool) -> None:
    playwright, context = _launch_context(headless=headless)
    page = context.new_page()
    try:
        page.goto("https://www.youtube.com/feed/library", wait_until="domcontentloaded")
        if "ServiceLogin" in page.url:
            print("Session requires re-login. Run the login command.", file=sys.stderr)
            sys.exit(1)
        context.storage_state(path=str(COOKIE_FILE))
        print(f"Session refreshed. Cookies stored in {COOKIE_FILE}.")
    finally:
        context.close()
        playwright.__exit__(None, None, None)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chromium with a visible window (requires GUI/X11).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    login = subparsers.add_parser("login", help="Perform a headless login and export cookies.")
    login.add_argument("--email", help="Google account email.")
    login.add_argument("--password", help="Google account password.")
    login.add_argument("--otp", help="Optional OTP code if already available.")

    export = subparsers.add_parser("export-cookies", help="Export cookies using the existing profile.")

    status = subparsers.add_parser("status", help="Show current profile and cookie status.")

    clear = subparsers.add_parser("clear", help="Delete the profile directory and cookie file.")
    clear.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")

    refresh = subparsers.add_parser("refresh", help="Touch the YouTube session and export cookies.")
    refresh.add_argument("--otp", help="Optional OTP code if a challenge appears.")

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    headless = not args.headed

    if args.command == "login":
        command_login(args.email, args.password, args.otp, headless=headless)
    elif args.command == "export-cookies":
        command_export(headless=headless)
    elif args.command == "status":
        command_status()
    elif args.command == "clear":
        command_clear(args.yes)
    elif args.command == "refresh":
        command_refresh(args.otp, headless=headless)
    else:  # pragma: no cover - defensive
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
