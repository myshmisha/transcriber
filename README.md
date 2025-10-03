# transcriber

## Headless YouTube authentication

The app now ships with helper utilities that keep a persistent Chromium
profile alive on headless servers so `yt-dlp` always has fresh cookies.

1. Install Playwright (once per Python environment):

   ```bash
   pip install playwright
   playwright install chromium
   ```

2. Run the login helper and follow the prompts:

   ```bash
   python scripts/youtube_profile.py login --email you@example.com

   Add `--headed` if you want to see the Chromium window for debugging on a
   machine with GUI support.
   ```

   The script launches Chromium in headless automation mode, signs in to
   Google/YouTube, and writes cookies to `yt_cookies.txt` plus a persistent
   profile under `youtube_profile/`.

3. Check status or refresh cookies at any time:

   ```bash
   python scripts/youtube_profile.py status
   python scripts/youtube_profile.py refresh
   ```

4. `transcriber.py` automatically loads cookies from `yt_cookies.txt` or, if
   available, reuses the Playwright-managed profile. You can override the
   defaults with `YT_COOKIE_FILE`, `YT_BROWSER`, and `YT_BROWSER_PROFILE`
   environment variables.
