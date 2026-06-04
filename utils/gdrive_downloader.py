"""
GDriveDownloader — downloads public Google Drive files and folders via the Drive API v3.

Why not gdown?
  gdown.download_folder() builds its file list by scraping Google Drive's HTML page.
  Google's folder view is a JavaScript SPA, so the raw HTML no longer contains the
  file listing — making gdown silently return nothing for every public folder,
  regardless of who owns it.

  The Drive API v3 returns stable, structured JSON and is the correct solution.

Setup (one-time):
  1. Go to https://console.cloud.google.com/apis/library/drive.googleapis.com
  2. Enable "Google Drive API" for the GCP project that owns your GOOGLE_API_KEY.
  That's it. No OAuth, no service account — just an API key for public resources.

Access model:
  GOOGLE_API_KEY authenticates *your application* for quota purposes.
  It can access ANY publicly shared Drive resource ("Anyone with the link → Viewer"),
  regardless of who owns the file or folder.

Only .pdf and .docx files are returned; all other types are silently skipped.
Raises PermissionError for private/restricted resources.
"""

import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

_DRIVE_API = "https://www.googleapis.com/drive/v3"

_SUPPORTED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_FOLDER_RE = re.compile(r"drive\.google\.com/drive/folders/([^/?&#]+)")
_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/?&#]+)")
_OPEN_RE = re.compile(r"[?&]id=([^&]+)")


class GDriveDownloader:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(self, api_key: str | None = None) -> None:
        # Walk candidate keys in priority order; skip any that aren't valid Google
        # API keys (which always start with "AIza") so a stale/wrong env var doesn't
        # shadow a correct one.
        candidates = [
            api_key,
            os.environ.get("GOOGLE_DRIVE_API_KEY"),
            os.environ.get("GOOGLE_API_KEY"),
        ]
        self._api_key = next(
            (k for k in candidates if k and k.startswith("AIza")),
            None,
        )
        if not self._api_key:
            raise RuntimeError(
                "No valid Google API key found. Set GOOGLE_API_KEY (or "
                "GOOGLE_DRIVE_API_KEY) to a key starting with 'AIza', "
                "or pass api_key= explicitly."
            )
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_folder_link(self, url: str) -> bool:
        return bool(_FOLDER_RE.search(url))

    def download(self, url: str, output_dir: str) -> list[str]:
        """
        Download supported files from a public Google Drive URL into output_dir.

        Returns absolute paths of the downloaded .pdf / .docx files.
        Raises PermissionError  – resource is private / requires sign-in.
        Raises FileNotFoundError – the file or folder no longer exists.
        """
        if self.is_folder_link(url):
            folder_id = _FOLDER_RE.search(url).group(1)
            return self._download_folder(folder_id, output_dir)

        file_id = self._extract_file_id(url)
        if not file_id:
            raise ValueError(f"Cannot extract a Google Drive ID from URL: {url}")
        return self._download_single(file_id, output_dir)

    # ------------------------------------------------------------------
    # Folder path
    # ------------------------------------------------------------------

    def _download_folder(self, folder_id: str, output_dir: str) -> list[str]:
        entries = self._list_folder(folder_id)
        downloaded: list[str] = []
        for entry in entries:
            if entry.get("mimeType") not in _SUPPORTED_MIMES:
                continue
            path = self._fetch_file(entry["id"], entry["name"], output_dir)
            downloaded.append(path)
        return downloaded

    def _list_folder(self, folder_id: str) -> list[dict]:
        """
        Return all direct children of a folder, handling pagination.
        One Drive API call per 1 000 files; most folders finish in one round-trip.
        """
        results: list[dict] = []
        page_token: str | None = None

        while True:
            params: dict = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "key": self._api_key,
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 1000,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = self._session.get(f"{_DRIVE_API}/files", params=params, timeout=30)
            self._check(resp, context=f"listing folder {folder_id!r}")

            body = resp.json()
            results.extend(body.get("files", []))

            page_token = body.get("nextPageToken")
            if not page_token:
                break

        return results

    # ------------------------------------------------------------------
    # Single-file path
    # ------------------------------------------------------------------

    def _download_single(self, file_id: str, output_dir: str) -> list[str]:
        """Fetch metadata, skip unsupported MIME types, then download."""
        resp = self._session.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"key": self._api_key, "fields": "name,mimeType"},
            timeout=30,
        )
        self._check(resp, context=f"fetching metadata for file {file_id!r}")

        meta = resp.json()
        if meta.get("mimeType") not in _SUPPORTED_MIMES:
            return []

        path = self._fetch_file(file_id, meta["name"], output_dir)
        return [path]

    # ------------------------------------------------------------------
    # Shared download helper
    # ------------------------------------------------------------------

    def _fetch_file(self, file_id: str, file_name: str, output_dir: str) -> str:
        """Stream a file to disk and return its absolute path."""
        resp = self._session.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"key": self._api_key, "alt": "media"},
            stream=True,
            timeout=120,
        )
        self._check(resp, context=f"downloading {file_name!r}")

        out_path = Path(output_dir) / file_name
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)

        return str(out_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_file_id(url: str) -> str | None:
        for pattern in (_FILE_RE, _OPEN_RE):
            m = pattern.search(url)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _check(resp: requests.Response, context: str) -> None:
        if resp.status_code == 200:
            return

        # Extract the actual error detail from the Drive API JSON body so the
        # caller sees a meaningful message instead of just "400 Bad Request".
        api_msg = ""
        api_reason = ""
        try:
            body = resp.json()
            err = body.get("error", {})
            api_msg = err.get("message", "")
            api_reason = (err.get("errors") or [{}])[0].get("reason", "")
            # Some errors include a direct activation URL — surface it if present
            for detail in err.get("details", []):
                for link in detail.get("links", []):
                    if "apis/api/drive" in link.get("url", ""):
                        api_msg += f"\nEnable the Drive API here: {link['url']}"
        except Exception:
            pass

        if resp.status_code == 400 and "API_KEY_INVALID" in str(body if api_msg else ""):
            raise RuntimeError(
                f"Invalid API key while {context}. "
                "Make sure GOOGLE_API_KEY (or GOOGLE_DRIVE_API_KEY) is a valid "
                f"Google API key (starts with 'AIza'). Detail: {api_msg}"
            )
        if resp.status_code in (400, 401, 403):
            if api_reason in ("accessNotConfigured", "SERVICE_DISABLED"):
                raise RuntimeError(
                    f"Google Drive API is not enabled for this project. "
                    f"{api_msg}"
                )
            if api_reason in ("forbidden", "insufficientPermissions"):
                raise PermissionError(
                    f"Drive resource is private / restricted while {context}. "
                    "Ensure the file or folder is shared as 'Anyone with the link → Viewer'."
                )
            raise RuntimeError(
                f"Drive API error ({resp.status_code}) while {context}: {api_msg or resp.text}"
            )
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"Drive resource not found while {context}. "
                "The file or folder may have been deleted or the ID is wrong."
            )
        resp.raise_for_status()
