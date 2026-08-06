#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    args = parser.parse_args()

    deadline = time.time() + args.timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                args.url,
                timeout=3,
            ) as response:
                value = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

            if value.get("status") == "ok":
                print(
                    "Local LLM health check: PASSED"
                )
                print(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                )
                return

        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc

        time.sleep(2)

    raise SystemExit(
        "Local LLM did not become ready. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    main()
