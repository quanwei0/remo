"""Release hygiene: our files must not reference the authors' machine or unrelated local projects, and must not
mention the codename of a project this implementation does not depend on."""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "third_party", "remo.egg-info", "__pycache__", "runs", "data"}
TEXT_EXT = {".py", ".sh", ".md", ".toml", ".txt", ".json", ".jsonnet", ".cfg", ".yaml", ".yml"}
FORBIDDEN = [
    (re.compile(r"/mnt/home|/mnt/data|quan\.wei|financegym_proj|ace_proj|\.conda/envs|gr-financegym"), "local machine path"),
    (re.compile(r"\bace\b", re.I), "forbidden codename"),
]


def our_text_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if os.path.splitext(f)[1] in TEXT_EXT:
                yield os.path.join(dirpath, f)


class TestRepoHygiene(unittest.TestCase):
    def test_no_local_paths_or_forbidden_names(self):
        hits = []
        for p in our_text_files():
            if p == os.path.abspath(__file__):
                continue
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                for pat, why in FORBIDDEN:
                    if pat.search(line):
                        hits.append(f"{os.path.relpath(p, ROOT)}:{i}: {why}: {line.strip()[:100]}")
        self.assertEqual(hits, [], "\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
