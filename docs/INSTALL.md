# INSTALL.md — **talk**

A boringly‑reliable install for the `talk` CLI. This sets up a dedicated Python 3.11
virtual environment, installs the package in editable mode (with extras), links **only**
the `talk` command to that venv, and keeps your global Python untouched.

---

## Contents

- [Requirements](#requirements)
- [Quick Start (TL;DR)](#quick-start-tldr)
- [What This Setup Does](#what-this-setup-does)
- [Install via Makefile (recommended)](#install-via-makefile-recommended)
- [Verifying Your Install](#verifying-your-install)
- [Everyday Use](#everyday-use)
- [Virtual Environment Details](#virtual-environment-details)
- [When Dependencies Change (and how to handle it)](#when-dependencies-change-and-how-to-handle-it)
- [Uninstall / Cleanup](#uninstall--cleanup)
- [Troubleshooting](#troubleshooting)
- [Windows / WSL Notes](#windows--wsl-notes)
- [FAQ](#faq)
- [Reference: Makefile Targets](#reference-makefile-targets)
- [Environment Variables (optional)](#environment-variables-optional)

---

## Requirements

- **OS:** Linux or macOS (Windows via WSL recommended).
- **Python:** 3.11 preferred (project supports 3.9–3.11).
- **Tools:** `pip`, `venv`, `make`.
- **Repo layout:** project root at `~/code` (assumed below).
- **User bin:** `~/.local/bin` should be on your `PATH`.

Install the basics (examples):

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv make

# macOS (Homebrew)
brew install python@3.11
brew install make
```

If `~/.local/bin` is not on your `PATH`, add it:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

> **Why 3.11?** Faster runtime, great wheel availability (e.g., `lxml`), and widely packaged.  
> You can change the venv path/Python version in the Makefile if you need to.

---

## Quick Start (TL;DR)

```bash
cd ~/code
make dev      # creates ~/venvs/talk311, installs in editable mode, links only `talk`, verifies

talk --help   # should work anywhere
```

This:

- Creates a venv at `~/venvs/talk311`.
- Installs `talk` **editable** with extras `[all,dev]`.
- Symlinks `~/.local/bin/talk` → `~/venvs/talk311/bin/talk`.
- Prints entry‑point/module info and runs a smoke test.

---

## What This Setup Does

- Uses **`pyproject.toml`** (PEP 517/621) for modern packaging (no legacy `setup.py` needed).  
  If you previously had a `setup.py`, it can be kept as a hidden backup (e.g., `.setup.py.bak-*`).
- Installs core runtime deps:
  - `pydantic`, `pydantic-settings`, `typing-extensions`
  - `requests`, **`beautifulsoup4`** (imported as `bs4`), `lxml`
- Installs optional extras:
  - **LLM backends:** `openai`, `anthropic`, `google-generativeai` (grouped under `[all]`).
  - **Dev tools:** `pytest`, `black`, `ruff`, `mypy` (grouped under `[dev]`).
- Keeps your global Python alone; **only `talk`** runs inside the venv via a symlink.

---

## Install via Makefile (recommended)

From the project root (`~/code`):

```bash
make dev
```

What `make dev` actually does:

1. **Create/upgrade the venv**  
   Creates `~/venvs/talk311` (prefers `python3.11`; falls back to `python3`), then upgrades `pip`, `setuptools`, `wheel`, `build`.

2. **Editable install with extras**  
   One resolver pass for both extras (faster, fewer conflicts):
   ```bash
   pip install -e '.[all,dev]'
   ```

3. **Link only `talk`**  
   Creates/refreshes a symlink so that typing `talk` anywhere uses the venv:
   ```
   ~/.local/bin/talk  →  ~/venvs/talk311/bin/talk
   ```
   Everything else (`python`, other CLIs) continue using your global/system environment.

4. **Verification**  
   Prints the console entry point (`talk.talk:main`), module file path, and runs a smoke test (`talk -h`).

---

## Verifying Your Install

```bash
which talk
# → /home/you/.local/bin/talk

readlink -f ~/.local/bin/talk
# → /home/you/venvs/talk311/bin/talk

make -C ~/code doctor
# Shows interpreter used, entry point, and module file path
```

If `talk -h` exits 0, you’re good.

---

## Everyday Use

- From **any directory**:  
  ```bash
  talk --help
  talk --version
  ```

- From the **repo root** (handy wrappers):  
  ```bash
  make talk-help
  make run-talk ARGS="--version"
  ```

- Dev tasks (enabled by `[dev]` extras):  
  ```bash
  make test      # pytest
  make lint      # ruff
  make format    # black
  make typecheck # mypy
  make build     # sdist+wheel via `python -m build`
  ```

Because `talk` is installed **editable**, code changes are live—no reinstall required.

---

## Virtual Environment Details

- **Location:** `~/venvs/talk311` (configurable via `VENV_DIR` in the Makefile).
- **Isolation model:** we don’t “activate” the venv for your shell; instead we symlink the single CLI entry point (`talk`) into `~/.local/bin`. This keeps your global Python workflows untouched.
- **Upgrading Python:** if you later want a different Python, create a new venv and update the symlink:
  ```bash
  python3.12 -m venv ~/venvs/talk312
  ln -sfn ~/venvs/talk312/bin/talk ~/.local/bin/talk
  ```
  Re-run `make dev` to reinstall for the new interpreter.

---

## When Dependencies Change (and how to handle it)

You’ll know deps changed if **`pyproject.toml`** is updated (new packages, versions, or extras). To apply changes:

```bash
cd ~/code
make dev           # re-runs the single-pass editable install with [all,dev]
```

If you want to upgrade to **latest compatible** versions proactively:

```bash
# inside the venv (optional; make dev handles activation for commands it runs)
source ~/venvs/talk311/bin/activate
pip install -U -e '.[all,dev]'
```

If you want **strict, repeatable** installs, consider adding **pip-tools**:

```bash
pip install pip-tools
echo "-e .[all,dev]" > requirements.in
pip-compile --upgrade
pip-sync
```

> Tip: When adding a new **import** (e.g., `from bs4 import BeautifulSoup`), make sure the corresponding distribution name (`beautifulsoup4`) is listed under `dependencies` in `pyproject.toml`—the import name and package name can differ.

---

## Uninstall / Cleanup

```bash
# Remove the CLI symlink
rm -f ~/.local/bin/talk

# Remove the venv
rm -rf ~/venvs/talk311
```

To reinstall: `make dev` again.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'bs4'`**  
The import is `bs4` but the package is **`beautifulsoup4`**. Ensure it’s in `pyproject.toml`, then:
```bash
make dev   # or: pip install -e '.[all]'
```

**`RequestsDependencyWarning: urllib3 …`**  
Old system libs leaking into your environment. Inside the venv we install compatible versions, but you can force an update:
```bash
source ~/venvs/talk311/bin/activate
pip install -U 'urllib3<3' requests
```

**`talk` runs the wrong Python / not found**  
Make sure `~/.local/bin` is on your `PATH`, then refresh the symlink:
```bash
export PATH="$HOME/.local/bin:$PATH" && hash -r
which -a talk
ls -l ~/.local/bin/talk
readlink -f ~/.local/bin/talk
```

**GNU Make errors about “missing separator”**  
Recipe lines must start with a **tab**. The shipped Makefile uses real tabs and avoids heredocs. If you edited it, ensure your editor isn’t converting tabs to spaces.

**Cygwin/WSL quirks**  
Prefer **WSL** over Cygwin. We avoid heredocs in Make (to prevent odd interactions with commands named `import`, etc.).

---

## Windows / WSL Notes

- Use **WSL** (Ubuntu) for the smoothest experience.
- Follow the Linux instructions inside your WSL shell.
- Ensure `/home/<you>/.local/bin` is on your PATH in WSL (`~/.bashrc`).

---

## FAQ

**Why a dedicated venv just for `talk`?**  
It keeps your system Python clean and makes the CLI portable and reversible. Only the `talk` command uses the venv—nothing else on your system is affected.

**Can I keep using my global Python for everything else?**  
Yes. We don’t activate the venv globally; we only link the `talk` entry point into `~/.local/bin`.

**Can I move the venv or rename it?**  
Yes—edit `VENV_DIR` in the Makefile and re-run `make dev`.

**Do I still need `setup.py`?**  
No. We use `pyproject.toml`. If you had `setup.py`, rename it or keep it as a hidden backup to avoid confusing build tools.

---

## Reference: Makefile Targets

- `make dev` — venv → editable install **[all,dev]** → link only `talk` → verify  
- `make install` — editable install (core only)  
- `make install-all` — editable install with `[all]` extras  
- `make install-dev` — editable install with `[dev]` extras  
- `make link-talk` — refresh the symlink to the venv’s `talk`  
- `make doctor` — shows interpreter/entry point/module file & smoke test  
- `make test | lint | format | typecheck | build` — common dev tasks  
- `make clean | distclean` — remove caches/build artifacts

---

## Environment Variables (optional)

Set API keys as env vars, e.g.:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
```


If `talk` supports API keys for LLM backends, you can set them in the shell or an `.env` file:

```bash
# .env or shell
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

We include `python-dotenv` via the `[all]` extra so `.env` files can be loaded by the app (depending on your initialization code).


Make sure you have a C compiler (for `patch`) and that your `patch` command is on `PATH` (Linux/macOS already OK; Windows users can install [GnuWin32 patch](http://gnuwin32.sourceforge.net/packages/patch.htm) or use Git Bash).



---

Happy hacking. If something’s off, run:
```bash
make -C ~/code doctor
```
and paste the output into your issue/DM.
