"""Deployment invariants for the container build.

Docker is not installed on the development machine, so nothing here is built
locally - CI builds both images on every push, and does so successfully. These
tests are the fast feedback in between: they check everything that *can* be
checked without a daemon, that the build would find the files it copies, that the
dependency extras it installs exist, that nothing the build needs is excluded by
.dockerignore, and that the compose file's mounts and entrypoint agree with how
the image is actually laid out.

That is not a substitute for `docker build`. It is the part of the verification
that does not need one, and it catches the failures that would otherwise only
appear on the machine of whoever first tries to deploy this. The rest - what
only a *running* container shows - is `scripts/check_containers.sh`, which CI
runs after the builds; the last section here is what keeps the two in step.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
DOCKERIGNORE = ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def load_script(name: str):
    """Import a script by path: `scripts/` is entry points, not a package."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copied_sources(dockerfile: str) -> list[str]:
    """Every source path a COPY instruction reads from the build context.

    `COPY --from=<stage>` reads from an earlier build stage rather than the
    context, so those lines are skipped: their paths exist only inside the
    image and cannot be checked against the repository.
    """
    sources = []
    for line in dockerfile.splitlines():
        if not line.strip().upper().startswith("COPY"):
            continue
        if "--from=" in line:
            continue
        parts = [p for p in line.split()[1:] if not p.startswith("--")]
        sources.extend(parts[:-1])  # the last argument is the destination
    return sources


# ---------------------------------------------------------------------------
# Build context
# ---------------------------------------------------------------------------


def test_every_copied_path_exists(dockerfile):
    """A COPY of a missing path fails the build, and only at build time."""
    for source in copied_sources(dockerfile):
        assert (ROOT / source).exists(), f"Dockerfile COPYs missing path {source!r}"


def test_reading_the_monitoring_record_is_a_declared_dependency():
    """Regression: openpyxl was used and never declared.

    `loader.load_workbook` reads .xlsx through pandas, which needs openpyxl. It
    was installed on the development machine as somebody else's transitive
    dependency, so every loader test passed locally and every one of them failed
    on a clean machine - eight CI runs in a row, unread.
    """
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = " ".join(declared["project"]["dependencies"])
    assert "openpyxl" in required, (
        "pandas cannot read the .xlsx monitoring record without it"
    )


def test_installed_extras_are_declared(dockerfile):
    """`pip install .[ml,api]` must name extras that pyproject actually has."""
    match = re.search(r'pip install "\.\[([^\]]+)\]"', dockerfile)
    assert match, "Dockerfile no longer installs the package with extras"

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    available = set(declared["project"]["optional-dependencies"])
    requested = {name.strip() for name in match.group(1).split(",")}
    assert requested <= available, f"undeclared extras: {sorted(requested - available)}"


def test_dockerignore_keeps_what_the_build_needs(dockerfile):
    """.dockerignore excludes *.md, and the build needs README.md for the install.

    setuptools reads `readme = "README.md"` from pyproject; if the file is not in
    the context the build fails on a metadata error that names neither.
    """
    ignore = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
    assert "*.md" in ignore
    assert "!README.md" in ignore, ".dockerignore drops the README the build needs"
    assert "README.md" in copied_sources(dockerfile)


def test_healthcheck_dependency_is_installed(dockerfile):
    """The HEALTHCHECK imports httpx, which only exists via the api extra."""
    assert "httpx" in dockerfile
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    api_extra = " ".join(declared["project"]["optional-dependencies"]["api"])
    assert "httpx" in api_extra


def test_healthcheck_targets_a_real_endpoint(dockerfile):
    """The health path in the HEALTHCHECK must be one the app serves."""
    from aquanexus.api.app import app

    match = re.search(r"http://localhost:8000(/\w+)", dockerfile)
    assert match, "HEALTHCHECK no longer probes an HTTP path"
    # The OpenAPI schema rather than app.routes: included routers are wrapped
    # objects in current FastAPI, and the schema is what is actually served.
    assert match.group(1) in app.openapi()["paths"]


def test_entrypoint_module_is_importable(dockerfile):
    """The CMD names an import path; a typo there fails only on `docker run`."""
    match = re.search(r'"uvicorn", "([\w.]+):(\w+)"', dockerfile)
    assert match, "Dockerfile CMD no longer starts uvicorn with a module path"

    module = __import__(match.group(1), fromlist=[match.group(2)])
    assert hasattr(module, match.group(2))


def test_image_runs_as_a_non_root_user(dockerfile):
    assert re.search(r"^USER\s+(?!root)", dockerfile, re.MULTILINE)


def test_image_pins_the_data_directory(dockerfile):
    """Regression: an installed package looks for data beside its own code.

    `Settings.ROOT_DIR` is derived from the source file location. In a checkout
    that is the repository; in the image the package lives in site-packages, so
    the default data path becomes something like
    /usr/local/lib/python3.12/data - and mounting the host's ./data at /app/data
    changes nothing. The API then starts degraded with a correct-looking mount
    in place, which is the worst kind of deployment failure: silent and
    plausible.
    """
    assert re.search(r"^ENV DATA_DIR=/app/data", dockerfile, re.MULTILINE), (
        "the image must pin DATA_DIR; the default resolves into site-packages"
    )


def test_data_dir_override_moves_the_subdirectories(tmp_path, monkeypatch):
    """Setting DATA_DIR alone must be enough - the fix the Dockerfile relies on."""
    from aquanexus.config import Settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = Settings()
    assert tmp_path == settings.DATA_DIR
    assert tmp_path / "models" == settings.MODELS_DIR
    assert tmp_path / "raw" == settings.RAW_DIR
    assert tmp_path / "processed" == settings.PROCESSED_DIR


def test_explicit_subdirectory_still_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "elsewhere"))
    from aquanexus.config import Settings

    assert tmp_path / "elsewhere" == Settings().MODELS_DIR


# ---------------------------------------------------------------------------
# Frontend image
# ---------------------------------------------------------------------------

FRONTEND = ROOT / "frontend"


@pytest.fixture(scope="module")
def frontend_dockerfile() -> str:
    return (FRONTEND / "Dockerfile").read_text(encoding="utf-8")


def test_frontend_build_copies_what_vite_needs(frontend_dockerfile):
    for source in copied_sources(frontend_dockerfile):
        # COPY takes globs and writes paths as ./src/ or tsconfig*.json, so
        # normalise before resolving rather than stat-ing the literal string.
        pattern = source.removeprefix("./").rstrip("/")
        assert list(FRONTEND.glob(pattern)), f"frontend Dockerfile COPYs {source!r}"


def test_frontend_serves_the_single_page_router():
    """A client route is not a file; without a fallback, reloading /predict 404s."""
    config = (FRONTEND / "nginx.conf").read_text(encoding="utf-8")
    assert "try_files" in config and "/index.html" in config


def test_frontend_api_url_is_set_at_run_time(frontend_dockerfile):
    """Regression: the API address used to be baked into the bundle.

    Vite substitutes import.meta.env at build time, so an image built against
    one backend could never be repointed - every environment needed its own
    build. The entrypoint writes /config.js from $API_BASE_URL at start-up
    instead, and the page reads it at load.
    """
    instructions = [line for line in frontend_dockerfile.splitlines()
                    if not line.lstrip().startswith("#")]
    assert not any("VITE_API_BASE_URL" in line for line in instructions), (
        "baking the URL in defeats the runtime config"
    )
    assert re.search(r"^ENTRYPOINT", frontend_dockerfile, re.MULTILINE)

    entrypoint = (FRONTEND / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "API_BASE_URL" in entrypoint
    assert "config.js" in entrypoint
    assert entrypoint.rstrip().endswith('exec "$@"'), (
        "the entrypoint must hand over to the image's CMD"
    )


def test_runtime_config_is_never_cached():
    """A cached config.js points the page at the previous deployment's API."""
    config = (FRONTEND / "nginx.conf").read_text(encoding="utf-8")
    assert re.search(r"location = /config\.js", config)
    assert "no-store" in config


def test_the_page_loads_the_runtime_config_before_the_bundle():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    config_at = html.find("/config.js")
    bundle_at = html.find("/src/main.tsx")
    assert config_at != -1, "index.html does not load the runtime config"
    assert config_at < bundle_at, "config.js must load before the bundle reads it"


def test_compose_points_the_frontend_at_a_host_reachable_api(compose):
    """The browser resolves that URL, not the compose network.

    `http://api:8000` works between containers and fails in the browser, which
    is the sort of thing that only shows up after a deploy.
    """
    settings = "\n".join(
        line for line in compose.splitlines() if not line.strip().startswith("#")
    )
    assert "API_BASE_URL" in settings
    assert "http://api:8000" not in settings


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def test_compose_mounts_paths_that_exist(compose):
    for host_path in re.findall(r"- \.(/[\w/]+):", compose):
        assert (ROOT / host_path.lstrip("/")).exists(), f"compose mounts missing {host_path}"


def test_compose_makes_the_source_mount_effective(compose):
    """Regression: mounting ./src over an image that installed the package.

    The image runs `pip install .` (not editable), so `import aquanexus` resolves
    to site-packages. Mounting ./src:/app/src therefore changes nothing on its
    own - and --reload would restart the server on edits that cannot take
    effect, which looks like a working dev loop while silently serving the code
    baked into the image. PYTHONPATH puts the mount ahead of site-packages.
    """
    if "./src:/app/src" not in compose:
        pytest.skip("compose no longer mounts the source tree")
    assert "PYTHONPATH" in compose, (
        "the source mount is inert without PYTHONPATH ahead of site-packages"
    )
    assert re.search(r"PYTHONPATH:\s*/app/src", compose)


def test_compose_reload_dir_matches_the_mount(compose):
    """--reload-dir must point at the mount target, not the host path."""
    if "--reload" not in compose:
        pytest.skip("compose no longer runs uvicorn with reload")
    match = re.search(r"--reload-dir\s+(\S+)", compose)
    assert match, "reload is on but no --reload-dir is set"
    target = match.group(1)
    assert f":{target}" in compose, f"--reload-dir {target} is not a mount target"


def test_compose_mounts_data_because_the_image_carries_no_models(dockerfile, compose):
    """Models are build outputs, not source, so the image has none.

    Without the data mount the API starts degraded - which /health reports
    honestly, but which would make `docker run aquanexus` look broken. The mount
    is the thing that makes compose work, so it must not quietly disappear.
    """
    assert not any(source.startswith("data") for source in copied_sources(dockerfile))
    assert "./data:/app/data" in compose


# ---------------------------------------------------------------------------
# Running the containers
# ---------------------------------------------------------------------------
#
# Everything above reads the build files. `scripts/check_containers.sh` runs
# what they produce, which is where the silent failures live - a mount that
# resolves into site-packages, an nginx that 404s a client route, a HEALTHCHECK
# whose arithmetic never converges. It needs a daemon, so it runs in CI. These
# tests hold the two halves together: that the workflow actually calls it, that
# it covers every route nginx is configured to treat specially, and that the
# stand-in artefacts it feeds the container cannot be mistaken for trained ones.

WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CONTAINER_CHECK = ROOT / "scripts" / "check_containers.sh"
STAND_IN = ROOT / "scripts" / "make_stand_in_artifacts.py"


@pytest.fixture(scope="module")
def container_check() -> str:
    return CONTAINER_CHECK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_builds_and_then_runs_both_images(workflow, container_check):
    """A build that is never run proves the image compiles, and nothing else."""
    assert "docker build -t aquanexus-api:ci ." in workflow
    assert "docker build -t aquanexus-frontend:ci ./frontend" in workflow
    assert "scripts/check_containers.sh" in workflow, (
        "CI builds the images without running them"
    )


def test_the_container_check_runs_the_frontend_image(container_check):
    """The frontend container is the half that had never been started at all."""
    assert "-e API_BASE_URL=" in container_check
    assert ":80 " in container_check or ':80"' in container_check


def test_the_container_check_covers_every_special_nginx_location(container_check):
    """A location with its own rules needs its own check, or the rules are untested."""
    config = (FRONTEND / "nginx.conf").read_text(encoding="utf-8")
    for location in re.findall(r"location\s+=?\s*(/\S*)\s*\{", config):
        if location == "/":
            assert "/no/such/page" in container_check, "the SPA fallback is unchecked"
            continue
        assert location in container_check, f"nothing checks the {location} rules"


def test_the_container_check_waits_for_the_healthcheck(container_check):
    """Polling /health with curl says nothing about the HEALTHCHECK in the image.

    The interval, timeout, start period and retry count are only exercised by
    the daemon, and they are what an orchestrator restarts a container over.
    """
    assert "State.Health.Status" in container_check
    assert "healthy" in container_check


def test_stand_in_artefacts_cannot_be_mistaken_for_trained_ones():
    """They exist to be loaded by a container, and to be obvious about it."""
    source = STAND_IN.read_text(encoding="utf-8")
    assert "STAND-IN ARTEFACT" in source
    assert '"stand_in": True' in source


def test_the_real_manifest_never_claims_to_be_a_stand_in():
    """The flag is only meaningful if the trained manifest never carries it."""
    from aquanexus.config import settings

    manifest_path = settings.MODELS_DIR / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no trained manifest; run scripts/train_models.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "stand_in" not in manifest


def test_stand_in_artefacts_load_and_refuse_to_overwrite(tmp_path, monkeypatch):
    """End to end, because the schema is the point: what CI mounts must load.

    Runs the generator against a temporary data directory, loads the result
    through the API's own registry, and then runs it a second time to check it
    will not overwrite artefacts that are already there - which is what stops
    somebody fabricating over a trained model on a machine that has one.
    """
    from aquanexus.api.registry import ModelRegistry
    from aquanexus.config import settings

    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "MODELS_DIR",
                 "HECRAS_DIR", "CACHE_DIR"):
        target = tmp_path if name == "DATA_DIR" else tmp_path / name.split("_")[0].lower()
        monkeypatch.setattr(settings, name, target)

    generator = load_script("make_stand_in_artifacts")
    monkeypatch.setattr(sys, "argv", ["make_stand_in_artifacts.py"])
    assert generator.main() == 0

    manifest = json.loads((settings.MODELS_DIR / "manifest.json").read_text("utf-8"))
    assert manifest["stand_in"] is True
    for meta in manifest["models"].values():
        assert meta["caveats"][0].startswith("STAND-IN ARTEFACT")

    registry = ModelRegistry().load(settings.MODELS_DIR)
    assert registry.error is None
    assert {"dissolved_oxygen", "hsi"} == set(registry.models)
    for model in registry.models.values():
        assert model.background is not None, "SHAP would explain a request against itself"
        assert model.collinear_pairs

    # Second run, no --force: refuses, and leaves what is there alone.
    before = (settings.MODELS_DIR / "manifest.json").read_bytes()
    assert generator.main() == 1
    assert before == (settings.MODELS_DIR / "manifest.json").read_bytes()


def test_the_stand_in_manifest_matches_the_trained_one(tmp_path, monkeypatch):
    """Same fields, because the same code writes both.

    Skipped without trained artefacts. Where they exist, this is what stops the
    container check from passing against a shape the real deployment never has.
    """
    from aquanexus.config import settings

    real_path = settings.MODELS_DIR / "manifest.json"
    if not real_path.is_file():
        pytest.skip("no trained manifest; run scripts/train_models.py")
    real = json.loads(real_path.read_text(encoding="utf-8"))

    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "MODELS_DIR",
                 "HECRAS_DIR", "CACHE_DIR"):
        target = tmp_path if name == "DATA_DIR" else tmp_path / name.split("_")[0].lower()
        monkeypatch.setattr(settings, name, target)

    generator = load_script("make_stand_in_artifacts")
    monkeypatch.setattr(sys, "argv", ["make_stand_in_artifacts.py"])
    assert generator.main() == 0
    stand_in = json.loads((settings.MODELS_DIR / "manifest.json").read_text("utf-8"))

    assert set(real["models"]) == set(stand_in["models"])
    for name, meta in real["models"].items():
        assert set(meta) == set(stand_in["models"][name]), f"{name} manifest fields differ"
        assert meta["features"] == stand_in["models"][name]["features"]


#: Long options BusyBox wget accepts. The Alpine image has BusyBox, not GNU
#: wget, and anything outside this set is rejected as "unrecognized option" -
#: which makes the probe fail while the server it probes is perfectly healthy.
BUSYBOX_WGET_OPTIONS = {
    "--continue", "--spider", "--quiet", "--output-document", "--header",
    "--post-data", "--post-file", "--proxy", "--user-agent",
    "--no-check-certificate", "--directory-prefix", "--passive-ftp",
}


def test_the_frontend_probe_uses_options_busybox_wget_has(frontend_dockerfile):
    """Regression: --tries made the frontend container permanently unhealthy.

    nginx served every request correctly; the HEALTHCHECK failed on its own
    arguments. Nothing caught it because nothing had ever run the container -
    `docker build` does not execute a HEALTHCHECK.
    """
    match = re.search(r"CMD wget ([^\n]*)", frontend_dockerfile)
    assert match, "the frontend HEALTHCHECK no longer probes with wget"

    used = {word.split("=")[0] for word in match.group(1).split() if word.startswith("--")}
    assert used <= BUSYBOX_WGET_OPTIONS, (
        f"BusyBox wget rejects {sorted(used - BUSYBOX_WGET_OPTIONS)}"
    )
