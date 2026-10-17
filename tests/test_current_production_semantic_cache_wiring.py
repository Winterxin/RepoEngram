"""Exercise installed current production; explicitly skip an absent production baseline."""

import hashlib
import inspect
import shutil
import subprocess
from copy import deepcopy

import pytest
from test_prepare import AntiPatternModel, WorkflowModel
from test_prepare import historical_repository as historical_repository
from test_prepare import repository as repository
from test_repo_snapshot_semantic_cache import RecordingModel, complete_identity

from skillforge.domain.agent_integration import CodeAgentTarget
from skillforge.domain.routing import SkillRoutingRequest
from skillforge.productization.evolution_artifacts import load_baseline
from skillforge.productization.prepare import PrepareRequest, SkillForgePreparer
from skillforge.repository.repo_snapshot_semantic_cache import RepoSnapshotSemanticCache
from skillforge.repository.skill_registry import SkillRegistryBuilder
from skillforge.repository.skill_router import SkillRouter


def test_current_four_channels_share_scope_and_keep_semantic_outputs(
    historical_repository, tmp_path, monkeypatch,
):
    if "tendency_model" not in inspect.signature(SkillForgePreparer).parameters:
        pytest.skip("CURRENT_PRODUCTION_BASELINE_BLOCKED: four-channel prepare absent")
    tendency = pytest.importorskip("skillforge.domain.semantic_tendency")
    from skillforge.repository.semantic_tendencies import tendency_request_fingerprint

    root = historical_repository
    folder = root / "src/app/plugins"
    folder.mkdir(exist_ok=True)
    (folder / "__init__.py").write_text("", encoding="utf-8")
    for args in (("add", "src"), ("commit", "-m", "Establish local package")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    for index in range(3):
        (folder / f"unit_{index}.py").write_text("VALUE = 1\n", encoding="utf-8")
        for args in (("add", "src"), ("commit", "-m", "Add independent local module")):
            subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    class TendencyModel:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return tendency.ScopedTendencyDecision(
                tendency_request_fingerprint(request),
                tendency.TendencyTransportDecision(
                    tendency.TendencyDecisionKind.INSUFFICIENT_EVIDENCE,
                    "The supplied observations are insufficient for this interpretation.",
                    tuple(e.evidence_id for e in request.evidence),
                ),
            )

    scopes = []

    class ObservedCache(RepoSnapshotSemanticCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.models = []
            self.publications = []
            scopes.append(self)

        def wrap(self, model):
            self.models.append(model)
            return super().wrap(model)

        def publish(self, *results):
            # The real prepare supplies its already assembled deterministic results.
            assert len(results) == 4 and all(r is not None for r in results)
            assert not results[-1].failures
            self.publications.append(results)
            return super().publish(*results)

    monkeypatch.setattr(
        "skillforge.productization.prepare.RepoSnapshotSemanticCache", ObservedCache,
    )
    directory = tmp_path.parent / (tmp_path.name + "-cache")

    def run(task):
        models = (RecordingModel(), WorkflowModel(), AntiPatternModel(), TendencyModel())
        for model in models:
            model.semantic_cache_identity = deepcopy(complete_identity())
            model.semantic_cache_identity["transport"]["source_sha256"] = hashlib.sha256(
                inspect.getsource(type(model).complete).encode("utf-8"),
            ).hexdigest()
        result = SkillForgePreparer(*models, semantic_cache_root=directory).prepare(
            PrepareRequest(root, task, CodeAgentTarget.CODEX),
        )
        baseline = load_baseline(root, result.evolution_baseline_path)
        registry = SkillRegistryBuilder().build(baseline.source_compilation)
        routes = tuple(SkillRouter().route(
            registry, SkillRoutingRequest(result.repository_id, query),
        ) for query in ("update app.api", "another task"))
        return models, baseline, registry, routes

    cold_models, *cold_artifacts = run("update app.api")
    assert cold_models[0].calls > 0
    assert all(model.requests for model in cold_models[1:])
    for name in (".skillforge", ".codex"):
        output = (root / name).resolve()
        assert output.parent == root.resolve()
        if output.exists():
            shutil.rmtree(output)
    warm_models, *warm_artifacts = run("another task")
    assert cold_artifacts == warm_artifacts
    assert warm_models[0].calls == 0 and all(not model.requests for model in warm_models[1:])
    assert len(scopes) == 2 and all(len(c.models) == 4 for c in scopes)
    assert all(len(c.publications) == 1 for c in scopes)
    assert scopes[1].hits == scopes[0].misses > 0 and scopes[1].misses == 0
