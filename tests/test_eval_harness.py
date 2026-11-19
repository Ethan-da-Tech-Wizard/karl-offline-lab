# ==============================================================================
# Program: test_eval_harness.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising eval harness functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

# Scope: eval/harness.py (EvalHarness). Three behaviours are pinned down with scripted fakes instead of a real model:
#   1. model/adapter overrides passed to run() are the ones given to ModelLoader.get_instance;
#   2. a failing case is curated automatically into an "eval_chosen" and an "eval_rejected" training example (the DPO source);
#   3. with process isolation, a worker that exceeds its time limit is terminated rather than waited for.
# These tests are written as plain functions that patch ModelLoader by hand and restore it in `finally` (ModelLoader is a
# process-wide singleton). Run: python -m pytest tests/test_eval_harness.py -q
import os
import sys
import json
import tempfile
import shutil
import time

# Allows running the file directly from a checkout as well as under pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.model_loader import ModelLoader
from eval.harness import EvalHarness

# Fake llama-cpp model: __call__ returns the completion dict shape {"choices": [{"text": ...}]}. The <think> prefix imitates
# DeepSeek-R1 output, so the harness's thought/answer splitting is exercised too.
class MockLlama:
    """Scripted model: answers the electronics-return and Meridian-CEO questions (wrapped in <think> tags) and returns an unrelated answer otherwise."""
    def tokenize(self, text_bytes, add_bos=True):
        """Cheap token count (about one token per 4 bytes)."""
        # A cheap token counter (about 4 bytes per token) is enough for the harness's context-budget arithmetic.
        return [0] * (len(text_bytes) // 4 + 1)
        
    def __call__(self, prompt, **kwargs):
        """Return a canned reply chosen by keywords in the prompt."""
        if "electronics" in prompt:
            return {"choices": [{"text": "<think>thinking</think>The return window is 15 days."}]}
        if "CEO of Meridian" in prompt:
            return {"choices": [{"text": "<think>thinking</think>Sandra Holt is the CEO."}]}
        return {"choices": [{"text": "<think>thinking</think>unrelated response"}]}


# Stands in for the real worker process body; it never answers within the timeout used by the test.
def _sleeping_eval_worker(queue, *args):
    """Eval worker replacement that sleeps 10 seconds before answering, used to trigger the harness timeout."""
    time.sleep(10)
    queue.put({"ok": True, "output": "late", "latency": 10.0})


def test_eval_harness_model_and_adapter_selection():
    """EvalHarness.run with a model_name and adapter_name override loads exactly that model path (deepseek-r1-llama-8b.gguf) and adapter through ModelLoader for every case, and a two-case dataset is graded 2/2 passed with correct case ids."""
    print("Testing EvalHarness model and adapter selection...")
    
    # Save original ModelLoader attributes to restore later
    # Originals are saved first and restored in `finally` so this test cannot change behaviour of later tests.
    orig_get_instance = ModelLoader.get_instance
    orig_model_name = ModelLoader.model_name
    orig_n_ctx = ModelLoader.n_ctx

    mock_llama = MockLlama()

    # Every (model_path, adapter_name) pair the harness requests is recorded, so the test can check ALL calls, not just the first.
    get_instance_calls = []

    def mock_get_instance(model_path=None, adapter_name=None):
        """Record the (model_path, adapter_name) passed by the harness and return the mock model."""
        get_instance_calls.append((model_path, adapter_name))
        return mock_llama

    ModelLoader.get_instance = mock_get_instance
    ModelLoader.model_name = lambda: "mock-model.gguf"

    # Fixed context size so the harness's prompt budgeting is deterministic.
    ModelLoader.n_ctx = lambda: 4096

    # Create temporary dataset
    # Scratch folder for the dataset file; removed in `finally`.
    temp_dir = tempfile.mkdtemp()
    dataset_path = os.path.join(temp_dir, "test_dataset.jsonl")
    
    # Two tiny cases graded by the "keyword_hit" grader: the model double answers both correctly, so both should pass.
    cases = [
        {"id": "test_001", "prompt": "What is the return policy for electronics?", "expected": "15 days", "grader": "keyword_hit", "keywords": ["15 days"], "require_all": True},
        {"id": "test_002", "prompt": "Who is the CEO of Meridian Technologies?", "expected": "Sandra Holt", "grader": "keyword_hit", "keywords": ["Sandra Holt"], "require_all": True}
    ]
    
    try:
        with open(dataset_path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c) + "\n")
                
        harness = EvalHarness()
        
        # Run evaluation with a specific model and adapter override
        report = harness.run(
            dataset_path=dataset_path,
            workflow_name="grounded_answer",
            # Isolation off: a separate worker process would not see the ModelLoader patched in this process, so generation runs in-process.
            hyperparams={"eval_process_isolation": False, "machine_speak_enabled": False},
            # These overrides are the feature under test: they must reach ModelLoader unchanged.
            model_name="deepseek-r1-llama-8b.gguf",
            adapter_name="llama_8b_math_greeting"
        )
        
        # Verify call counts and parameters in ModelLoader
        # Guards against a harness that silently skips loading.
        assert len(get_instance_calls) > 0, "ModelLoader.get_instance should be called"
        # The first call is inside the run guard; subsequent ones are in the _run_model method
        # Checked for every call: a single call with the wrong model would mean part of the run used the wrong weights.
        for path, adapter in get_instance_calls:
            assert path is not None and "deepseek-r1-llama-8b.gguf" in path, f"Expected deepseek-r1-llama-8b.gguf, got {path}"
            assert adapter == "llama_8b_math_greeting", f"Expected llama_8b_math_greeting, got {adapter}"
            
        # Verify evaluation report metrics
        assert report.total == 2, f"Expected 2 total cases, got {report.total}"
        assert report.passed == 2, f"Expected 2 passed cases, got {report.passed}"
        assert report.failed == 0

        # Both scripted answers contain their keywords, so the pass rate must be exactly 1.0.
        assert report.pass_rate == 1.0
        
        # Verify individual case results
        assert report.cases[0].case_id == "test_001"
        assert report.cases[0].grade["passed"] is True
        assert report.cases[1].case_id == "test_002"
        assert report.cases[1].grade["passed"] is True
        
        print("EvalHarness model and adapter selection tests PASSED!")
        
    finally:
        # Restore ModelLoader state
        ModelLoader.get_instance = orig_get_instance
        ModelLoader.model_name = orig_model_name
        ModelLoader.n_ctx = orig_n_ctx
        shutil.rmtree(temp_dir)


def test_eval_harness_failure_curation():
    """A failing case is automatically curated into two training examples (the dataset's expected answer as 'chosen', the model's bad output as 'rejected') that export as one DPO pair."""
    print("Testing EvalHarness failure curation...")
    from eval.harness import EvalHarness
    from app.engine.model_loader import ModelLoader
    import app.utils.training_curator as tc
    
    # Save original states
    orig_get_instance = ModelLoader.get_instance
    orig_model_name = ModelLoader.model_name
    orig_n_ctx = ModelLoader.n_ctx

    # The curator writes to a module-level path; it is redirected to the temp dir so the real data/training/curated.jsonl is untouched.
    orig_curated_path = tc.CURATED_PATH
    
    temp_dir = tempfile.mkdtemp()
    tc.CURATED_PATH = os.path.join(temp_dir, "curated_eval_test.jsonl")
    dataset_path = os.path.join(temp_dir, "dataset.jsonl")
    
    # Always answers with text that cannot match the expected keyword.
    class FailingMockLlama:
        """Model double that always answers 'bad response output'."""
        def tokenize(self, text_bytes, add_bos=True):
            """Return a one-token list."""
            return [0]
        def __call__(self, prompt, **kwargs):
            """Return the fixed bad reply inside <think> tags."""
            return {"choices": [{"text": "<think>thinking</think>bad response output"}]}
            
    ModelLoader.get_instance = lambda **kwargs: FailingMockLlama()
    ModelLoader.model_name = lambda: "mock.gguf"
    ModelLoader.n_ctx = lambda: 4096
    
    # One case whose expected answer ("Shakespeare") the fake model never produces.
    case = {"id": "test_failed", "prompt": "Who wrote Hamlet?", "expected": "Shakespeare", "grader": "keyword_hit", "keywords": ["Shakespeare"], "require_all": True}
    
    try:
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(case) + "\n")
            
        harness = EvalHarness()
        report = harness.run(
            dataset_path=dataset_path,
            workflow_name="general_chat",
            hyperparams={"eval_process_isolation": False, "machine_speak_enabled": False},
        )
        
        # Verify case failed
        assert report.total == 1
        assert report.passed == 0
        assert report.failed == 1
        
        # Verify curated examples
        # Each failure yields two training examples: the expected answer as the good one and the model's own output as the bad one.
        examples = tc.get_all_examples()
        assert len(examples) == 2, f"Expected 2 entries, got {len(examples)}"
        
        # "source" tags tell the exporters which example is the preferred (chosen) and which the rejected side of a DPO pair.
        e_chosen = [e for e in examples if e["source"] == "eval_chosen"][0]
        e_rejected = [e for e in examples if e["source"] == "eval_rejected"][0]
        
        # messages = [system, user, assistant]; index 2 is the assistant reply.
        assert e_chosen["messages"][2]["content"] == "Shakespeare"
        assert e_rejected["messages"][2]["content"] == "bad response output"
        assert e_chosen["messages"][1]["content"] == "Who wrote Hamlet?"
        assert e_rejected["messages"][1]["content"] == "Who wrote Hamlet?"
        
        # Verify DPO export works for these
        dpo_file = os.path.join(temp_dir, "dpo.jsonl")

        # Verifies the end of the pipeline: the two examples are paired by prompt into one {prompt, chosen, rejected} line.
        tc.export_dpo(dpo_file)
        
        with open(dpo_file, "r") as f:
            dpo_lines = f.readlines()
        assert len(dpo_lines) == 1
        dpo_pair = json.loads(dpo_lines[0].strip())
        assert dpo_pair["chosen"][0]["content"] == "Shakespeare"
        assert dpo_pair["rejected"][0]["content"] == "bad response output"
        
        print("EvalHarness failure curation PASSED!")
    finally:
        ModelLoader.get_instance = orig_get_instance
        ModelLoader.model_name = orig_model_name
        ModelLoader.n_ctx = orig_n_ctx
        tc.CURATED_PATH = orig_curated_path
        shutil.rmtree(temp_dir)


def test_eval_harness_process_timeout_terminates_worker():
    """With process isolation, a worker that exceeds timeout_limit is terminated and TimeoutError('...terminated...') is raised quickly (<3s) instead of waiting the full 10s sleep."""
    harness = EvalHarness()

    # Swap the subprocess entry point for the slow fake so the timeout path can be reached without any real model.
    harness._worker_target = _sleeping_eval_worker

    start = time.perf_counter()
    try:
        try:
            harness._run_model_with_timeout(
                system_prompt="system",
                user_prompt="user",
                hyperparams={"max_tokens": 1},
                # 0.1 s against a worker that sleeps 10 s: the harness must kill the process and raise, not wait the full 10 s.
                timeout_limit=0.1,
            )
            raise AssertionError("Expected timeout")
        except TimeoutError as exc:
            # The error message must say the worker was terminated, so the failure reason is clear.
            assert "terminated" in str(exc)
    finally:
        elapsed = time.perf_counter() - start

    assert elapsed < 3.0


if __name__ == "__main__":
    test_eval_harness_model_and_adapter_selection()
    test_eval_harness_failure_curation()
    print("All eval harness unit tests PASSED!")
