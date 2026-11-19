# ==============================================================================
# Program: test_swarm.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Swarm Orchestrator Tests — Karl Workbench ========================================== Integration tests verifying the multi-agent planning, execution, and self-correction loop using a stateful mock LLM and a temporary codebase sandbox.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
Swarm Orchestrator Tests — Karl Workbench
==========================================
Integration tests verifying the multi-agent planning, execution, and self-correction
loop using a stateful mock LLM and a temporary codebase sandbox.
"""

# Importing the helper first has a side effect the tests rely on: it makes sure a QApplication /
# offscreen Qt platform exists (or a stub when PyQt6 is missing) before any QThread subclass is
# constructed. The noqa silences F401 because the import is used only for that side effect.
import tests.qt_test_helper  # noqa: F401

# Standard-library imports used by the fixtures below: os/tempfile build and inspect the sandbox,
# json serialises the scripted Architect plan, sys.executable runs the generated test script with
# the same interpreter that runs pytest.
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# SwarmOrchestratorThread is the Architect -> Coder -> Tester pipeline under test; rollback_swarm_run
# is the module-level undo helper that replays the .karl_swarm_backups manifest.
from app.engine.swarm_orchestrator import SwarmOrchestratorThread, rollback_swarm_run



# One TestCase for the whole orchestrator: every test builds its own thread object but shares this
# fixture. The thread's run() is called directly (not start()) so everything executes
# synchronously on the test thread, which makes signal ordering deterministic and avoids needing
# an event loop.
class TestSwarmOrchestrator(unittest.TestCase):
    """Tests for SwarmOrchestratorThread (Architect plans, Coder edits, tests verify, self-correct) and rollback_swarm_run. The local LLM is replaced by scripted mocks and the QThread is run synchronously via run()."""
    def setUp(self):
        """Create a temp workspace with a dummy test script that asserts add(2, 3) == 5, and disable Machine Speak so prompts are not compacted and mock LLMs can match plain text."""
        # A real TemporaryDirectory (not a mock filesystem) because the orchestrator shells out to
        # subprocesses and copies files; only a real directory exercises those paths.
        # Create a temporary sandbox directory for code editing and testing
        self.sandbox_dir = tempfile.TemporaryDirectory()
        # All file edits happen inside this temp directory, never in the real repo.
        self.workspace_path = self.sandbox_dir.name

        # Machine Speak rewrites prompts into a compact dialect; the scripted LLMs below match on plain
        # English markers such as "Architect", so the feature must be off or no mock would ever match.
        # Patch config_store to disable machine speak in tests to avoid prompt compaction
        self.config_patcher = patch("app.engine.config_store.get_ui_config")
        self.mock_get_ui_config = self.config_patcher.start()
        self.mock_get_ui_config.return_value = {
            "enable_machine_speak": False,
            "theme_preset": "Karl Obsidian Core",
            "layout_preset": "Focused Workbench",
        }

        # The test script is the oracle for the self-correction test: it exits 1 until math_utils.add
        # really returns a+b, giving the orchestrator a genuine pass/fail signal from a subprocess.
        # Create dummy python test script in the workspace that asserts add(2, 3) == 5
        self.test_script_path = os.path.join(self.workspace_path, "run_tests.py")
        test_script_content = (
            # sys.dont_write_bytecode stops the script littering __pycache__ into the workspace, which would
            # otherwise show up as an unexpected file in the sandbox.
            "import sys\n"
            "sys.dont_write_bytecode = True\n"
            "try:\n"
            "    from math_utils import add\n"
            "    assert add(2, 3) == 5, f'Expected 5, got {add(2,3)}'\n"
            "    print('TESTS PASSED')\n"
            "    sys.exit(0)\n"
            "except Exception as e:\n"
            "    print(f'TESTS FAILED: {e}')\n"
            "    sys.exit(1)\n"
        )
        with open(self.test_script_path, "w", encoding="utf-8") as f:
            f.write(test_script_content)

    def tearDown(self):
        # Order matters: stop the patcher first so a failing cleanup() cannot leave get_ui_config patched
        # for later test modules.
        """Stop the config patcher and delete the temp workspace."""
        self.config_patcher.stop()
        self.sandbox_dir.cleanup()

    def test_generated_smoke_suite_passes_without_test_command(self):
        # Scenario: test_command is empty, so the orchestrator must fall back to its generated smoke suite
        # instead of declaring success with nothing run. Guards against silently "passing" unverified code.
        """When no test command is configured, the orchestrator generates a smoke suite (compile/import checks); valid code passes with the 'GENERATED SMOKE TEST PASSED' trace and temporary helper files are cleaned up."""
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create a helper module.",
            test_command=""
        )

        # This private method runs the proposal in a throwaway copy of the workspace, so the real
        # workspace is untouched; the final assert on helper.py proves that isolation.
        passed, trace = orchestrator._verify_proposals_in_temp_workspace(
            [{"filepath": "helper.py", "content": "def ok():\n    return True\n"}],
            layer_index=1,
        )

        self.assertTrue(passed)
        self.assertIn("GENERATED SMOKE TEST PASSED", trace)
        self.assertFalse(os.path.exists(os.path.join(self.workspace_path, "helper.py")))

    def test_generated_smoke_suite_fails_on_python_syntax_error(self):
        # Negative twin of the previous test: a syntax error must surface as FAILED with the compile
        # message, otherwise the self-correction loop would never be told what to fix.
        """The generated smoke suite must fail with a 'python compile failed' trace when an edited file has a syntax error."""
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create a helper module.",
            test_command=""
        )

        # "def broken(:" is deliberately unparseable by the Python compiler.
        passed, trace = orchestrator._verify_proposals_in_temp_workspace(
            [{"filepath": "helper.py", "content": "def broken(:\n    pass\n"}],
            layer_index=1,
        )

        self.assertFalse(passed)
        self.assertIn("GENERATED SMOKE TEST FAILED", trace)
        self.assertIn("python compile failed", trace)

    def test_backup_workspace_file_preserves_existing_content(self):
        # Safety net for the undo feature: backups must exist BEFORE the write, and live in a hidden
        # folder inside the workspace so rollback works even if the app restarts.
        """Before a file is overwritten, its previous content is copied into the .karl_swarm_backups folder so the run can be undone."""
        source_path = os.path.join(self.workspace_path, "existing.py")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write("VALUE = 'before'\n")

        # The orchestrator is constructed only to get its backup root and run id; no run is started.
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Update an existing file.",
            test_command=""
        )

        # _resolve_task_path returns (relative, absolute); index [1] is the absolute path to copy.
        backup_path = orchestrator._backup_workspace_file("existing.py", orchestrator._resolve_task_path("existing.py")[1])

        self.assertTrue(os.path.exists(backup_path))
        with open(backup_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "VALUE = 'before'\n")
        self.assertIn(".karl_swarm_backups", backup_path)

    def test_rollback_swarm_run_restores_existing_file(self):
        # Round trip: backup, record in manifest (existed=True), overwrite, then rollback must restore.
        """rollback_swarm_run puts a modified file back to its backed-up content and reports it under 'restored'."""
        source_path = os.path.join(self.workspace_path, "existing.py")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write("VALUE = 'before'\n")

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Update an existing file.",
            test_command=""
        )
        # True in the manifest record means "file existed before the run", so rollback copies it back.
        backup_path = orchestrator._backup_workspace_file("existing.py", orchestrator._resolve_task_path("existing.py")[1])
        orchestrator._record_backup_manifest("existing.py", backup_path, True)
        # Simulate the swarm having overwritten the file after the backup.
        with open(source_path, "w", encoding="utf-8") as f:
            f.write("VALUE = 'after'\n")

        # The run id ties the rollback call to the manifest written by this orchestrator instance.
        result = rollback_swarm_run(self.workspace_path, orchestrator._run_id)

        self.assertTrue(result["ok"])
        self.assertIn("existing.py", result["restored"])
        with open(source_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "VALUE = 'before'\n")

    def test_rollback_swarm_run_removes_new_file(self):
        # Other half of rollback: a file that was created by the run has no previous content, so the only
        # correct undo is deleting it. False in the manifest marks "did not exist before".
        """Files that did not exist before the run are deleted on rollback and reported under 'removed'."""
        new_path = os.path.join(self.workspace_path, "new_file.py")
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create a new file.",
            test_command=""
        )
        backup_path = orchestrator._backup_workspace_file("new_file.py", orchestrator._resolve_task_path("new_file.py")[1])
        # existed=False -> no backup content is needed, only the record that the path is new.
        orchestrator._record_backup_manifest("new_file.py", backup_path, False)
        with open(new_path, "w", encoding="utf-8") as f:
            f.write("VALUE = 'new'\n")

        result = rollback_swarm_run(self.workspace_path, orchestrator._run_id)

        self.assertTrue(result["ok"])
        self.assertIn("new_file.py", result["removed"])
        self.assertFalse(os.path.exists(new_path))

    # Crash-safety: the manifest is the only record that lets rollback work, so a crash mid-write must
    # never corrupt it. The implementation writes a temp file then os.replace()s it (atomic on POSIX).
    def test_backup_manifest_write_is_atomic(self):
        """A failed manifest replace leaves the previous complete manifest.json intact (temp + os.replace)."""
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Atomic manifest.",
            test_command=""
        )
        # First write succeeds and gives us a known-good manifest to compare against.
        orchestrator._record_backup_manifest("a.py", "/b/a.py", True)
        manifest = orchestrator._backup_root / "manifest.json"
        before = manifest.read_text(encoding="utf-8")

        # Force os.replace to fail as if the disk filled; the second record must raise and leave the
        # original manifest file byte-identical.
        with patch("app.engine.swarm_orchestrator.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                orchestrator._record_backup_manifest("b.py", "/b/b.py", False)

        # The old manifest must still parse and be unchanged; only a stray .tmp may remain.
        self.assertEqual(manifest.read_text(encoding="utf-8"), before)
        # json.loads proves the surviving manifest is still valid JSON (not merely equal text) and still
        # holds the first record.
        self.assertEqual(json.loads(before)["files"][0]["filepath"], "a.py")

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_swarm_self_correction_loop(self, mock_get_llm):
        # The big integration test of the feature. Order of events it pins down:
        #   1. Architect call -> JSON plan with one task.
        #   2. Coder call #1  -> buggy code (a - b) -> run_tests.py fails.
        #   3. The failure trace is appended to the next Coder prompt.
        #   4. Coder call #2  -> fixed code (a + b) -> run_tests.py passes.
        # It matters because self-correction is the headline claim of the swarm; if the trace never
        # reaches the prompt, the loop would just repeat the same bug until max_loops.
        """End-to-end with a stateful mock LLM: Architect plans one task, the Coder first writes buggy code (a - b), the test run fails, the failure trace is fed back, the Coder fixes it (a + b) and the second run passes. Asserts two test runs (False then True), the final code, and a successful finish naming the file."""
        # We simulate a stateful local LLM call sequence
        # Call 1: Architect planning
        # Call 2: Coder first attempt (buggy: return a - b)
        # Call 3: Coder second attempt after error feedback (corrected: return a + b)
        # call_count is kept for debugging only; the branches below route on prompt text.
        call_count = 0

        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model: first call returns the Architect plan, the next Coder call writes buggy code (a - b), and a Coder call that sees the earlier failure in its prompt returns the fix (a + b)."""
            nonlocal call_count
            call_count += 1
            
            # Match the Architect prompt first: the Coder prompt also contains the word "tasks" in some
            # templates, so order of these if/elif branches is significant.
            # 1. Architect call
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "Create a math utility with addition.",
                    "tasks": [
                        {
                            "filepath": "math_utils.py",
                            "instructions": "Define an add(a, b) function."
                        }
                    ]
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            
            # 2. Coder call
            elif "Coder" in prompt or "add(a, b)" in prompt:
                # The Coder prompt is recognised by role name or by the instruction text from the plan.
                # If coder sees previous warning in prompt, return the fix
                # The retry prompt embeds the previous failure ("TESTS FAILED" comes from run_tests.py output),
                # which is how the mock "sees" the feedback exactly like a real model would.
                if "Warning: Previous test failed" in prompt or "TESTS FAILED" in prompt:
                    correct_code = "def add(a, b):\n    return a + b\n"
                    return {"choices": [{"text": correct_code}]}
                else:
                    # Buggy implementation first (subtraction instead of addition)
                    buggy_code = "def add(a, b):\n    return a - b\n"
                    return {"choices": [{"text": buggy_code}]}
            
            # Catch-all keeps unexpected prompts (e.g. a Judge call) from raising StopIteration.
            # Catch-all
            return {"choices": [{"text": ""}]}

        # The mock returns different text per call (plan, buggy code, fixed code) based on prompt
        # contents and call order.
        # Register the stateful mock
        # MagicMock wrapping a function: still records calls but delegates to the scripted behaviour.
        mock_llm_callable = MagicMock(side_effect=stateful_mock_llm)
        # ModelLoader.get_instance is patched at the class, so every agent inside the orchestrator gets
        # this fake instead of loading a real GGUF.
        mock_get_llm.return_value = mock_llm_callable

        # Instantiate orchestrator targeting sandbox
        objective = "Create an add function in math_utils.py"
        # f-string with sys.executable avoids relying on "python" being on PATH in CI.
        test_command = f"{sys.executable} run_tests.py"
        
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective=objective,
            test_command=test_command
        )

        # Collect every Qt signal the orchestrator emits so assertions can inspect the full event
        # sequence. Plain lists/dicts are used instead of QSignalSpy to keep the test free of Qt test deps.
        # Collect every Qt signal the orchestrator emits so assertions can inspect the full event
        # sequence.
        # Track signals emitted
        signals = {
            "status_messages": [],
            "plan": None,
            "edited_files": {},
            "test_results": [],
            "finished": None
        }

        # Signals connected to lambdas execute synchronously because run() is called on this thread
        # (direct connection), so no event loop is needed.
        orchestrator.status_update.connect(lambda msg: signals["status_messages"].append(msg))
        orchestrator.task_plan_created.connect(lambda plan: signals.update({"plan": plan}))
        orchestrator.file_edited.connect(lambda path, content: signals["edited_files"].update({path: content}))
        orchestrator.test_result.connect(lambda passed, trace: signals["test_results"].append((passed, trace)))
        orchestrator.finished_swarm.connect(lambda success, summary: signals.update({"finished": (success, summary)}))
        # Stand-in for the user's cherry-pick dialog: auto-accept every proposed file. Without this the
        # orchestrator would wait for human approval forever.
        orchestrator.edits_proposed.connect(lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals]))

        # Run QThread synchronously in test thread to simplify assertions
        # Run QThread synchronously in test thread to simplify assertions.
        orchestrator.run()

        # --- ASSERTIONS ---
        # Verify Architect planned properly
        # Verify Architect planned properly: the plan signal carries the parsed JSON task list.
        self.assertIsNotNone(signals["plan"])
        self.assertEqual(signals["plan"]["tasks"][0]["filepath"], "math_utils.py")

        # Verify Coder created 'math_utils.py' (file_edited fires only after approval and write).
        # Verify Coder created 'math_utils.py'
        self.assertIn("math_utils.py", signals["edited_files"])

        # Verify the self-correction cycle:
        # - Attempt 1 failed (2 - 3 = -1, not 5)
        # - Attempt 2 succeeded (2 + 3 = 5)
        # Therefore two test runs: first False, then True.
        # Verify the self-correction cycle:
        # - Attempt 1 failed (2 - 3 = -1, not 5)
        # - Attempt 2 succeeded (2 + 3 = 5)
        # Therefore, we should have two test runs: first False, then True
        self.assertEqual(len(signals["test_results"]), 2)
        self.assertFalse(signals["test_results"][0][0])  # First test run failed
        self.assertTrue(signals["test_results"][1][0])   # Second test run passed

        # Verify the final code is the corrected version, not the first buggy draft.
        # Verify final code is correct
        final_code = signals["edited_files"]["math_utils.py"]
        self.assertIn("return a + b", final_code)

        # Success flag plus a summary string that mentions the file lets the UI show what changed.
        # Verify final orchestrator status
        self.assertIsNotNone(signals["finished"])
        self.assertTrue(signals["finished"][0])  # Success is True
        self.assertIn("math_utils.py", signals["finished"][1])  # Summary names modified file

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    # Independence rule: tasks that do not import one another land in the same dependency layer and
    # run concurrently. Pins the layering logic and that parallel edits do not clobber each other.
    def test_swarm_parallel_execution(self, mock_get_llm):
        """Two independent tasks (no import dependency) are placed in a single layer and edited in parallel; both files are produced and the run succeeds."""
        # We simulate a stateful LLM to return Architect plan then Coder edits. Here replies are keyed by
        # file name in the prompt so each parallel Coder gets its own file's content, regardless of the
        # order the thread pool schedules them.
        # We simulate a stateful LLM to return Architect plan then Coder edits
        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model: returns an Architect plan with two independent files, then the matching Coder content for each file."""
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "Create two independent helper files.",
                    "tasks": [
                        {
                            "filepath": "math_utils.py",
                            "instructions": "Define an add(a, b) function."
                        },
                        {
                            "filepath": "string_utils.py",
                            "instructions": "Define a reverse(s) function."
                        }
                    ]
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            elif "math_utils.py" in prompt:
                return {"choices": [{"text": "def add(a, b):\n    return a + b\n"}]}
            elif "string_utils.py" in prompt:
                return {"choices": [{"text": "def reverse(s):\n    return s[::-1]\n"}]}
            return {"choices": [{"text": ""}]}

        # Wrap the function in MagicMock for the same reason as in the self-correction test.
        mock_llm_callable = MagicMock(side_effect=stateful_mock_llm)
        mock_get_llm.return_value = mock_llm_callable

        objective = "Create math_utils.py and string_utils.py"
        # "echo" is a portable always-passing test command; correctness of the code is not under test here.
        test_command = "echo 'tests passed'"

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective=objective,
            test_command=test_command
        )

        # No "test_results" key: this test only cares about plan, layers and edits.
        signals = {
            "status_messages": [],
            "plan": None,
            "layers": None,
            "edited_files": {},
            "finished": None
        }

        orchestrator.status_update.connect(lambda msg: signals["status_messages"].append(msg))
        orchestrator.task_plan_created.connect(lambda plan: signals.update({"plan": plan}))
        # dependency_layers_built delivers the topological layers; one layer with two tasks == parallel.
        orchestrator.dependency_layers_built.connect(lambda layers: signals.update({"layers": layers}))
        orchestrator.file_edited.connect(lambda path, content: signals["edited_files"].update({path: content}))
        orchestrator.finished_swarm.connect(lambda success, summary: signals.update({"finished": (success, summary)}))
        orchestrator.edits_proposed.connect(lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals]))

        orchestrator.run()

        # Check that we parsed imports and sorted into 1 layer of 2 parallel tasks.
        # Check that we parsed imports and sorted into 1 layer of 2 parallel tasks
        self.assertIsNotNone(signals["plan"])
        self.assertEqual(len(signals["plan"]["tasks"]), 2)
        self.assertIsNotNone(signals["layers"])
        self.assertEqual(len(signals["layers"]), 1)
        self.assertEqual(len(signals["layers"][0]), 2)
        self.assertIn("math_utils.py", signals["edited_files"])
        self.assertIn("string_utils.py", signals["edited_files"])
        self.assertIsNotNone(signals["finished"])
        self.assertTrue(signals["finished"][0])

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_multiverse_candidates_default_to_one_unchanged_behavior(self, mock_get_llm):
        # Regression guard for backward compatibility: Swarm 2.0 added "multiverse" best-of-N candidates,
        # and the default of 1 must behave exactly as before (no extra LLM calls, no new signals).
        """candidates_per_task defaults to 1 -- no candidate/winner signals fire,
        and exactly one LLM call happens per phase, matching pre-Swarm-2.0 behavior."""
        call_count = 0

        def stateful_mock_llm(prompt, **kwargs):
            # Counting calls is what proves no hidden extra generations occur.
            """Scripted Architect and Coder replies that also count how many LLM calls are made."""
            nonlocal call_count
            call_count += 1
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "x",
                    "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            return {"choices": [{"text": "def add(a, b):\n    return a + b\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            test_command="echo ok",
        )
        # candidates_generated / winner_selected are the multiverse-only signals; both must stay silent.
        signals = {"candidates": [], "winner": None, "finished": None}
        orchestrator.candidates_generated.connect(lambda fp, n: signals["candidates"].append((fp, n)))
        orchestrator.winner_selected.connect(lambda fp, i, r: signals.update({"winner": (fp, i, r)}))
        orchestrator.finished_swarm.connect(lambda s, summary: signals.update({"finished": (s, summary)}))
        orchestrator.edits_proposed.connect(
            lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals])
        )

        # Nothing special: runs the same pipeline with the defaults.
        orchestrator.run()

        # exactly: 1 architect + 1 coder call.
        self.assertEqual(call_count, 2)  # exactly: 1 architect + 1 coder call
        self.assertEqual(signals["candidates"], [])  # no multiverse signal for a single candidate
        self.assertIsNone(signals["winner"])
        self.assertTrue(signals["finished"][0])

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_multiverse_selects_best_of_n_candidates(self, mock_get_llm):
        # Multiverse path: with candidates_per_task=3 the Coder is asked three times and a judge scores
        # each candidate. Pins that syntactic validity beats broken output, which is the whole point of
        # generating several candidates.
        """candidates_per_task > 1 generates N candidates and the Judge picks
        the syntactically valid one over broken ones."""
        # Shared across the Coder threads, hence only incremented, never read for control flow except
        # in the first-two-broken rule below.
        coder_call_count = 0

        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model: Architect plan, then three Coder candidates of which the first two are syntactically broken and the third is valid."""
            nonlocal coder_call_count
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "x",
                    "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            # Architect call does not count; only Coder calls advance the counter.
            coder_call_count += 1
            # First two candidates are syntactically broken (missing colon); only the third is valid.
            # First two candidates are syntactically broken; only the third is valid.
            if coder_call_count < 3:
                return {"choices": [{"text": "def add(a, b)\n    return a + b\n"}]}
            return {"choices": [{"text": "def add(a, b):\n    return a + b\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            test_command="echo ok",
            # hyperparams is where the UI sets candidates_per_task; here 3 candidates are requested.
            hyperparams={"candidates_per_task": 3},
        )
        signals = {"candidates": [], "scores": [], "winner": None, "finished": None}
        orchestrator.candidates_generated.connect(lambda fp, n: signals["candidates"].append((fp, n)))
        # candidate_scored carries a score dict per candidate; only the syntax_ok flag is asserted, as the
        # other fields are heuristic.
        orchestrator.candidate_scored.connect(lambda fp, i, sc: signals["scores"].append((i, sc["syntax_ok"])))
        orchestrator.winner_selected.connect(lambda fp, i, r: signals.update({"winner": (fp, i, r)}))
        orchestrator.finished_swarm.connect(lambda s, summary: signals.update({"finished": (s, summary)}))
        orchestrator.edits_proposed.connect(
            lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals])
        )

        orchestrator.run()

        # One announcement of (file, 3) proves N candidates were generated for that file.
        self.assertEqual(signals["candidates"], [("math_utils.py", 3)])
        self.assertEqual(len(signals["scores"]), 3)
        self.assertEqual(signals["scores"][0][1], False)  # candidate 0 broken
        self.assertEqual(signals["scores"][1][1], False)  # candidate 1 broken
        self.assertEqual(signals["scores"][2][1], True)   # candidate 2 valid
        self.assertEqual(signals["winner"][1], 2)          # winner is the valid one
        self.assertTrue(signals["finished"][0])

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    # Human-in-the-loop steering: a message injected by the user must show up as a fresh user turn in
    # the Coder's tool-loop prompt, otherwise the "mid-task correction" UI would be cosmetic.
    def test_inject_guidance_reaches_the_coder_tool_loop(self, mock_get_llm):
        """inject_guidance() queues a message that surfaces as a user turn in
        the Coder's next tool-loop generation."""
        # Every prompt is stored so the test can search them for the guidance marker afterwards.
        seen_prompts = []

        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model that records every prompt; the first Coder turn emits a tool call so a second turn happens and can show the injected guidance."""
            seen_prompts.append(prompt)
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "x",
                    "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            # len(seen_prompts)==2 means: prompt 1 was the Architect, prompt 2 is the first Coder turn.
            if len(seen_prompts) == 2:
                # First coder turn: emit a tool call so the loop continues to a second turn (where injected
                # guidance should be visible). Without a tool call the loop would end after one turn.
                # First coder turn: emit a tool call so the loop continues to a
                # second turn (where injected guidance should be visible).
                return {"choices": [{"text": (
                    "<reasoning>thinking</reasoning>"
                    "<tool_call name='read_file'>\n  path: math_utils.py\n</tool_call>"
                )}]}
            return {"choices": [{"text": "def add(a, b):\n    return a + b\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            test_command="echo ok",
        )
        orchestrator.edits_proposed.connect(
            lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals])
        )
        # Guidance injected before the run starts is still queued and drained on the coder's very first
        # tool-loop turn check.
        # Guidance injected before the run starts is still queued and drained
        # on the coder's very first tool-loop turn check.
        orchestrator.inject_guidance("math_utils.py", "Use a docstring explaining the function.")

        orchestrator.run()

        # The orchestrator prefixes injected text with this exact label.
        matching = [p for p in seen_prompts if "Human guidance (mid-task correction)" in p]
        self.assertTrue(matching, "expected the injected guidance to appear in a later coder prompt")
        self.assertIn("docstring", matching[0])

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    # Drift detection: if the same failure trace appears twice in a row for a file the model is stuck
    # in a loop, so the orchestrator should auto-inject a corrective nudge and emit drift_detected.
    def test_drift_detector_injects_guidance_after_repeated_failure_trace(self, mock_get_llm):
        """The same syntax error twice in a row for one file should trip the
        drift detector and auto-inject corrective guidance."""
        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model that always returns the same broken syntax, so the failure trace repeats verbatim."""
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "x",
                    "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            # Always the same broken syntax, so the resulting trace repeats verbatim.
            # Always the same broken syntax, so the resulting trace repeats verbatim.
            return {"choices": [{"text": "def add(a, b\n    return a + b\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            test_command="echo ok",
            # max_loops=3 gives at least two identical failures (needed to trip the detector) while bounding
            # the test's runtime.
            hyperparams={"max_loops": 3, "timeout_seconds": 60},
        )
        orchestrator.edits_proposed.connect(
            lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals])
        )
        # drift_detected is the UI hook (a warning banner); capturing it proves detection fired.
        drift_events = []
        orchestrator.drift_detected.connect(lambda fp, msg: drift_events.append((fp, msg)))

        orchestrator.run()

        self.assertTrue(drift_events, "expected a drift alert after repeated identical syntax errors")
        self.assertEqual(drift_events[0][0], "math_utils.py")
        self.assertIn("Drift detected", drift_events[0][1])

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    # max_loops is a hard cap on retries per layer; guards against an infinite correct-and-fail loop
    # burning GPU time. Here the code is always wrong, so only the cap can end the run.
    def test_max_loops_hyperparam_limits_layer_retries(self, mock_get_llm):
        """max_loops caps how many times a failing layer is retried."""
        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model whose Coder output is always buggy (subtraction), so verification keeps failing."""
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "x",
                    "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            # Always buggy (subtraction instead of addition), so verification keeps failing.
            # Always buggy (subtraction instead of addition), so verification keeps failing.
            return {"choices": [{"text": "def add(a, b):\n    return a - b\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            # Uses the real run_tests.py so failure is genuine, not simulated.
            test_command=f"{sys.executable} run_tests.py",
            # max_loops=1 means exactly one verification attempt.
            hyperparams={"max_loops": 1},
        )
        orchestrator.edits_proposed.connect(
            lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals])
        )
        test_results = []
        orchestrator.test_result.connect(lambda passed, trace: test_results.append(passed))
        finished = {}
        orchestrator.finished_swarm.connect(lambda success, summary: finished.update({"success": success}))

        orchestrator.run()

        # The attribute must reflect the hyperparameter that was passed.
        self.assertEqual(orchestrator.max_loops, 1)
        # no retry beyond the single allowed loop.
        self.assertEqual(len(test_results), 1)  # no retry beyond the single allowed loop
        # A run that exhausted its loops reports failure rather than pretending to succeed.
        self.assertFalse(finished["success"])

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    # Deadline behaviour: a wall-clock timeout must end the run cleanly with a "timed out" summary.
    # Instead of sleeping, the test replaces _timed_out with a lambda that is already True.
    def test_execution_timeout_stops_gracefully(self, mock_get_llm):
        """A run whose deadline has already passed should stop before doing
        any layer work instead of hanging or crashing."""
        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model returning a normal Architect plan and Coder output (the run is stopped by the deadline, not by the model)."""
            plan = {
                "explanation": "x",
                "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
            }
            return {"choices": [{"text": json.dumps(plan)}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            test_command="echo ok",
            hyperparams={"timeout_seconds": 30},
        )
        # The hyperparameter must be stored in seconds on the thread.
        self.assertEqual(orchestrator.timeout_seconds, 30)

        finished = {}
        orchestrator.finished_swarm.connect(
            lambda success, summary: finished.update({"success": success, "summary": summary})
        )
        # Simulate the deadline already having elapsed, without depending on real wall-clock time in the
        # test (which would be slow and flaky).
        # Simulate the deadline already having elapsed, without depending on
        # real wall-clock time in the test.
        orchestrator._timed_out = lambda: True

        orchestrator.run()

        self.assertEqual(finished.get("success"), False)
        self.assertIn("timed out", finished.get("summary", "").lower())

    @patch("app.engine.model_loader.ModelLoader.get_instance")
    # Security regression: the Coder agent can emit write_file tool calls; paths outside the plan's
    # allowed set must be refused, otherwise a prompt-injected model could write anywhere in the
    # workspace (here secret.py).
    def test_coder_write_blocked_outside_task_plan(self, mock_get_llm):
        """write_file calls targeting a path outside allowed_paths must be
        blocked rather than silently written to the real workspace."""
        # Imported lazily because swarm_agents pulls in heavy modules; only this test needs it.
        from app.engine.swarm_agents import CoderAgent

        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model whose Coder tries to write secret.py, a file outside the plan."""
            # The model "decides" to write a file that is not in the plan, then signals done.
            return {"choices": [{"text": (
                "<tool_call name='write_file'>\n"
                "  path: secret.py\n"
                "  content: pwned = True\n"
                "</tool_call>"
                "<tool_call name='done'></tool_call>"
            )}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        # CoderAgent is used directly, bypassing the orchestrator, to test the guard in isolation.
        coder = CoderAgent()
        task = {"filepath": "math_utils.py", "instructions": "Define add(a, b)."}
        # allowed_paths is the plan's whitelist; secret.py is not in it.
        coder.generate(
            task, {}, workspace_path=self.workspace_path,
            allowed_paths={"math_utils.py"},
        )

        # The file must not exist: the block must prevent the write, not merely log it.
        self.assertFalse(os.path.exists(os.path.join(self.workspace_path, "secret.py")))

    # Single-step debugging: step() lets exactly one boundary through while keeping pause mode armed,
    # so the user can walk through a run one task at a time.
    def test_step_unblocks_without_disabling_pause_mode(self):
        """step() releases the pause event for one boundary but keeps pause_on_step armed for the next."""
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path, objective="x", test_command="",
            hyperparams={"pause_on_step": True},
        )
        # Start in the paused state, as the orchestrator would be sitting at a step boundary.
        orchestrator.pause_event.clear()
        self.assertTrue(orchestrator.pause_on_step)

        # One step() call releases the event once.
        orchestrator.step()

        self.assertTrue(orchestrator.pause_event.is_set())
        self.assertTrue(orchestrator.pause_on_step)  # step() re-arms for the next boundary

    # resume() is the counterpart: continue normally, so pausing is switched off for good.
    def test_resume_disables_pause_mode(self):
        """resume() releases the pause event and turns step-pausing off entirely."""
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path, objective="x", test_command="",
            hyperparams={"pause_on_step": True},
        )
        # Paused first, so is_set() after resume() proves the call actually released it.
        orchestrator.pause_event.clear()

        orchestrator.resume()

        self.assertTrue(orchestrator.pause_event.is_set())
        self.assertFalse(orchestrator.pause_on_step)

    # Telemetry is emitted every 5th token to avoid flooding the UI with signals; the first token
    # emits immediately so the panel shows activity at once.
    def test_record_token_telemetry_emits_with_throttle(self):
        """Token telemetry is throttled: it emits on the 1st call and every 5th afterwards (1 and 5 here), with cumulative tokens, tokens/second and elapsed time."""
        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path, objective="x", test_command="",
        )
        events = []
        orchestrator.telemetry_update.connect(lambda payload: events.append(payload))

        # Seven calls: expect emits at #1 and #5, not #2-4, #6-7.
        for _ in range(7):
            orchestrator._record_token_telemetry()

        # Emits on the 1st call and every 5th call thereafter.
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["cumulative_tokens"], 1)
        self.assertEqual(events[1]["cumulative_tokens"], 5)
        self.assertIn("tokens_per_second", events[0])
        self.assertIn("elapsed_seconds", events[0])

    # End-to-end pause: the thread must emit swarm_paused with file and layer, block, and continue
    # only after resume(); otherwise the debugger UI's Pause button would be pointless.
    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_pause_on_step_emits_and_resume_unblocks(self, mock_get_llm):
        """With pause_on_step the orchestrator emits a paused event (file and layer index) at a step boundary and blocks until resume() is called."""
        # One-file plan keeps the layer index predictable (layer 1).
        def stateful_mock_llm(prompt, **kwargs):
            """Scripted Architect plan and Coder output for a one-file task."""
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "x",
                    "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            return {"choices": [{"text": "def add(a, b):\n    return a + b\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            test_command="echo ok",
            hyperparams={"pause_on_step": True},
        )
        orchestrator.edits_proposed.connect(
            lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals])
        )
        paused_events = []

        # Resuming inside the slot is safe because run() is synchronous here; with a real thread the UI
        # would call resume() from its own thread.
        def _on_paused(info):
            """Record the paused event and immediately resume the orchestrator."""
            paused_events.append(info)
            orchestrator.resume()

        orchestrator.swarm_paused.connect(_on_paused)
        orchestrator.run()

        # At least one pause must have been signalled.
        self.assertTrue(paused_events)
        self.assertEqual(paused_events[0]["filepath"], "math_utils.py")
        self.assertEqual(paused_events[0]["layer_index"], 1)
        # resume() disables further pausing.
        self.assertFalse(orchestrator.pause_on_step)  # resume() disabled further pausing

    # Mid-run steering of the plan: while paused, the user can rewrite the task description and the
    # Coder must use the new text. Pins that the override is applied before the Coder prompt is built.
    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_override_active_task_replaces_instructions(self, mock_get_llm):
        """While paused, override_active_task() replaces the active task's instructions, and the Coder prompt then contains the replacement text."""
        seen_prompts = []

        # Records prompts so we can look for the override text in the one used by the Coder.
        def stateful_mock_llm(prompt, **kwargs):
            """Scripted model that records every Coder prompt so the overridden instructions can be checked."""
            seen_prompts.append(prompt)
            if "Architect" in prompt or "tasks" in prompt:
                plan = {
                    "explanation": "x",
                    "tasks": [{"filepath": "math_utils.py", "instructions": "Define an add(a, b) function."}],
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            # Returns a subtract function just to differ from the original instruction ("add").
            return {"choices": [{"text": "def sub(a, b):\n    return a - b\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=stateful_mock_llm)

        orchestrator = SwarmOrchestratorThread(
            workspace_path=self.workspace_path,
            objective="Create an add function",
            test_command="echo ok",
            hyperparams={"pause_on_step": True},
        )
        orchestrator.edits_proposed.connect(
            lambda proposals: orchestrator.commit_selected_edits([p["filepath"] for p in proposals])
        )

        def _on_paused(info):
            """When paused, replace the active task's instructions with 'Define a subtract(a, b) function instead.' and resume."""
            # Rewrites the instructions of the paused task, then lets the run continue.
            orchestrator.override_active_task(info["filepath"], "Define a subtract(a, b) function instead.")
            orchestrator.resume()

        orchestrator.swarm_paused.connect(_on_paused)
        orchestrator.run()

        # "Task: Edit" is the header the Coder prompt template puts before the task text.
        coder_prompts = [p for p in seen_prompts if "Task: Edit" in p]
        self.assertTrue(coder_prompts)
        # The replacement text must appear and (implicitly) the original not drive the prompt.
        self.assertIn("Define a subtract(a, b) function instead.", coder_prompts[0])



if __name__ == "__main__":
    unittest.main()
