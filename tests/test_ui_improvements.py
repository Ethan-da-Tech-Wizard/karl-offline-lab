# ==============================================================================
# Program: test_ui_improvements.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising ui improvements functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

# Qt-dependent tests: they import app.ui lazily inside each test so that collection works even
# when PyQt6 is absent (the whole class then fails only when run).
import os
import sys
import tempfile
import unittest
# Creates the offscreen QApplication all widget tests need.
import tests.qt_test_helper  # noqa: F401
# pytest provides the integration/model markers.
import pytest

# Makes `python tests/test_ui_improvements.py` runnable from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One TestCase per feature area; every test builds a fresh AppState so tests share no state.
class TestUIImprovements(unittest.TestCase):
    """Smoke/behaviour tests for recent GUI features: theme generation, codex library, workbench docks, swarm studio, agentic chat history and system-config search. Needs PyQt6, so it only runs where the GUI stack is installed."""
    # Guards the QSS compiler: get_theme_stylesheet formats a big template with palette keys. A new
    # palette lacking a key, or a typo in the template, would leave a literal {key} in the stylesheet
    # and Qt would silently ignore the broken rule.
    def test_theme_qss_generation(self):
        """The generated Qt stylesheet is non-trivial and has every {placeholder} (accent, bg_deep, ...) substituted; a leftover brace would mean broken styling."""
        from app.ui.themes import THEMES, get_theme_stylesheet
        # Every registered theme is exercised, not only the default one.
        for theme_name in THEMES.keys():
            qss = get_theme_stylesheet(theme_name)
            self.assertIsNotNone(qss)
            # Length > 100 ensures it is a real stylesheet rather than an empty/placeholder string.
            self.assertGreater(len(qss), 100)
            # Check that formatting keys were properly substituted
            # A leftover {placeholder} means the template substitution missed a key.
            self.assertNotIn("{accent}", qss)
            self.assertNotIn("{bg_deep}", qss)

    # Colour helpers are pure functions, so exact hex values are the right assertion.
    def test_theme_config_read_write(self):
        """Colour helpers produce exact values (darken_hex_color, _tint_hex_color), custom palettes override accent/background keys, and a warm tone shift changes the base palette."""
        from app.ui.themes import darken_hex_color, _tint_hex_color, get_theme_colors
        
        # Test color darkening helper: 0.5 means "scale each RGB channel to half": #00C2FF -> #00617F.
        # Test color darkening helper
        # Exact expected hex strings pin the colour maths so theme tweaks cannot silently drift.
        self.assertEqual(darken_hex_color("#00C2FF", 0.5), "#00617F")
        
        # Test tinting helper: adds (+5, -5, +10) per channel with clamping: #102030 -> #151B3A.
        # Test color tinting helper
        self.assertEqual(_tint_hex_color("#102030", 5, -5, 10), "#151B3A")
        
        # Verify custom combination palette building: a user accent and the "Pitch Black" tone must
        # override both the accent and the background keys.
        # Verify custom combination palette building
        colors = get_theme_colors("macOS Sonoma", custom_accent="#FF0000", bg_tone="Pitch Black")
        self.assertEqual(colors["accent"], "#FF0000")
        self.assertEqual(colors["bg_deep"], "#000000")
        self.assertEqual(colors["sidebar_bg"], "#000000")
        
        # Verify sepia/warm tone shifting: only requires the background to differ, not an exact colour.
        # Verify sepia/warm tone shifting
        base_colors = get_theme_colors("Karl Obsidian")
        warm_colors = get_theme_colors("Karl Obsidian", bg_tone="Warm Sepia")
        self.assertNotEqual(warm_colors["bg_deep"], base_colors["bg_deep"])

    # Docs workspace: markers keep it out of offline runs because DocsWorkspace builds a RAG index.
    @pytest.mark.integration
    @pytest.mark.model
    def test_codex_library_loading_and_search(self):
        """The Docs workspace loads the default codex library (AI Steering, Python, Docker, FastAPI), search filters the visible list, and selecting a topic renders its content on demand."""
        # Imported lazily so the skip happens before heavy modules load.
        from tests.conftest import embedding_model_available
        if not embedding_model_available():
            self.skipTest("sentence-transformers embedding model is unavailable (offline and not cached)")

        from app.ui.workspaces.docs_data import DEFAULT_LIBRARY
        from app.ui.workspaces.docs import DocsWorkspace
        from app.state import AppState
        
        # Verify default library structure: the four topics are the built-in baseline that ships in
        # docs_data.DEFAULT_LIBRARY.
        # Verify default library structure
        self.assertIn("AI Steering", DEFAULT_LIBRARY)
        self.assertIn("Python", DEFAULT_LIBRARY)
        self.assertIn("Docker", DEFAULT_LIBRARY)
        self.assertIn("FastAPI", DEFAULT_LIBRARY)
        
        # Instantiate AppState and DocsWorkspace.
        # Instantiate AppState and DocsWorkspace
        state = AppState()
        workspace = DocsWorkspace(state)
        
        # Check cache is populated.
        # Check cache is populated
        self.assertGreater(len(workspace._cache), 0)
        self.assertIn("AI Steering", workspace._cache)
        self.assertIn("Python", workspace._cache)
        
        # Test search filtering logic: the search box hides non-matching list rows rather than removing them.
        # Test search filtering logic
        workspace._search_input.setText("rust")
        # Collect only rows that are visible after filtering.
        
        visible_items = []
        for idx in range(workspace._topics_list.count()):
            item = workspace._topics_list.item(idx)
            if not item.isHidden():
                visible_items.append(item.text())
        
        self.assertIn("Rust", visible_items)
        
        # Test selecting a topic loads it dynamically (content is rendered on demand, not preloaded).
        # Test selecting a topic loads it dynamically
        workspace._on_topic_selected("Rust")
        # toHtml() returns the browser's rendered HTML; lowercase for a case-insensitive check.
        rendered_html = workspace._browser.toHtml().lower()
        self.assertIn("borrow", rendered_html)
        self.assertIn("ownership", rendered_html)

    # Layout contract of the Workbench: a QMainWindow so docks can be moved/closed, with sessions and
    # reasoning panels as QDockWidgets.
    def test_workbench_docks(self):
        """The Workbench is a QMainWindow with a central widget plus sessions and reasoning docks and the branch/token status labels."""
        from app.state import AppState
        from app.ui.workspaces.workbench import WorkbenchWorkspace
        from PyQt6.QtWidgets import QMainWindow, QDockWidget

        state = AppState()
        workbench = WorkbenchWorkspace(state)

        # Verify it inherits from QMainWindow.
        # Verify it inherits from QMainWindow
        self.assertTrue(isinstance(workbench, QMainWindow))

        # Verify central widget is set (the chat area).
        # Verify central widget is set
        self.assertIsNotNone(workbench.centralWidget())

        # Verify the two docks exist. The extra labels are status widgets used by the branch navigator and
        # the token budget display.
        # Verify the two docks exist
        self.assertTrue(hasattr(workbench, "_sessions_dock"))
        self.assertTrue(hasattr(workbench, "_reasoning_dock"))
        self.assertTrue(isinstance(workbench._sessions_dock, QDockWidget))
        self.assertTrue(isinstance(workbench._reasoning_dock, QDockWidget))
        self.assertTrue(hasattr(workbench, "_branch_stats_lbl"))
        self.assertTrue(hasattr(workbench, "_token_remaining_lbl"))

    # Smoke test: constructing the workspace exercises its whole __init__ wiring; the hasattr checks
    # pin the widget names other code (and the VS Code mirror) rely on.
    def test_swarm_studio_constructs(self):
        """The Swarm Studio workspace builds with its layers tree, task table and traceback browser."""
        from app.state import AppState
        from app.ui.workspaces.swarm_studio import SwarmStudioWorkspace

        state = AppState()
        workspace = SwarmStudioWorkspace(state)

        self.assertIsNotNone(workspace)
        self.assertTrue(hasattr(workspace, "_layers_tree"))
        self.assertTrue(hasattr(workspace, "_task_table"))
        self.assertTrue(hasattr(workspace, "_traceback_browser"))

    # Regression: the agentic loop appends internal "[Iteration N] Continue." user prompts. Persisting
    # them would pollute training data and the session history, so _on_loop_done must keep only the
    # first user message and the final answer.
    def test_agentic_loop_persists_only_final_assistant_answer(self):
        """After an agentic multi-iteration run only the final assistant answer is stored in the active chat history; the internal 'Iteration N' prompts must not leak into the saved conversation."""
        from app.state import AppState
        from app.ui.workspaces.workbench import WorkbenchWorkspace
        from app.utils.memory_manager import MemoryManager

        # Minimal stand-in for AgenticThread: _on_loop_done only reads .chat_history from it.
        class FakeThread:
            """Agentic thread double holding the history a real thread would produce across iterations."""
            def __init__(self, history):
                """Store the supplied history."""
                self.chat_history = history

        state = AppState()
        # TemporaryDirectory so MemoryManager writes sessions in a throwaway place, not data/sessions.
        with tempfile.TemporaryDirectory() as tmpdir:
            state.memory = MemoryManager(sessions_dir=tmpdir)
            workbench = WorkbenchWorkspace(state)

            # Seed the visible chat with the user's question, as the UI does on submit.
            user_node = workbench.chat_history.add_message("user", "how many sides does a hexagon have?")
            workbench._chat_view.push_user("how many sides does a hexagon have?", user_node.id)
            # The fake thread reports a four-message history: question, intermediate answer, loop prompt,
            # and the final answer.
            workbench._thread = FakeThread([
                {"role": "user", "content": "how many sides does a hexagon have?"},
                {"role": "assistant", "content": "A hexagon has 6 sides."},
                {"role": "user", "content": "[Iteration 2] Continue."},
                {"role": "assistant", "content": "FINAL ANSWER: A hexagon has 6 sides."},
            ])

            # 2 = number of iterations the loop reports.
            workbench._on_loop_done(2)
            # Snapshot of the active path through the session tree.
            active = list(workbench.chat_history)

            # Only two messages remain; ids are tree node ids assigned at insertion, so they are read back.
            self.assertEqual(
                active,
                [
                    {"role": "user", "content": "how many sides does a hexagon have?", "id": active[0]["id"]},
                    {"role": "assistant", "content": "FINAL ANSWER: A hexagon has 6 sides.", "id": active[1]["id"]},
                ],
            )
            # Belt and braces: no leaked iteration marker anywhere in the stored text.
            self.assertNotIn("Iteration", "\n".join(msg["content"] for msg in active))

    # Settings search + diagnostics in one construct: building SystemConfigWorkspace is expensive, so
    # the three behaviours are checked in a single test.
    def test_system_config_features(self):
        """System Config: settings search hides non-matching rows and restores them when cleared, the preflight diagnostic text reports an adapter mismatch, and the model/adapter lists are cached."""
        from app.state import AppState
        from app.ui.workspaces.system_config import SystemConfigWorkspace
        
        state = AppState()
        # A model/adapter mismatch (Qwen model, llama adapter) is the scenario that must trigger the
        # compatibility warning.
        state.model_name = "deepseek-r1-distill-qwen-1.5b.gguf"
        state.adapter_name = "llama-adapter"
        
        workspace = SystemConfigWorkspace(state)
        
        # Test settings search filtering: typing "temp" matches Temperature but not Top-P.
        # Test settings search filtering
        workspace._settings_search_input.setText("temp")
        row_temp = next(r[1] for r in workspace._settings_rows if r[0] == "Temperature")
        row_topp = next(r[1] for r in workspace._settings_rows if r[0] == "Top-P")
        self.assertFalse(row_temp.isHidden())
        self.assertTrue(row_topp.isHidden())
        
        # Clear search and make sure both are visible.
        # Clear search and make sure both are visible
        workspace._settings_search_input.setText("")
        self.assertFalse(row_temp.isHidden())
        self.assertFalse(row_topp.isHidden())
        
        # Test preflight checks / health check warnings.
        # Test preflight checks / health check warnings
        workspace._run_model_preflight_checks()
        status_text = workspace._model_status.text()
        self.assertIn("MODEL DIAGNOSTIC REPORT", status_text)
        self.assertIn("Adapter Compatibility Mismatch", status_text)
        
        # Test cache population: filesystem listings are cached so the UI does not hit disk on every
        # repaint.
        # Test cache population
        workspace.refresh_filesystem_cache()
        self.assertIsNotNone(workspace._cached_models_list)
        self.assertIsNotNone(workspace._cached_adapters_list)

    # Codex RAG uses a second index so reference docs never mix with the user's own documents.
    # Skipped offline because ingest needs the embedding model.
    @pytest.mark.integration
    @pytest.mark.model
    def test_codex_rag_features(self):
        """Codex search uses its own FAISS index (codex_index.faiss, separate from the document index), is populated with chunks, and a selected codex topic is injected into the chat input as a '[Codex: Topic]' block."""
        from tests.conftest import embedding_model_available
        if not embedding_model_available():
            self.skipTest("sentence-transformers embedding model is unavailable (offline and not cached)")

        # Late imports for the same reasons as above.
        from app.state import AppState
        from app.ui.workspaces.docs import DocsWorkspace
        from app.ui.workspaces.workbench import WorkbenchWorkspace
        import tempfile
        import shutil
        
        # Two RAG namespaces share one temp dir; the namespace decides the index file name.
        state = AppState()
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        
        from app.utils.rag_pipeline import RAGPipeline
        state.rag = RAGPipeline(index_path=temp_dir, namespace="user")
        state.codex_rag = RAGPipeline(index_path=temp_dir, namespace="codex")
        
        workbench = WorkbenchWorkspace(state)
        docs_workspace = DocsWorkspace(state, workbench_ref=workbench)
        
        # Different filenames prove the indexes are separate.
        self.assertIn("codex_index.faiss", state.codex_rag.INDEX_FILE)
        self.assertIn("index.faiss", state.rag.INDEX_FILE)
        
        # Ingest a codex topic (HTML) and a user document (TXT) into their respective indexes.
        test_file = os.path.join(temp_dir, "TestTopic.html")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("<h2>TestTopic</h2><p>This is a descriptor metaclass borrow checker query manual.</p>")
            
        state.codex_rag.ingest_file(test_file)
        self.assertGreater(state.codex_rag.total_chunks, 0)
        
        user_file = os.path.join(temp_dir, "UserDoc.txt")
        with open(user_file, "w", encoding="utf-8") as f:
            f.write("Information about local client setups.")
        state.rag.ingest_file(user_file)
        
        # Seed the docs view as if it had loaded two topics from disk.
        docs_workspace._cache["TestTopic"] = {
            "filepath": test_file,
            "content": "<h2>TestTopic</h2><p>This is a descriptor metaclass borrow checker query manual.</p>"
        }
        docs_workspace._cache["OtherTopic"] = {
            "filepath": os.path.join(temp_dir, "OtherTopic.html"),
            "content": "Some other content"
        }
        docs_workspace._topics_list.clear()
        docs_workspace._topics_list.addItem("TestTopic")
        docs_workspace._topics_list.addItem("OtherTopic")
        
        # Semantic search should select the topic that contains the query text.
        docs_workspace._filter_topics("borrow checker")
        
        current = docs_workspace._topics_list.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(current.text(), "TestTopic")
        
        # The send action opens a confirmation dialog; replace it so the test is non-blocking and restore
        # it afterwards, since QMessageBox is process-global.
        from PyQt6.QtWidgets import QMessageBox
        original_info = QMessageBox.information
        QMessageBox.information = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
        try:
            docs_workspace._send_to_workbench()
        finally:
            QMessageBox.information = original_info
            
        # The topic text must arrive in the chat input wrapped in a "[Codex: Topic]" block.
        input_text = workbench._input.toPlainText()
        self.assertIn("[Codex: TestTopic]", input_text)
        self.assertIn("descriptor metaclass borrow checker query manual", input_text)

if __name__ == "__main__":
    unittest.main()
