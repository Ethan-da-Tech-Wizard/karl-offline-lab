# Program: karl-offline-lab - test_ui_drag_drop.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the training dataset list model and for the knowledge base workspace accepting dropped files while filtering out unsupported types.
"""Tests for dataset list model behaviour and drag-and-drop file ingestion."""
import os
import sys
from PyQt6.QtCore import QMimeData, QUrl, Qt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui.workspaces.knowledge_base import KnowledgeBaseWorkspace
from app.ui.workspaces.training_studio.dataset_tab import DatasetListModel

class MockState:
    """Minimal stand-in for the application state with a fake curator and fake retrieval index."""
    def __init__(self):
        """Build the fake curator, fake retrieval pipeline and the retrieval settings the workspace reads."""
        class MockCurator:
            """Fake curator holding two in-memory training examples."""
            def __init__(self):
                """Create the two sample examples."""
                self.examples = [
                    {"instruction": "say hello", "output": "hello there", "source": "thumbs_up"},
                    {"instruction": "fix python syntax", "output": "print('fixed')", "source": "corrected"}
                ]
            def get_all_examples(self):
                """Return the in-memory examples."""
                return self.examples
            def delete_example(self, row):
                """Remove the example at the given row if the row is in range."""
                if 0 <= row < len(self.examples):
                    self.examples.pop(row)
        
        class MockRag:
            """Fake retrieval pipeline that reports an empty, unloaded index."""
            def list_sources(self):
                """Return no indexed sources."""
                return []
            @property
            def is_encoder_loaded(self):
                """Report that no encoder is loaded."""
                return False
            @property
            def total_chunks(self):
                """Report zero indexed chunks."""
                return 0
            @property
            def total_sources(self):
                """Report zero indexed sources."""
                return 0
            INDEX_FILE = "index.faiss"
            META_FILE = "metadata.json"
        
        self.curator = MockCurator()
        self.rag = MockRag()
        self.rag_threshold = 0.5
        self.rag_top_k = 3

def test_dataset_list_model():
    """Scenario: the list model is filled with two entries.
    Expected: it starts empty, then reports two rows, returns items by row, returns None out of range and shows the text for display.
    """
    model = DatasetListModel()
    assert model.rowCount() == 0

    data = [
        "✓ positive: test 1",
        "✎ corrected: test 2"
    ]
    model.update_data(data)
    assert model.rowCount() == 2
    assert model.get_item(0) == data[0]
    assert model.get_item(1) == data[1]
    assert model.get_item(2) is None

    # Test display role formatting
    val = model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole)
    assert "test 1" in str(val)

def test_drag_drop_filter():
    """Scenario: a Python file and a PNG image are dropped on the knowledge base workspace.
    Expected: only the Python file is queued for ingestion and the status label says one file was queued.
    """
    state = MockState()
    # Instantiate workspace widget
    workspace = KnowledgeBaseWorkspace(state)
    
    # Verify drops are accepted
    assert workspace.acceptDrops()

    # Create dummy files
    # Create two real temporary files, one supported and one not, to drop.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f_py, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_png:
        py_name = f_py.name
        png_name = f_png.name

    try:
        # Mock drag/drop event
        # Build the drop payload with both file URLs.
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(py_name), QUrl.fromLocalFile(png_name)])
        
        class MockDropEvent:
            """Fake drop event exposing the dropped data and recording whether it was accepted."""
            def __init__(self, mime_data):
                """Store the mime data and mark the event as not yet accepted."""
                self._mime = mime_data
                self.accepted = False
            def mimeData(self):
                """Return the dropped mime data."""
                return self._mime
            def acceptProposedAction(self):
                """Record that the drop was accepted."""
                self.accepted = True

        mock_event = MockDropEvent(mime)
        # Deliver the fake drop event to the workspace.
        workspace.dropEvent(mock_event)
        
        # Verify queue contains the .py file and NOT the .png file
        queue_paths = [item["path"] for item in workspace._ingest_queue]
        assert py_name in queue_paths
        assert png_name not in queue_paths
        assert workspace._ingest_status.text() == "Queued 1 file(s) via Drag & Drop."
        
    finally:
        # Remove the temporary files whatever the test outcome.
        if os.path.exists(py_name):
            os.remove(py_name)
        if os.path.exists(png_name):
            os.remove(png_name)
