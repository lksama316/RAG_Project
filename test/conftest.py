"""Shared test isolation for node unit tests."""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _external_call_not_mocked(*args, **kwargs):
    raise RuntimeError("External dependency call was not mocked by the test.")


# node_item_name_confirm contains an unused top-level import named format_json.
# Stub it so unit tests can import the real node logic without installing an
# unrelated package.
if "format_json" not in sys.modules:
    format_json_stub = types.ModuleType("format_json")
    format_json_stub.format_json = _external_call_not_mocked
    sys.modules["format_json"] = format_json_stub

# Importing the real mongo/minio utility modules opens network connections.
# Node tests replace these module-level functions with mocks as needed.
mongo_history_stub = types.ModuleType("utils.mongo_history_utils")
mongo_history_stub.get_recent_messages = _external_call_not_mocked
mongo_history_stub.save_chat_message = _external_call_not_mocked
mongo_history_stub.update_message_item_names = _external_call_not_mocked
mongo_history_stub.clear_history = _external_call_not_mocked
sys.modules["utils.mongo_history_utils"] = mongo_history_stub

minio_utils_stub = types.ModuleType("utils.minio_utils")
minio_utils_stub.get_minio_client = _external_call_not_mocked
sys.modules["utils.minio_utils"] = minio_utils_stub