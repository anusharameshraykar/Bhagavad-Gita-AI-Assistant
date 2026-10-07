import chromadb

from src import config
from src.deployment import ensure_chroma_collection


def test_missing_chroma_collection_is_built(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="src")
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)
    monkeypatch.setattr(config, "EMBEDDER", "tfidf")

    assert ensure_chroma_collection() is True
    client = chromadb.PersistentClient(path=str(chroma_dir))
    assert client.get_collection(config.COLLECTION).count() > 0
    assert "starting deployment-time index build" in caplog.text
    assert "Embedding generation completed" in caplog.text


def test_existing_chroma_collection_is_not_rebuilt(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="src")
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection(config.COLLECTION)
    collection.add(ids=["existing"], documents=["keep this index"])

    assert ensure_chroma_collection() is False
    assert client.get_collection(config.COLLECTION).get()["ids"] == ["existing"]
    assert "skipping index and embedding build" in caplog.text
    assert "Embedding generation started" not in caplog.text
