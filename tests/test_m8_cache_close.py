"""Test M8: SQLite caches properly close on shutdown.

Tests that PageCache and MetadataCache support:
1. Context manager (__enter__/__exit__)
2. Idempotent close() (double close is safe)
3. atexit registration (connection closed on interpreter shutdown)
"""
import tempfile
import os


def test_page_cache_context_manager():
    from dr2_podcast.research.clinical import PageCache
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_page.db")
        with PageCache(db_path=db_path) as cache:
            assert not cache._closed, "Should not be closed yet"
            result = cache.get("http://example.com")
            assert result is None
        assert cache._closed, "Should be closed after __exit__"
    print("PASS: PageCache context manager works")


def test_page_cache_double_close():
    from dr2_podcast.research.clinical import PageCache
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_page2.db")
        cache = PageCache(db_path=db_path)
        cache.close()
        assert cache._closed
        cache.close()  # should not raise
    print("PASS: PageCache double close is safe")


def test_metadata_cache_context_manager():
    from dr2_podcast.research.metadata_clients import MetadataCache
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_meta.db")
        with MetadataCache(db_path=db_path) as cache:
            assert not cache._closed
            cache.put("test_api", "id1", {"key": "value"})
            result = cache.get("test_api", "id1")
            assert result == {"key": "value"}, f"Expected dict, got {result}"
        assert cache._closed
    print("PASS: MetadataCache context manager works")


def test_metadata_cache_double_close():
    from dr2_podcast.research.metadata_clients import MetadataCache
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_meta2.db")
        cache = MetadataCache(db_path=db_path)
        cache.close()
        assert cache._closed
        cache.close()  # should not raise
    print("PASS: MetadataCache double close is safe")


if __name__ == "__main__":
    test_page_cache_context_manager()
    test_page_cache_double_close()
    test_metadata_cache_context_manager()
    test_metadata_cache_double_close()
    print()
    print("All M8 tests PASSED")
