    """Estimate class tests
    """
    import unittest

    from msc.embedding import GPEmbeddor
    
    TIME: float = 5
    DATASET: str = 
    class TestEmbeddorGetsData(unittest.TestCase):
        """Test that the Embeddor gets data"""
        
        def test_embeddor_attributes(self):
            """Ensures the embeddor attributes are as expected"""
            emb = GPEmbeddor()
