import unittest
import numpy as np
from src.retriever.retriever import FAISSRetriever

class TestFAISSRetriever(unittest.TestCase):
    def setUp(self):
        # Create a small test index with known vectors
        self.dimension = 4
        self.test_vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.test_texts = [
            "This is document 1",
            "This is document 2",
            "This is document 3",
            "This is document 4"
        ]
        self.retriever = FAISSRetriever(dimension=self.dimension)
        self.retriever.add_texts(self.test_texts, self.test_vectors)

    def test_initialization(self):
        self.assertEqual(self.retriever.dimension, self.dimension)
        self.assertIsNotNone(self.retriever.index)

    def test_add_texts(self):
        # Test if texts were added correctly
        self.assertEqual(len(self.retriever.texts), len(self.test_texts))
        self.assertEqual(self.retriever.texts, self.test_texts)

    def test_search(self):
        # Test search with a known vector
        query_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = self.retriever.search(query_vector, k=2)
        
        # The first result should be the exact match (document 1)
        self.assertEqual(results[0], "This is document 1")

    def test_search_k_limit(self):
        # Test if k limit is respected
        k = 2
        query_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = self.retriever.search(query_vector, k=k)
        self.assertEqual(len(results), k)

    def test_empty_index(self):
        # Test searching an empty index
        empty_retriever = FAISSRetriever(dimension=self.dimension)
        query_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = empty_retriever.search(query_vector, k=1)
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()