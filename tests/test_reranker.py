import unittest
import torch
from src.reranker.reranker import CrossAttentionReranker

class TestCrossAttentionReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = CrossAttentionReranker()
        self.test_query = "What is machine learning?"
        self.test_documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning is a type of machine learning.",
            "Natural language processing uses machine learning.",
            "This is an irrelevant document about cars."
        ]

    def test_initialization(self):
        self.assertIsNotNone(self.reranker.model)
        self.assertIsNotNone(self.reranker.tokenizer)

    def test_rerank(self):
        # Test basic reranking functionality
        reranked_docs = self.reranker.rerank(
            query=self.test_query,
            documents=self.test_documents,
            top_k=3
        )
        
        # Check if we get the requested number of results
        self.assertEqual(len(reranked_docs), 3)
        
        # Check if all returned documents are from the input documents
        for doc in reranked_docs:
            self.assertIn(doc, self.test_documents)

    def test_rerank_with_scores(self):
        # Test reranking with score output
        reranked_docs, scores = self.reranker.rerank(
            query=self.test_query,
            documents=self.test_documents,
            top_k=2,
            return_scores=True
        )
        
        # Check if we get scores
        self.assertEqual(len(scores), 2)
        # Check if scores are in descending order
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

    def test_empty_documents(self):
        # Test behavior with empty document list
        reranked_docs = self.reranker.rerank(
            query=self.test_query,
            documents=[],
            top_k=3
        )
        self.assertEqual(len(reranked_docs), 0)

    def test_top_k_limit(self):
        # Test if top_k limits output correctly
        k = 2
        reranked_docs = self.reranker.rerank(
            query=self.test_query,
            documents=self.test_documents,
            top_k=k
        )
        self.assertEqual(len(reranked_docs), k)

    def test_batch_processing(self):
        # Test batch processing with multiple documents
        batch_size = 2
        self.reranker.batch_size = batch_size
        reranked_docs = self.reranker.rerank(
            query=self.test_query,
            documents=self.test_documents,
            top_k=3
        )
        self.assertEqual(len(reranked_docs), 3)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_support(self):
        # Test GPU support if available
        reranker = CrossAttentionReranker(device="cuda")
        self.assertEqual(reranker.device, "cuda")
        reranked_docs = reranker.rerank(
            query=self.test_query,
            documents=self.test_documents,
            top_k=2
        )
        self.assertEqual(len(reranked_docs), 2)

if __name__ == '__main__':
    unittest.main()