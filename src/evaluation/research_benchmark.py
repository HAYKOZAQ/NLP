import time
import pandas as pd
from typing import List, Dict
import numpy as np
import sys
import os
import re

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.retrieval import HybridRetriever
from src.core.generator import create_rag_chain


class HallucinationChecker:
    """
    Multi-method hallucination detection for RAG systems.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.nli_model = None
        self._init_nli_model()

    def _init_nli_model(self):
        """Initialize NLI model for entailment checking."""
        try:
            from transformers import pipeline

            self.nli_model = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,  # CPU
            )
            print("✓ NLI model loaded for hallucination detection")
        except Exception as e:
            print(f"⚠ NLI model unavailable: {e}")
            self.nli_model = None

    def extract_claims(self, answer: str) -> List[str]:
        """Extract individual claims/sentences from the answer."""
        # Split by sentence endings
        sentences = re.split(r"[։\.!\?]", answer)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims

    def check_nli_entailment(self, claim: str, context: str) -> float:
        """Check if context entails the claim using NLI."""
        if not self.nli_model or not context or not claim:
            return 0.5  # Neutral if unavailable

        try:
            # NLI format: premise [SEP] hypothesis
            # Shortened to avoid token window overflow (512 tokens)
            input_text = f"{context[:1000]} [SEP] {claim}"
            result = self.nli_model(input_text)[0]

            # DeBERTa NLI labels: ENTAILMENT, NEUTRAL, CONTRADICTION
            label = result["label"].upper()
            score = result["score"]

            if "ENTAIL" in label:
                return score  # High = grounded
            elif "CONTRADICT" in label:
                return 1.0 - score  # High contradiction = hallucination
            else:
                return 0.5  # Neutral
        except Exception as e:
            return 0.5

    def check_llm_verification(self, answer: str, context: str) -> Dict:
        """Use LLM to verify if answer is grounded in context."""
        if not self.llm:
            return {"score": 0.5, "explanation": "LLM unavailable"}

        verification_prompt = f"""You are a fact-checker. Analyze if the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context[:2000]}

ANSWER:
{answer}

Respond with ONLY a JSON object:
{{"grounded": true/false, "hallucinated_claims": ["list of claims not in context"], "score": 0.0-1.0}}"""

        try:
            response = self.llm.invoke(verification_prompt)
            # Try to parse JSON from response
            import json

            # Find JSON in response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "score": result.get("score", 0.5),
                    "hallucinated_claims": result.get("hallucinated_claims", []),
                    "grounded": result.get("grounded", True),
                }
        except Exception as e:
            pass

        return {"score": 0.5, "explanation": "Parse error"}

    def check_citation_grounding(self, answer: str) -> float:
        """Check if answer contains proper citations."""
        # Armenian citation patterns
        citation_patterns = [
            r"\[Աղբյուր:.*?\]",  # Armenian source citation
            r"\[Source:.*?\]",  # English source citation
            r"Հոդված\s+\d+",  # Article references (e.g., Հոդված 12)
            r"մաս\s+\d+",  # Part references (e.g., մաս 3)
        ]

        citation_count = sum(
            len(re.findall(p, answer, re.IGNORECASE)) for p in citation_patterns
        )

        if citation_count >= 2:
            return 1.0
        elif citation_count == 1:
            return 0.75
        elif "[" in answer and "]" in answer:
            return 0.6
        return 0.3

    def get_hallucination_score(self, answer: str, context: str) -> Dict:
        """
        Comprehensive hallucination detection combining multiple methods.
        Returns score from 0 (no hallucination) to 1 (full hallucination).
        """
        scores = []
        details = {}

        # Method 1: Citation grounding
        citation_score = self.check_citation_grounding(answer)
        scores.append(1.0 - citation_score)
        details["citation_grounding"] = citation_score

        # Method 2: NLI-based claim verification
        if self.nli_model:
            claims = self.extract_claims(answer)
            if claims:
                claim_scores = [
                    self.check_nli_entailment(c, context) for c in claims[:5]
                ]
                avg_entailment = np.mean(claim_scores)
                scores.append(1.0 - avg_entailment)
                details["nli_entailment"] = avg_entailment
                details["claims_checked"] = len(claims[:5])

        # Method 3: LLM verification (optional, slower)
        if self.llm and len(answer) > 100:
            llm_result = self.check_llm_verification(answer, context)
            if "score" in llm_result:
                scores.append(1.0 - llm_result["score"])
                details["llm_verification"] = llm_result

        # Weighted average (NLI weighted higher if available)
        if len(scores) == 1:
            final_score = scores[0]
        elif len(scores) == 2:
            final_score = scores[0] * 0.3 + scores[1] * 0.7
        else:
            final_score = scores[0] * 0.2 + scores[1] * 0.5 + scores[2] * 0.3

        return {
            "hallucination_score": round(final_score, 3),
            "groundedness_score": round(1.0 - final_score, 3),
            "details": details,
        }


class ResearchBenchmark:
    """
    Automated Quality Control & Experimental Rigor.
    """

    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.hallucination_checker = HallucinationChecker(llm)

    def get_groundedness_score(self, answer, context):
        """Enhanced groundedness using HallucinationChecker."""
        result = self.hallucination_checker.get_hallucination_score(answer, context)
        return result["groundedness_score"]

    def run_strategy_comparison(self, test_queries: List[Dict]):
        # Strategies mapped to weights (semantic, bm25)
        # For 'Re-ranked', we'll use a special flag logic below
        strategies = {
            "BM25 Only": {"weight": 0.0, "rerank": False},
            "Dense Only": {"weight": 1.0, "rerank": False},
            "Hybrid (Ensemble)": {"weight": 0.7, "rerank": False},
            "Re-ranked (Innovation)": {"weight": 0.7, "rerank": True},
        }
        results = []

        print(f"Starting Rigorous Benchmark Across {len(strategies)} Strategies...")

        for name, config in strategies.items():
            print(f"Testing Strat: {name}...")

            # Initialize retriever with specific config for this run
            retriever = HybridRetriever(self.index, use_reranker=config["rerank"])

            for query_item in test_queries:
                q = query_item["q"]

                tic = time.perf_counter()
                try:
                    # Retrieve
                    retrieved_chunks = retriever.retrieve(
                        q, k=5, semantic_weight=config["weight"]
                    )

                    # Generate (Using native chain-like logic)
                    context_text = ""
                    for c in retrieved_chunks:
                        source = c["metadata"].get("source", "Unknown")
                        context_text += f"\n[Source: {source}]\n{c['text']}\n"

                    # Construct prompt for LLM
                    prompt = f"""You are a Senior Legal Analyst. Use the provided context to answer the question.
                    Only use provided context. Cite source.
                    CONTEXT: {context_text}
                    QUESTION: {q}
                    RESPONSE:"""

                    response = self.llm.invoke(prompt)
                    toc = time.perf_counter()

                    latency = toc - tic

                    target_source = query_item.get("source")
                    if target_source:
                        recall = (
                            1.0
                            if any(
                                target_source in c["metadata"].get("source", "")
                                for c in retrieved_chunks
                            )
                            else 0.0
                        )
                    else:
                        recall = 0.0  # Cannot compute recall without labeled source

                    # Compute Hallucination Score (comprehensive check)
                    hallucination_result = (
                        self.hallucination_checker.get_hallucination_score(
                            response, context_text
                        )
                    )
                    groundedness = hallucination_result["groundedness_score"]
                    hallucination = hallucination_result["hallucination_score"]

                    results.append(
                        {
                            "Strategy": name,
                            "Query": q[:50] + "..." if len(q) > 50 else q,
                            "Latency": round(latency, 2),
                            "Recall": recall,
                            "Groundedness": groundedness,
                            "Hallucination": hallucination,
                            "Details": hallucination_result.get("details", {}),
                        }
                    )
                except Exception as e:
                    print(f"Error in {name}: {e}")

        df = pd.DataFrame(results)

        # Summary statistics
        summary = (
            df.groupby("Strategy")
            .agg(
                {
                    "Latency": "mean",
                    "Recall": "mean",
                    "Groundedness": "mean",
                    "Hallucination": "mean",
                }
            )
            .reset_index()
        )

        # Print detailed results
        print("\n📊 Detailed Hallucination Analysis:")
        for _, row in df.iterrows():
            print(f"  [{row['Strategy']}] {row['Query']}")
            print(
                f"    → Hallucination: {row['Hallucination']:.2f} | Groundedness: {row['Groundedness']:.2f}"
            )
            if row.get("Details"):
                for k, v in row["Details"].items():
                    if k != "llm_verification":
                        print(f"      • {k}: {v}")

        return summary, df
