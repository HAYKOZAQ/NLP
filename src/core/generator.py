import os
from google import genai
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class GeminiGenerator:
    """
    Google Gemini Generator using official google-genai SDK.
    Primary model: gemini-3-flash-preview, fallback: gemma-3-27b-it
    """

    def __init__(
        self, model_id="gemini-3-flash-preview", fallback_model_id="gemma-3-27b-it"
    ):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id

    def _generate(self, model: str, prompt_text: str) -> str:
        """Internal method to generate content with a specific model."""
        response = self.client.models.generate_content(
            model=model,
            contents=prompt_text,
            config={"temperature": 0.3},
        )
        return response.text

    def invoke(self, prompt_text: str) -> str:
        # Try primary model first
        try:
            print(f"📡 Connecting to Gemini [{self.model_id}]...")
            return self._generate(self.model_id, prompt_text)

        except Exception as e:
            print(f"⚠ Primary model failed: {e}")
            print(f"🔄 Falling back to [{self.fallback_model_id}]...")

            # Try fallback model
            try:
                return self._generate(self.fallback_model_id, prompt_text)

            except Exception as fallback_e:
                return f"Both models failed. Primary ({self.model_id}): {e} | Fallback ({self.fallback_model_id}): {fallback_e}"


def get_generator(
    model_id="gemini-3-flash-preview", fallback_model_id="gemma-3-27b-it"
):
    return GeminiGenerator(model_id=model_id, fallback_model_id=fallback_model_id)


def create_rag_chain(
    retriever,
    llm,
    hallucination_checker=None,
    groundedness_threshold: float = 0.2,
    max_retries: int = 2,
):
    """
    RAG logic with automatic retry for hallucination handling.
    """

    class RAGChain:
        def __init__(
            self,
            retriever,
            llm,
            hallucination_checker,
            groundedness_threshold,
            max_retries,
        ):
            self.retriever = retriever
            self.llm = llm
            self.hallucination_checker = hallucination_checker
            self.groundedness_threshold = groundedness_threshold
            self.max_retries = max_retries
            self.template = """### IDENTITY
You are Orin AI Pro, an expert Legal Assistant specializing in Armenian Judiciary and Regulatory Law.
Your mission is to provide accurate, strictly grounded legal advice based ONLY on the provided context.

### INSTRUCTIONS

1. **LANGUAGE PROTOCOL**:
   - You MUST answer all queries in **Armenian** (Հայերեն).
   - Do not tell the user how you have obtained the information just ansewr directly.
   - Do not use Translit. Do not translate legal terms loosely.
   - If the user asks in English/Russian, reply in Armenian but you may include a short parenthesis translation if helpful.
   - Do not try to tell information that wasn't asked about or wasn't provided in the context.
   - Only talk about details that are asked don't talk about other laws that aren't directly relevant in order to keep your response comprehensive

2. **LEGAL GROUNDING (CRITICAL)**:
   - **CITE SOURCES**: Every key legal claim must be backed by a reference to the source document (e.g., [Աղբյուր: ՀՀ Քրեական Օրենսգիրք, Հոդված X Մաս Y]). Always put citation after the armenian period.
   - **TEMPORAL AWARENESS**: Check the filenames/titles in the context. If you see years (e.g., 2014, 2018), explicitly mention the validity period if relevant. Prioritize newer laws if multiple versions exist.

3. **RESPONSE STRUCTURE**:
   - Start with a direct answer.
   - Use bullet points for lists of conditions or rules.
   - If the Context does not contain the answer, you may state: "Ինձ տրամադրված իրավական փաստաթղթերում ուղղակի պատասխանը չկա, սակայն..." (The direct answer is not in documents, however...) and provide closely related context if available. If truly nothing is relevant, state "Տրամադրված փաստաթղթերում այս հարցի պատասխանը բացակայում է:"

4. **NON-LEGAL QUERIES**:
   - If the query is completely unrelated to law (e.g., cooking), politely refuse.
   - If the query is general but related (e.g., "What is a contract?"), verify if context helps. If not, give a general definition but clarify it's general knowledge.

### KNOWLEDGE CONTEXT
{context}

Question:
{question}

Answer:
"""

        def invoke(self, question: str) -> str:
            # 1. Retrieve
            docs = self.retriever.invoke(question)

            # 2. Format
            formatted_context = ""
            raw_context = ""
            for d in docs:
                source = d.get("metadata", {}).get("source", "Unknown")
                text = d.get("text", "")
                raw_context += text + "\n"
                if source.endswith(".pdf"):
                    source = source[:-4]
                formatted_context += f"\n[Աղբյուր: {source}]\n{text}\n"

            # print(formatted_context)  # Disabled to prevent UnicodeEncodeError on Windows console

            # 3. Generate with hallucination retry
            base_prompt = self.template.format(
                context=formatted_context, question=question
            )

            response = None
            for attempt in range(self.max_retries + 1):
                full_prompt = base_prompt
                if attempt > 0:
                    # Add stricter grounding instruction on retry
                    full_prompt += "\n\nCRITICAL: Your previous response was not well-grounded. Base your answer STRICTLY on the provided context. Do not add any information not explicitly found in the sources above."

                response = self.llm.invoke(full_prompt)

                # Check groundedness if checker available
                if self.hallucination_checker and raw_context:
                    h_result = self.hallucination_checker.get_hallucination_score(
                        response, raw_context
                    )
                    groundedness = h_result.get("groundedness_score", 1.0)

                    if groundedness >= self.groundedness_threshold:
                        if attempt > 0:
                            print(
                                f"Retry {attempt} passed groundedness check ({groundedness:.2f})"
                            )
                        return response

                    print(
                        f"Low groundedness ({groundedness:.2f}) on attempt {attempt + 1}/{self.max_retries + 1}"
                    )
                else:
                    # No checker, return immediately
                    return response

            # Return last response even if groundedness is low
            print("Returning response despite low groundedness (max retries reached)")
            return response

    return RAGChain(
        retriever, llm, hallucination_checker, groundedness_threshold, max_retries
    )
