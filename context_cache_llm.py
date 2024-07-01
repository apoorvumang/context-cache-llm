from mlx_lm import load
import mlx.core as mx


from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union
import time

# Local imports
from mlx_lm.models.base import KVCache
from mlx_lm.sample_utils import top_p_sampling
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer

class ContextCachingLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = None
        self.prefix_tokens = None
        self.full_prompt_template = None

    def prepare_context(self, system_prompt, document_context):
        """
        Prepare and cache the context (system prompt and document).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT: {document_context}\nQUESTION: {{{{user_question}}}}"},
        ]
        self.full_prompt_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Find the position of the placeholder
        placeholder_position = self.full_prompt_template.find("{{user_question}}")
        if (placeholder_position == -1):
            raise ValueError("Placeholder {{user_question}} not found in the prompt template")
        
        # Get the prefix (everything before the placeholder)
        prefix = self.full_prompt_template[:placeholder_position]
        
        self.prefix_tokens = mx.array(self.tokenizer.encode(prefix))
        
        # Measure the time taken by self._process_prefix
        start_time = time.time()
        self.cache = self._process_prefix(self.prefix_tokens)
        end_time = time.time()
        
        time_taken = end_time - start_time
        print("Time taken to process prefix:", time_taken)
        print("Prefix token length", len(self.prefix_tokens))


    def _process_prefix(self, prefix_tokens):
        """
        Process the prefix tokens and return the KV cache.
        """
        kv_heads = (
            [self.model.n_kv_heads] * len(self.model.layers)
            if isinstance(self.model.n_kv_heads, int)
            else self.model.n_kv_heads
        )
        cache = [KVCache(self.model.head_dim, n) for n in kv_heads]
        
        # Add batch dimension
        prefix_tokens = prefix_tokens.reshape(1, -1)
        
        logits = self.model(prefix_tokens, cache=cache)

        mx.eval(logits)
        
        return cache


    def generate(
        self,
        question: str,
        max_tokens: int = 100,
        temp: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        logit_bias: Optional[Dict[int, float]] = None,
        verbose: bool = False,
        verbose_time: bool = False,
        formatter: Optional[Callable] = None,
    ) -> str:
        """
        Generate an answer using the cached context and the provided question.
        """
        if self.cache is None or self.full_prompt_template is None:
            raise ValueError("Context not prepared. Call prepare_context first.")

        # Replace the placeholder with the actual question
        full_prompt = self.full_prompt_template.replace("{{user_question}}", question)
        
        if not isinstance(self.tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(self.tokenizer)
        else:
            tokenizer = self.tokenizer

        if verbose:
            print("=" * 10)
            print("Prompt:", full_prompt)

        prompt_tokens = mx.array(tokenizer.encode(full_prompt))
        detokenizer = tokenizer.detokenizer

        tic = time.perf_counter()
        detokenizer.reset()

        for (token, logprobs), n in zip(
            self._generate_step(prompt_tokens, temp, repetition_penalty, repetition_context_size, top_p, logit_bias),
            range(max_tokens),
        ):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            if token == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token)

            if verbose:
                if formatter:
                    detokenizer.finalize()
                    formatter(detokenizer.last_segment, mx.exp(logprobs[token]).item())
                else:
                    print(detokenizer.last_segment, end="", flush=True)

        token_count = n + 1
        detokenizer.finalize()

        if verbose:
            print(detokenizer.last_segment, flush=True)
            print("=" * 10)
            if token_count == 0:
                print("No tokens generated for this prompt")
                return ""
        if verbose_time:
            gen_time = time.perf_counter() - tic
            prompt_tps = prompt_tokens.size / prompt_time
            gen_tps = (token_count - 1) / gen_time
            print(f"Prompt: {prompt_tps:.3f} tokens-per-sec, {prompt_time:.3f} prompt time, {len(prompt_tokens)} tokens.")
            print(f"Generation: {gen_tps:.3f} tokens-per-sec, {gen_time:.3f} generation time, {token_count} tokens.")

        return detokenizer.text

    def _generate_step(
        self,
        prompt: mx.array,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """
        A generator producing token ids based on the given prompt from the model.
        """
        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            if logit_bias:
                indices = mx.array(list(logit_bias.keys()))
                values = mx.array(list(logit_bias.values()))
                logits[:, indices] += values
            logprobs = logits - mx.logsumexp(logits)

            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                if top_p > 0 and top_p < 1.0:
                    token = top_p_sampling(logits, top_p, temp)
                else:
                    token = mx.random.categorical(logits * (1 / temp))

            return token, logprobs

        y = prompt
        start_index = self.prefix_tokens.size

        repetition_context = prompt[start_index:].tolist()

        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]

        def _step(y):
            nonlocal repetition_context
            logits = self.model(y[None], cache=self.cache)
            logits = logits[:, -1, :]

            y, logprobs = sample(logits)

            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            return y, logprobs.squeeze(0)

        # Process new tokens (after the prefix)
        for i in range(start_index, len(y)):
            _, _ = _step(y[i:i+1])

        y, logprobs = _step(y[-1:])

        mx.async_eval(y)
        while True:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs