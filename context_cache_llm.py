from mlx_lm import load
import mlx.core as mx


from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union
import time

# Local imports
from mlx_lm.models.base import KVCache
from mlx_lm.sample_utils import top_p_sampling
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer

class ContextCachingLLM:
    def __init__(self, model, tokenizer, verbose_time=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = None
        self.verbose_time = verbose_time
        self.messages = []
        self.prefix_tokens = mx.array([])

    def prepare_context(self):
        """
        Prepare and cache the context - whatever is in messages till now
        """
        self._update_cache()


    def add_message(self, 
                    message,
                    role = "user",
                    update_cache=False):
        """
        Add a user message to the context.
        """
        if role not in ["user", "system", "assistant"]:
            raise ValueError("Role should be one of 'user', 'system', 'assistant'")
        
        self.messages.append({"role": role, "content": message})
        if update_cache:
            self._update_cache()

    def reset_messages(self):
        """
        Reset the messages in the context.
        """
        self.messages = []
        self.cache = None
        self.prefix_tokens = mx.array([])

    def _update_cache(self):
        """
        Update the KV cache with the current messages.
        """
        if self.cache is None:
            kv_heads = (
                [self.model.n_kv_heads] * len(self.model.layers)
                if isinstance(self.model.n_kv_heads, int)
                else self.model.n_kv_heads
            )
            self.cache = [KVCache(self.model.head_dim, n) for n in kv_heads]
            
        prefix = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=False
        )
        self.prefix_tokens = mx.array(self.tokenizer.encode(prefix))

        start_time = time.time()
        prefix_tokens = mx.array(self.tokenizer.encode(prefix)).reshape(1, -1)
        logits = self.model(prefix_tokens, cache=self.cache)
        mx.eval(logits) # needed since mlx is lazy execution
        end_time = time.time()
        
        time_taken = end_time - start_time

        if self.verbose_time:
            # print time taken to update cache, and token length
            print(f"Time taken to update cache: {time_taken:.3f} seconds for {len(prefix_tokens[0])} tokens.")


    def generate(
        self,
        max_tokens: int = 100,
        temp: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        logit_bias: Optional[Dict[int, float]] = None,
        verbose: bool = False,
        formatter: Optional[Callable] = None,
    ) -> str:
        """
        Generate an answer using the cached context and the provided question.
        """
        if self.cache is None:
            raise ValueError("Context not prepared. Call prepare_context first.")

        # Replace the placeholder with the actual question
        
        full_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
        if self.verbose_time:
            gen_time = time.perf_counter() - tic
            prompt_tps = prompt_tokens.size / prompt_time
            gen_tps = (token_count - 1) / gen_time
            print(f"Prompt: {prompt_tps:.3f} tokens-per-sec, {prompt_time:.3f} prompt time, {len(prompt_tokens)} tokens.")
            print(f"Generation: {gen_tps:.3f} tokens-per-sec, {gen_time:.3f} generation time, {token_count} tokens.")

        self.add_message(detokenizer.text, role="assistant", update_cache=False) # cache is already updated

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