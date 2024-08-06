from mlx_lm import load
import mlx.core as mx


from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union
import time

# Local imports
from mlx_lm import load
import mlx.core as mx

from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union
import time

# Local imports
from mlx_lm.models.base import KVCache
from mlx_lm.sample_utils import top_p_sampling
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer

class ContextCachingLLM:
    def __init__(self, model, tokenizer, verbose_time=False, 
                 hidden_layer=20,
                 save_hidden_states=True,
                 context_marker="```"):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = None
        self.verbose_time = verbose_time
        self.messages = []
        self.hidden_layer = hidden_layer
        self.hidden_states = []
        self.save_hidden_states = save_hidden_states
        self.model.model.output_hidden_states = True
        self.model.model.output_hidden_layer = hidden_layer
        self.context_text = None
        self.context_text_parts = None
        self.context_hidden_states = None
        self.context_offset_mapping = None
        self.context_marker = context_marker

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

    def add_to_last_message(self, message, update_cache=False):
        """
        Add to the last message in the context.
        """
        if len(self.messages) == 0:
            raise ValueError("No messages to add to.")
        self.messages[-1]["content"] += message
        if update_cache:
            self._update_cache()

    def update_last_message(self, message, update_cache=False):
        """
        Update the last message in the context.
        """
        if len(self.messages) == 0:
            raise ValueError("No messages to update.")
        self.messages[-1]["content"] = message
        if update_cache:
            self._update_cache()

    def reset_messages(self):
        """
        Reset the messages in the context.
        """
        self.messages = []
        self.cache = None
        self.hidden_states = []
        self.context_text = None
        self.context_hidden_states = None
        self.context_text_parts = None
        self.context_offset_mapping = None
        mx.metal.clear_cache()

    def _update_cache(self):
        """
        Update the KV cache with the current messages and store hidden states.
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

        start_time = time.time()
        prefix_tokens = mx.array(self.tokenizer.encode(prefix)).reshape(1, -1)
        logits, hidden_states = self.model(prefix_tokens, cache=self.cache,)
        mx.eval(logits)  # needed since mlx is lazy execution
        # mx.eval(hidden_states)  # needed since mlx is lazy execution
        end_time = time.time()
        
        time_taken = end_time - start_time

        # Store hidden states for the specified layer
        # We need to do this only for the 'context' part for now
        if self.save_hidden_states:
            self._update_hidden_states(hidden_states)

        if self.verbose_time:
            # print time taken to update cache, and token length
            print(f"Time taken to update cache: {time_taken:.3f} seconds for {len(prefix_tokens[0])} tokens.")

    def _update_hidden_states(self, hidden_states):
        """
        Store hidden states for the context part.
        """
        prefix = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=False
        )

        start_char_index = prefix.find(self.context_marker) + len(self.context_marker)
        end_char_index = prefix.rfind(self.context_marker)
        
        tokenizer_output = self.tokenizer.encode_plus(prefix, return_offsets_mapping=True)

        offset_mapping = fix_offsets(tokenizer_output["offset_mapping"])

        start_token_index = char_to_token_index(start_char_index, offset_mapping) or 0
        end_token_index = char_to_token_index(end_char_index, offset_mapping) or len(offset_mapping) - 1

        self.context_hidden_states = hidden_states[0][0][start_token_index:end_token_index, :]
        # get context text parts using offset mapping
        self.context_text_parts = [prefix[start:end] for start, end in offset_mapping[start_token_index:end_token_index]]
        self.context_text = "".join(self.context_text_parts)
        self.context_offset_mapping = [
            (max(0, start - start_char_index), max(0, end - start_char_index))
            for start, end in offset_mapping[start_token_index:end_token_index]
        ]
        print("Hidden states updated with shape:", self.context_hidden_states.shape)
        # print context lengths in characters and tokens
        print(f"Context length: {len(self.context_text)} characters, {len(self.context_text_parts)} tokens.")



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
    

    def stream_generate(
        self,
        max_tokens: int = 100,
        **kwargs,
    ) -> Generator[Tuple[str, mx.array], None, None]:
        
        if self.cache is None:
            raise ValueError("Context not prepared. Call prepare_context first.")
        
        full_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not isinstance(self.tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(self.tokenizer)
        else:
            tokenizer = self.tokenizer

        prompt_tokens = mx.array(tokenizer.encode(full_prompt))
        detokenizer = tokenizer.detokenizer

        detokenizer.reset()
        generated_hidden_states = []
        self.add_message("", role="assistant", update_cache=False) # cache is already updated

        for (token, _, token_hidden_state), n in zip(
            self._generate_step(prompt_tokens, **kwargs),
            range(max_tokens),
        ):
            if token == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token)
            generated_hidden_states.append(token_hidden_state[0][-1])
            
            self.update_last_message(detokenizer.text, update_cache=False)
            # Yield the last segment and current hidden states if streaming
            yield detokenizer.last_segment, None

        detokenizer.finalize()
        # all_hidden_states = mx.concatenate([self.hidden_states, mx.stack(generated_hidden_states)], axis=0) # dunno why this here but not deleting
        generated_hidden_states.append(token_hidden_state[0][-1])
        self.update_last_message(detokenizer.text, update_cache=False)
        yield detokenizer.last_segment, mx.stack(generated_hidden_states)

    def get_context_attribution(self, hidden_states):
        """
        Input: hidden_states of the generated text. Shape: (num_tokens, hidden_dim)
        Output: attribution of each token in the generated text to the context. Shape: (num_tokens, num_context_tokens)
        The attribution is calculated using the cosine similarity between the hidden states of the generated text and the context.
        """
        # # Ensure hidden_states is a 2D array
        # if hidden_states.ndim == 1:
        #     hidden_states = hidden_states.reshape(1, -1)
        
        # Normalize the hidden states
        generated_norm = mx.linalg.norm(hidden_states, axis=1, keepdims=True)
        context_norm = mx.linalg.norm(self.context_hidden_states, axis=1, keepdims=True)

        # print shapes of norms
        # print(generated_norm.shape, context_norm.shape)
        
        # Compute cosine similarity
        similarity = (hidden_states @ self.context_hidden_states.T) / (generated_norm * context_norm.T)
        
        # Normalize the similarity scores
        # attribution = mx.softmax(similarity, axis=1)
        attribution = similarity
        
        return attribution

    def _generate_step(
        self,
        prompt: mx.array,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array, mx.array], None, None]:
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
        start_index = self.cache[0].offset

        repetition_context = prompt[start_index:].tolist()

        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]

        def _step(y):
            nonlocal repetition_context
            logits, hidden_state = self.model(y[None], cache=self.cache)
            logits = logits[:, -1, :]

            y, logprobs = sample(logits)

            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            return y, logprobs.squeeze(0), hidden_state[-1]  # Return the last token's hidden state # TODO: this ain't last token thing?

        # # Process new tokens (after the prefix)
        # for i in range(start_index, len(y)):
        #     _, _, _ = _step(y[i:i+1])
        # y, logprobs, hidden_state = _step(y[-1:])

        y, logprobs, hidden_state = _step(y[start_index:])

        mx.async_eval(y)
        while True:
            next_y, next_logprobs, next_hidden_state = _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs, hidden_state
            y, logprobs, hidden_state = next_y, next_logprobs, next_hidden_state


def fix_offsets(offsets):
    fixed_offsets = []
    for i, (start, end) in enumerate(offsets):
        if i == 0:
            fixed_offsets.append((start, end))
        else:
            prev_start, prev_end = fixed_offsets[-1]
            
            # If there's a gap, extend the previous token
            if prev_end < start:
                fixed_offsets[-1] = (prev_start, start)
            
            # Add the current token as is, preserving zero-width tokens
            fixed_offsets.append((start, end))
    
    return fixed_offsets

def char_to_token_index(char_index, offset_mapping):
    for token_index, (start, end) in enumerate(offset_mapping):
        if start <= char_index < end:
            return token_index
    return None  # Return None if the character index is out of bounds
