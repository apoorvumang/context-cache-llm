import numpy as np
from scipy.signal import find_peaks
from mlx_lm import load
from mlx_lm.utils import get_model_path, load_tokenizer
from context_cache_llm import ContextCachingLLM
import pdb
import json
from scipy.signal import convolve2d
import json


def print_attribution(segments):    
    print(f"Found {len(segments)} diagonal segments:")
    for i, ((answer_start, answer_end), (doc_start, doc_end), score) in enumerate(segments):
        print(f"Segment {i+1}: Answer tokens {answer_start}-{answer_end}, "
              f"Document characters {doc_start}-{doc_end}, "
              f"Length: {answer_end - answer_start + 1}, "
              f"Score: {score:.4f}")
        

def get_attribution_json_dict(segments):
    attribution_dict = []
    for i, ((answer_start, answer_end), (doc_start, doc_end), score) in enumerate(segments):
        attribution_dict.append({
            "answer_tokens": [int(answer_start), int(answer_end)],
            "document_chars": [int(doc_start), int(doc_end)],
            "score": round(float(score),3)
        })
    return attribution_dict


def save_attribution_data(attribution, context_text_parts, generated_text_parts, output_file):
    # transpose attribution
    attribution = np.transpose(attribution, (1, 0))
    data = {
        "attribution": attribution.tolist(),  # Convert numpy array to list
        "context_tokens": context_text_parts,
        "generated_tokens": generated_text_parts
    }
    with open(output_file, 'w') as f:
        json.dump(data, f)

# doc_fname = "large_doc.txt"
doc_fname = "small_doc.txt"

def main():
    # Load the model and tokenizer
    # model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    # model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    # model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")
    # model, tokenizer = load("mlx-community/Qwen2-1.5B-4bit")
    # model, tokenizer = load("mlx-community/NeuralDaredevil-8B-abliterated-4bit") # this has bad tokenizer
    # tokenizer = load_tokenizer(get_model_path("mlx-community/Meta-Llama-3-8B-Instruct-4bit"))

    # model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

    # Initialize the ContextCachingLLM
    llm_session = ContextCachingLLM(model, tokenizer, verbose_time=True)

    # Prepare the context (this can be done when the document is loaded)
    with open(doc_fname, "r") as f:
        doc = f.read()
    
    system_prompt = f"""You are a helpful assistant. The user has selected some content from a website given below as CONTEXT. 
Please answer the user's questions related to that. Some key points:
- Always give a direct answer without any prefix or disclaimer.
- User prefers shorter, to the point answers.

CONTEXT:
```{doc}```
"""
    
#     system_prompt = f"""You are a helpful assistant. I have selected some content from a website given below as CONTEXT. 
# Please answer the my questions related to that. Some key points:
# - Always give a direct answer without any prefix or disclaimer.
# - I prefer shorter, to the point answers.

# CONTEXT:{doc}\n\n"""
    
    print("Starting to prepare context...")
    llm_session.add_message(system_prompt, role="system", update_cache=True)
    # llm_session.add_message(system_prompt, role="user", update_cache=True)
    print("Context prepared.")
    
    turn = 0
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        llm_session.add_message(question, role="user", update_cache=False)
        print("Generated response:")
        generated_text_parts = []
        for segment, attr_output in llm_session.stream_generate(return_similarity_matrix=False,temp=0.001):
            print(segment, end="", flush=True)
            generated_text_parts.append(segment)
            if attr_output is not None:
                # similarity_matrix, attribution_segments = attr_output
                attribution_segments = attr_output
                print(attribution_segments)
                json_out = json.dumps(get_attribution_json_dict(attribution_segments))
                print(json_out)
                print_attribution(attribution_segments)
                # save_attribution_data(np.array(similarity_matrix), llm_session.context_text_parts, generated_text_parts[:-1], 'attribution_data.json') # last token - we don't have hidden state so no attr
                
        print()
        turn += 1

    pdb.set_trace()

if __name__ == "__main__":
    main()


# OLD CODE FOR APPLYING FILTERS, MIGHT NOT BE NEEDED


# def create_diagonal_filter(size=5, diagonal_range=2):
#     """
#     Create a diagonal filter matrix.
    
#     :param size: Size of the square filter matrix
#     :param diagonal_range: Range of positive values around the main diagonal
#     :return: NumPy array representing the filter
#     """
#     filter_matrix = np.zeros((size, size))
#     for i in range(size):
#         for j in range(size):
#             if abs(i - j) <= diagonal_range:
#                 filter_matrix[i, j] = diagonal_range - abs(i - j) + 1
#             else:
#                 filter_matrix[i, j] = -1
#     return filter_matrix

# def apply_diagonal_filter(matrix, filter_size=5, diagonal_range=2):
#     """
#     Apply a diagonal filter to the input matrix.
    
#     :param matrix: Input matrix to be filtered
#     :param filter_size: Size of the square filter matrix
#     :param diagonal_range: Range of positive values around the main diagonal
#     :return: Filtered matrix
#     """
#     diagonal_filter = create_diagonal_filter(filter_size, diagonal_range)
    
#     # Normalize the filter
#     diagonal_filter = diagonal_filter / np.sum(np.abs(diagonal_filter))
    
#     # Apply convolution with 'same' mode to preserve shape
#     filtered_matrix = convolve2d(matrix, diagonal_filter, mode='same', boundary='wrap')
    
#     return filtered_matrix

# def apply_complex_diagonal_filter(matrix, filter_size=5, diagonal_range=2, off_diagonal_value=-1):
#     """
#     Apply a complex diagonal filter to the input matrix.
    
#     :param matrix: Input matrix to be filtered
#     :param filter_size: Size of the square filter matrix
#     :param diagonal_range: Range of positive values around the main diagonal
#     :param off_diagonal_value: Value for elements off the diagonal range
#     :return: Filtered matrix
#     """
#     complex_filter = create_diagonal_filter(filter_size, diagonal_range)
#     complex_filter[complex_filter == -1] = off_diagonal_value
    
#     # Normalize the filter
#     complex_filter = complex_filter / np.sum(np.abs(complex_filter))
    
#     # Apply convolution with 'same' mode to preserve shape
#     filtered_matrix = convolve2d(matrix, complex_filter, mode='same', boundary='wrap')
    
#     return filtered_matrix
