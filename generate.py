from mlx_lm import load
from mlx_lm.utils import get_model_path, load_tokenizer
from context_cache_llm import ContextCachingLLM
import pdb

def main():
    # Load the model and tokenizer
    # model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    # model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    # model, tokenizer = load("mlx-community/Qwen2-7B-Instruct-4bit")
    # model, tokenizer = load("mlx-community/Qwen2-1.5B-4bit")
    model, tokenizer = load("mlx-community/NeuralDaredevil-8B-abliterated-4bit") # this has bad tokenizer
    tokenizer = load_tokenizer(get_model_path("mlx-community/Meta-Llama-3-8B-Instruct-4bit"))

    # Initialize the ContextCachingLLM
    llm_session = ContextCachingLLM(model, tokenizer, verbose_time=True)

    # Prepare the context (this can be done when the document is loaded)
    with open("large_doc.txt", "r") as f:
        doc = f.read()
    
    system_prompt = f"""You are a helpful assistant. The user has selected some content from a website given below as CONTEXT. 
Please answer the user's questions related to that. Some key points:
- Always give a direct answer without any prefix or disclaimer.
- User prefers shorter, to the point answers.

CONTEXT:{doc}"""
    
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
        # Ask the user for a question
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        # Generate the answer
        llm_session.add_message(question, role="user", update_cache=False)
        print("Generated response:")
        for segment, hidden_states in llm_session.stream_generate(temp=0.7):
            print(segment, end="", flush=True)
        print()
        turn += 1

    pdb.set_trace()

if __name__ == "__main__":
    main()