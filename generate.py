from mlx_lm import load
from context_cache_llm import ContextCachingLLM

def main():
    # Load the model and tokenizer
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

    # Initialize the ContextCachingLLM
    context_caching_llm = ContextCachingLLM(model, tokenizer)

    # Prepare the context (this can be done when the document is loaded)
    with open("large_doc.txt", "r") as f:
        doc = f.read()
    
    system_prompt = "You are a helpful assistant. The user has selected some content from a website given as CONTEXT. Please answer the user's question. Always give a direct answer without any prefix or disclaimer."

    print("Starting to prepare context...")
    context_caching_llm.prepare_context(system_prompt, doc)

    print("Context prepared.")
    
    while True:
        # Ask the user for a question
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        # Generate the answer
        response = context_caching_llm.generate(question, verbose_time=True, temp=0.7)
        print("Generated response:")
        print(response)

if __name__ == "__main__":
    main()