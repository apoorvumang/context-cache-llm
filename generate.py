from mlx_lm import load
from context_cache_llm import ContextCachingLLM

def main():
    # Load the model and tokenizer
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

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
    
    print("Starting to prepare context...")
    llm_session.add_message(system_prompt, role="system", update_cache=True)
    print("Context prepared.")
    
    while True:
        # Ask the user for a question
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        # Generate the answer
        llm_session.add_message(question, role="user", update_cache=False)
        response = llm_session.generate(temp=0.7)
        print("Generated response:")
        print(response)

if __name__ == "__main__":
    main()