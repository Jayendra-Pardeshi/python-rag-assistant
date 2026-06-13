from langchain_core.prompts import ChatPromptTemplate

def get_qa_prompt():
    system_prompt = (
        "You are an expert Python Programming Tutor. Your goal is to help learners "
        "solve coding problems using the provided context from Stack Overflow.\n"
        "Context Rules:\n"
        "1. Use ONLY the provided context to answer.\n"
        "2. If the context doesn't contain the answer, admit it clearly.\n"
        "3. Always provide code snippets if available in the context.\n"
        "4. Keep the explanation concise and beginner-friendly.\n"
        "\n"
        "Context:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return prompt