from src.langgraphagenticai.state.state import State
import unicodedata

class BasicChatbotNode:
    def __init__(self, model):
        self.llm = model

    def __call__(self, state: State) -> dict:
        # Step 1: Get the raw response from the LLM
        response = self.llm.invoke(state['messages'])

        # Step 2: Debug print the raw output to confirm what's inside
        print("🛠 DEBUG: Raw LLM response repr ->", repr(response))

        # Step 3: Normalize and strip all non-ASCII (like •, emojis, smart quotes)
        clean_response = unicodedata.normalize("NFKD", response).encode("ascii", "ignore").decode("ascii")

        # Step 4: Optional debug to verify cleaned output
        print("✅ Cleaned response repr ->", repr(clean_response))

        return {"messages": clean_response}
