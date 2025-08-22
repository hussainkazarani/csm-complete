# test_vllm.py
from llm_interface import LLMInterface

# Try to initialize with your GGUF path
try:
    llm = LLMInterface("./models/llama32-1b.gguf") # Use your exact path
    print("SUCCESS: LLM loaded with GGUF path (This shouldn't happen!)")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
