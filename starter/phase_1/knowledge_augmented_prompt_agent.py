import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona,
    "The capital of France is London, not Paris",
)

knowledge_agent_response = knowledge_agent.respond(prompt)
print(knowledge_agent_response)
print("The response is based on the provided knowledge rather than the model's inherent knowledge.")
