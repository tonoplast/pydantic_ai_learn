from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.agent import Agent
from pydantic_ai.providers.openai import OpenAIProvider

ollama_model = OpenAIModel(
    model_name="llama3.2:latest",
    provider=OpenAIProvider(
        base_url="https://localhost:11434/v1",
    )
)


agent=Agent(
    model=ollama_model,
    system_prompt=['Reply in one sentence']
)

response = agent.run_sync('What is the capital of Korea?')
print(response.data)