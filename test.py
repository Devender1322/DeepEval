# # import openai
# from openai import OpenAI
import os
# import openai

# # Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
# client = OpenAI()
# prompt_template = "You are the Healthcare Assistant, a helpful AI assistant. Your task is to provide test cases for Healthcare system."
# messagess=[{"role": "user", "content": prompt_template}]
# def generate_test_cases(max_test_cases=5):
#     test_cases = []

#     # Generate test cases using OpenAI's API
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=messagess,
#         # max_tokens=1024,
#         n=max_test_cases,
#         temperature=0,
#         max_tokens=150,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )

#     # Extract generated test cases from OpenAI's response
#     for choice in response.choices:
#         test_cases.append(choice.text.strip())

#     return response.choices[0].text.strip()

# # Example prompt template


# test_cases = generate_test_cases(prompt_template)

# # Print generated test cases
# for i, test_case in enumerate(test_cases, 1):
#     print(f"Test Case {i}: {test_case}")



from langchain_openai import AzureOpenAI

llm = AzureOpenAI(
    deployment_name="text-davinci-002-prod",
    model_name="gpt-3.5-turbo",
    api_version="2023-05-15",
)

test_case = "You are the Healthcare Assistant, a helpful AI assistant. Your task is to provide test cases for Healthcare system."

generated_test_cases = llm(test_case)
print(generated_test_cases)