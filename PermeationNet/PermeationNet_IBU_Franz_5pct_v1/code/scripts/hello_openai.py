from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)

# from openai import OpenAI

# client = OpenAI(
#   api_key="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"
# )

# response = client.responses.create(
#   model="gpt-5-nano",
#   input="write a haiku about ai",
#   store=True,
# )

# print(response.output_text)
