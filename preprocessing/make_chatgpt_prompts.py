from openai import OpenAI
import pandas as pd

client = OpenAI()
cleaned_prompts = []

df = pd.read_csv("svo_probes.csv")

for prompt in df["clean_prompt"]:
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "Make the sentence grammatically correct. I want the order of the tokens to be minimally disturbed."
        },
        {
          "role": "user",
          "content": prompt
        }
      ],
      temperature=0.7,
      max_tokens=42, 
      top_p=1
    )

    output=response.choices[0].message.content
    cleaned_prompts.append(output)

print("PROMPTS IN {} DONE".format(p_range))
df = pd.DataFrame(cleaned_prompts, columns=['GPT-3.5 Prompts'])
df.to_csv('gpt3.5_prompts.csv', index=False)
