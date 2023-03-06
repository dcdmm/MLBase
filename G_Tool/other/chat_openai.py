import openai

openai.api_key = "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"

'''
temperature number Optional Defaults to 1
    What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    We generally recommend altering this or top_p but not both.

top_p number Optional Defaults t    
    An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    We generally recommend altering this or temperature but not both.
'''

# 单轮对话
def generate_answer(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    res_msg = completion.choices[0].message
    return res_msg["content"].strip()


if __name__ == '__main__':
    # 维护一个列表用于存储多轮对话的信息
    messages = [{"role": "system", "content": "你现在是一个体检医生"}]
    while True:
        prompt = input("请输入你的问题:")
        messages.append({"role": "user", "content": prompt})
        res_msg = generate_answer(messages)
        messages.append({"role": "assistant", "content": res_msg})
        print(res_msg)
