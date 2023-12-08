from transformers import MBartTokenizer, MBartForConditionalGeneration

model_name = "IlyaGusev/mbart_ru_sum_gazeta"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

article_text = '''
Чувак, гениально. Я с тобой. Только надо придумать как быть с холодцом который я вчера наварил.
'''

input_ids = tokenizer(
    [article_text],
    max_length=600,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

print(input_ids)

# output_ids = model.generate(
#     **input_ids,
#     max_new_tokens=5,
#     num_beams=4,
#     num_return_sequences=4,
# )[0]
#
# summary = tokenizer.decode(output_ids, skip_special_tokens=True)
# print(summary)
