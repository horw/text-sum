import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='1/bpe.model')

# Токенизируем текст
tokens = sp.encode('Чувак, гениально. Я с тобой. Только надо придумать как быть с холодцом который я вчера наварил.', out_type=str)
print(tokens)