import sentencepiece as spm

# Обучаем модель BPE
spm.SentencePieceTrainer.train('--input=lenta-ru-news.csv --pad_id=0 --bos_id=-1 --eos_id=1 --unk_id=2 \
                                --model_prefix=bpe --vocab_size=16000 --model_type=bpe')