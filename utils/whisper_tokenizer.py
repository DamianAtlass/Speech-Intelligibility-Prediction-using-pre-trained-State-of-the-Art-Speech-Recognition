from whisper.tokenizer import get_tokenizer, Tokenizer

tokenizer: Tokenizer = get_tokenizer(
        multilingual=True,
        language="en",
        task="transcribe",
        num_languages=100)

def text_to_token(s: str):
    return tokenizer.encode(s)