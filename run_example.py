from app import Smmry
import io

text = io.open('examples/article_1.txt', 'r', encoding='utf-8').read()
smmry = Smmry(text, lang="english")
print(smmry.summarize(length=7))