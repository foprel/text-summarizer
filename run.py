from app import Smmry
import io

text = io.open('article_1.txt', 'r', encoding='utf-8').read()
smmry = Smmry(text, lang="english")
smmry.summarize(length=2)