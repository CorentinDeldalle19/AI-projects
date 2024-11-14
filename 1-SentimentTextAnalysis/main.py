import deepl
from textblob import TextBlob
from goose3 import Goose

def translate(text, targetLanguage='EN-US'):
    api_key = '29dcb429-6b03-4b87-bd5d-52aa20732948:fx'
    translator = deepl.Translator(api_key)

    result = translator.translate_text(text, target_lang=targetLanguage)
    return result.text

def analysis(input):
    if input[:8] == "https://":
        g = Goose()
        article = g.extract(url=input)
        text = article.cleaned_text
    else:
        text = input

    text = translate(text, targetLanguage='EN-US')

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

def main():
    print("Hi !")
    print("Enter a text or the URL of an article and I will tell you if it's positive or negative \n")
    x = input()
    y = analysis(x)

    if y > 0:
        print("Globally positive")
    elif y < 0:
        print("Globally negative")
    elif y == 0:
        print("Your text has no meaning")

if __name__ == "__main__":
    main()