

# 비슷한단어 set으로 반환
def find_similar_words(word):
    try:
        from nltk.corpus import wordnet
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        similar_words = set()
        synsets = wordnet.synsets(word)

        for synset in synsets:
            for lemma in synset.lemmas():
                similar_words.add(lemma.name())
        return similar_words
    except:
        print("nltk wordnet 오류 -> 유사 단어를 검색하지 않습니다")
        return list()



