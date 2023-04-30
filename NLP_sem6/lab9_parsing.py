import spacy
parse = spacy.load("en_core_web_sm")
mytext = "Magic lies in my mind but my blades follows my heart"
doc = parse(mytext)
for token in doc:
    print(
        f"""
        TOKEN: {token.text}
        ~~~~~~~~~~~~~~~~~~~
        TAG = {token.tag}
        HEAD_TEXT = {token.head.text}
        TEXT_DEP = {token.dep}
        """
    )

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
mytext = "Magic lies in my mind but my blades follows my heart"

output = nlp.annotate(mytext, properties={
  'annotators': 'parse',
  'outputFormat': 'json'
})

print(output['sentences'][0]['parse'])

from nltk.parse.corenlp import CoreNLPParser

parser = CoreNLPParser()
parse = next(parser.raw_parse(mytext))
print(parse)

