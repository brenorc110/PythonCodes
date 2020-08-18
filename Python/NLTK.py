import nltk
#nltk.download();

texto = "I am the one that you want. I'm the one that you will never forget."
print(texto.split[','])

frases = nltk.tokenize.sent_tokenize(texto)
print(frases)

tokens = nltk.word_tokenize(texto)
print(tokens)

# Mostrar palavras(tokens) e classificação gramatical de cada.
#Penn part of speech tags <-BUSCAR
classes = nltk.pos_tag(tokens)
print(classes)

#identificar entidadades pessoas empresas, etc
entidades = nltk.chunk.ne_chunk(classes)
print(entidades)
