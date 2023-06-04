import enchant
import re

def cleanText(t):
    return re.sub(r"[\n\t\s]*", "", t)

def accuracyByLevenshteinDistance(textOCR,label):
    #tira espaços, tab e \n
    text = cleanText(textOCR)
    labelText = cleanText(label)

    #calcula a distancia de levenshtein entre os textos
    levDistance = enchant.utils.levenshtein(text, labelText)

    #calcula a acurácia fazendo accuracy = n-erros/n
    accuracy = (len(labelText)-levDistance) / len(labelText)
    return accuracy
