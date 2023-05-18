import enchant
import re
   
def accuracyByLevenshteinDistance(textOCR,label):
    #tira espaços, tab e \n
    text = re.sub(r"[\n\t\s]*", "", textOCR)
    labelText = re.sub(r"[\n\t\s]*", "", label)
    
    #calcula a distancia de levenshtein entre os textos
    levDistance = enchant.utils.levenshtein(text, labelText)
    
    #calcula a acurácia fazendo accuracy = n-erros/n
    accuracy = (len(labelText)-levDistance) / len(labelText)
    return accuracy
