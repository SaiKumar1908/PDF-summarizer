import os
from PyPDF2 import PdfReader
from nltk.cluster.util import cosine_distance
import nltk
import numpy as np
from stop_words import get_stop_words
import networkx as nx
def convertingPdfToText(pdfName):
    reader = PdfReader(pdfName)
    number_of_pages = len(reader.pages)
    text = []
    for i in range(number_of_pages):
       page = reader.pages[i]
       text.append(page.extract_text())
    return text

def storingTheContentToFile(fileName,contentOfPdf): 
       f = open(fileName,'x')
       f = open(fileName,'a')
       for i in contentOfPdf:
          for j in i:
            if j != '\n':
               f.write(j)
       f.close()

def  read_article(fileName):
       f = open(fileName,'r')
       filedata = f.readlines()
       article = filedata[0].split('. ')
       sentences=[]
       for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
       sentences.pop()
       return sentences

def sentence_similarity(s1,s2,stopwords=None):
       if stopwords is None:
           stopwords=[]
       s1 = [w.lower() for w in s1]
       s2 = [w.lower() for w in s2]
       all_words = list(set(s1+s2))
       
       vector1,vector2 = [0]*len(all_words),[0]*len(all_words)
       for w in s1:
          if w in stopwords:
              continue
          vector1[all_words.index(w)]+=1

       for w in s2:
          if w in stopwords:
              continue
          vector2[all_words.index(w)]+=1
       return 1-cosine_distance(vector1,vector2) 

def gen_sim_matrix(sentences,stopwords):
       similarity_matrix = np.zeros((len(sentences),len(sentences)))
       for idx1 in range(len(sentences)):
           for idx2 in range(len(sentences)):
               if idx1 == idx2:
                   continue
               similarity_matrix[idx1][idx2] =  sentence_similarity(sentences[idx1],sentences[idx2],stopwords)
       return similarity_matrix

def generate_summary(fileName,top_n=5):
       stopwords = get_stop_words('english')
       summarize_text=[]
       sentences = read_article(fileName)
       sentence_similarity_matrix = gen_sim_matrix(sentences,stopwords)
       sentence_similarity_graph=nx.from_numpy_array(sentence_similarity_matrix)
       scores = nx.pagerank(sentence_similarity_graph)
       ranked_sentence=sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
       for i in range(top_n):
           summarize_text.append(" ".join(ranked_sentence[i][1]))
       print('Summary \n','. '.join(summarize_text))
       
       os.remove(f"E:\PythonBasicPrograms\MiniProject\{fileName}")
     
pdfName = input('Enter your pdf name for which you want to generate summary:')
contentOfPdf = convertingPdfToText(pdfName)
fileName = input('Enter your file name in which you want to store the pdf content:')
storingTheContentToFile(fileName,contentOfPdf)
generate_summary(fileName,30)