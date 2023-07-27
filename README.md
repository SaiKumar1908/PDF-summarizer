# PDF-summarizer
import os
from PyPDF2 import PdfReader
from nltk.cluster.util import cosine_distance
import nltk
import numpy as np
from stop_words import get_stop_words
import networkx as nx
Download the PyPDF2, nltk.custer.util nltk, numpy, stopwords and networkx for generating pdf summary
