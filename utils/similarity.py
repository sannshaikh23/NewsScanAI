from numpy import dot
from numpy.linalg import norm

def compare_embeddings(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
