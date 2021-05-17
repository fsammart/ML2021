import math

# Function to find distance
def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d


def calculate_margin(m,b,points):
    points.sort(key = lambda x: shortest_distance(x[0],x[1], m,-1,b))
    p0 = points[0]
    return shortest_distance(p0[0],p0[1], m,-1,b)

def calculate_correctness(m,b,points):
    for p in points:
        if (m*p[0] + b - p[1]) *p[3] <0:
            return False

    return True