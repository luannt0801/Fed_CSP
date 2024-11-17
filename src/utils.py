import yaml
import os
import ping3
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import logging
from src.logging import *

"""
    Define color
"""

class color:
   PURPLE = '\033[1;35;48m'
   CYAN = '\033[1;36;48m'
   BOLD = '\033[1;37;48m'
   BLUE = '\033[1;34;48m'
   GREEN = '\033[1;32;48m'
   YELLOW = '\033[1;33;48m'
   RED = '\033[1;31;48m'
   BLACK = '\033[1;30;48m'
   UNDERLINE = '\033[4;37;48m'
   END = '\033[1;37;0m'

def find_color(color_):
    if color_ == "red":
        return color.RED
    elif color_ == "yellow":
        return color.YELLOW
    elif color_ == "green":
        return color.GREEN
    elif color_ == "cyan":
        return color.CYAN
    elif color_ == "purple":
        return color.PURPLE
    elif color_ == "blue":
        return color.BLUE
    else:
        return ""
    
def cur_time_str():
    cur_time = datetime.now()
    cur_time_string = "[" + cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + "] "
    return cur_time_string


# def print_log(line, color_="green", show_time=True):
#     if type(line) == str:
#         color_str = find_color(color_)
#         if show_time:
#             print(color.PURPLE + cur_time_str(), end=color.END)
#         else:
#             line = "           " + line
#         print(color_str + line + color.END)
#     else:
#         print(line)

def print_log(line, color_="green", show_time=True):
    if isinstance(line, str):
        color_str = find_color(color_)
        message = f"{cur_time_str()} {line}" if show_time else f"           {line}"
        # Ghi vào logger (sử dụng cấp độ INFO cho các tin nhắn từ print_log)
        logger.info(line)
        # In ra console với màu sắc
        print(color_str + message + CustomFormatter.reset)
    else:
        print(line)


def int_to_ubyte(num):
    return num.to_bytes(1, "big", signed=False)


def int_to_Nubyte(num, N):
    return num.to_bytes(N, "big", signed=False)


def choose_file_in_folder_by_order(folder, file_order):
    dir_name = folder
    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter(lambda x: os.path.isfile(os.path.join(dir_name, x)),
                                  os.listdir(dir_name)))
    return list_of_files[file_order]


def ping_host(host, count=10):
    ping_result = [ping3.ping(host) for _ in range(count)]
    ping_result = [result for result in ping_result if
                   result is not None]  # Loại bỏ các kết quả None (không thành công)

    if ping_result:
        avg_latency = sum(ping_result) / len(ping_result)
        min_latency = min(ping_result)
        max_latency = max(ping_result)
        packet_loss = (1 - len(ping_result) / count) * 100
    else:
        avg_latency = None
        min_latency = None
        max_latency = None
        packet_loss = 100

    return {
        'host': host,
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'packet_loss': packet_loss
    }

def elbow_method(X, max_k=10):
    """
    The elbow method for determining the optimal number of clusters (k) in KMeans clustering.
    """
    wcss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wcss.append(np.sum((X - kmeans.centroids[kmeans.labels]) ** 2))
    plt.plot(range(1, max_k+1), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.show()

def sillohowd_method(X, max_k = 10):
    silhouette_scores = []
    s_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels
        
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    plt.plot(range(2, max_k+1), silhouette_scores)
    plt.title('Silhouette Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()