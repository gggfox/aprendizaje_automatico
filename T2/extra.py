import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(X,Y):
    area = np.pi*3
    red = "#ff0000"
    green = "#00ff00"
    colors = [red if i == 0 else green for i in Y]
    
    plt.xlabel('test1')
    plt.ylabel('test2')
    
    plt.scatter(X[0], X[1], s=area, c=colors)
    plt.show()

def main():
    df = pd.read_csv("dataset-1.csv")
    X = np.array(df.columns[0:2])
    print(df['accepted'])
    print(X)
    scatter_plot([df['mark1'],df['mark2']],df['accepted']) 


if __name__ == '__main__':
    main()