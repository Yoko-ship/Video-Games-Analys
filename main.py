import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Games:
    def __init__(self):
        df = pd.read_csv("vgsales.csv")
        self.cleanned = df.dropna()
        self.figure = (12,6)
    
    def draw_bar_chart(self,top_publishers,x,y,xlabel,ylabel,title):
        x = top_publishers[x]
        y = top_publishers[y]
        plt.figure(figsize=self.figure)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.bar(x,y,width=0.3,color='green')
        plt.show()

    def draw_line_chart(self,value,title,xlabel,ylabel):
        x = value["Year"]
        y = value["Name"]
        plt.figure(figsize=(12,6))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x,y,color="green",marker='o',markersize=4,linewidth=2)
        plt.grid(True,linestyle='-',alpha=0.6)
        plt.show()


    def handle_data(self):
        released_games_count = self.cleanned.groupby("Year")["Name"].count().reset_index()
        unique_platforms = self.cleanned["Platform"].unique()
        year_range = self.cleanned["Year"].agg(["min","max"]).astype(np.int64)


        global_sales = self.cleanned.groupby("Year")["Global_Sales"].sum()
        platform_sales = self.cleanned.groupby("Platform")["Global_Sales"].sum()
        platform_released_games = self.cleanned.groupby("Platform")["Name"].count()


        genre_games = self.cleanned.groupby("Genre")["Name"].count().reset_index()
        global_genre_sales = self.cleanned.groupby("Genre")["Global_Sales"].sum()
        genre_by_years = self.cleanned.groupby(["Year","Genre"])["Genre"].count().tail(20)

        top_publishers = self.cleanned.groupby("Publisher")["Global_Sales"].count().sort_values(ascending=False)[:5].reset_index()
        publishers_genre = self.cleanned.groupby("Publisher")["Genre"].unique()


        popular_genres_jp = self.cleanned.groupby("Platform")["JP_Sales"].sum().sort_values(ascending=False)
        popular_genres_eu = self.cleanned.groupby("Platform")["EU_Sales"].sum().sort_values(ascending=False)
        popular_genres_global = self.cleanned.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).reset_index()
        top_popular_genres_global = popular_genres_global[popular_genres_global["Global_Sales"] > 10]


        self.draw_line_chart(released_games_count,"Количество выпущенных игр по годам","Год","Количество игр")
        self.draw_bar_chart(top_publishers,"Publisher","Global_Sales","Издатель","Количество проданных уникальных игр","Топ 5 издателей с 1980 - 2020")
        self.draw_bar_chart(top_popular_genres_global,"Platform","Global_Sales","Платформа","Количество уникальных продаваемых игр по платформам","Топ платформ с 1980 - 2020")
        self.draw_bar_chart(genre_games,"Genre","Name","Жанр","Количество","Количество выпущенных игр по жанру с 1980 - 2020")


if __name__ == "__main__":
    df = Games()
    df.handle_data()






