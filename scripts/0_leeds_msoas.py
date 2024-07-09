import pandas as pd


def main():
    df = pd.read_csv("data/external/ZoningTemplate.csv")
    for msoa in df.loc[df["lad_name"].eq("Leeds"), "msoa"].to_list():
        print(f'"{msoa}"')


if __name__ == "__main__":
    main()
