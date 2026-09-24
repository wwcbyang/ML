import pandas as pd


def main():
    path = "data/titanic_train.csv"
    df = pd.read_csv(path) # Read csv file into a DataFrame
    print(df.head(5))
    print(type(df))
    print(df.shape) # 행, 열 반환 -> (type: tuple)
    print(df.info()) # Null, type, 데이터갯수 등
    print(df.describe()) # 대략적인 수치 분석

    sex_counts = df['Sex'].value_counts()
    pclass_counts = df['Pclass'].value_counts() # cf) Pclass -> 객실등급
    print(sex_counts)
    print(pclass_counts)
    print(type(sex_counts)) # Series
    return

if __name__ == "__main__":
    main()