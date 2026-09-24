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
    #srs_pclass = df['Pclass']
    #print(srs_pclass) # -> Series와 DataFrame(원본)은 같은 Index.
    #print(df.value_counts()) # 모든 데이터를 세어보는 것 -> 너무 많음
    print(df['Embarked'].value_counts())
    print(df['Embarked'].value_counts(dropna=False)) # NaN 값도 포함해서 셈. 2개가 나옴.

    return

if __name__ == "__main__":
    main()