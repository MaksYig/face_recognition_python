import os
import pickle


def print_data(name):
    data = pickle.loads(open(f'Data/{name}_encoding.pkl', 'rb').read())
    print(data)
    print(f"Length:{len(data['encodings'])}")





def main():
    print_data("Yigal_Maksimov")


if __name__ == '__main__':
    main()