import os
import pickle


def print_data(name):
    data = pickle.loads(open(f'Data/{name}_encoding.pickle', 'rb').read())
    print(data)


def main():
    print_data("Irina_Baribkina")


if __name__ == '__main__':
    main()