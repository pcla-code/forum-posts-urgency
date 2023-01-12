import csv

if __name__ == '__main__':
    frequencies = {i: 0 for i in range(1, 8)}
    with open('Stanford.csv', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            label = int(round(float(row[2])))
            frequencies[label] += 1
    print(frequencies)
    print(sum([i for i in frequencies.values()]))
