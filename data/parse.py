import pandas as pd

def return_type(descr = None):
    if descr == 'INPUT':
        return 1
    if descr == 'OUTPUT':
        return 2
    return 0

def parse_file(filename):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    pin_names = []
    x_coord = []
    y_coord = []
    driver_types = []

    for i in range(26, 122, 3):
        name = Lines[i].split()[1]
        type = return_type(Lines[i].split()[7])
        x = int(Lines[i+2].split()[3])
        y = int(Lines[i+2].split()[4])
        pin_names.append(name)
        x_coord.append(x)
        y_coord.append(y)
        driver_types.append(type)

    for line in Lines[126:]:
        splitted = line.split()
        name = splitted[0]
        type = return_type()
        x = int(splitted[5])
        y = int(splitted[6])
        pin_names.append(name)
        x_coord.append(x)
        y_coord.append(y)
        driver_types.append(type)
        
    data = {'name_pin': pin_names, 'x': x_coord, 'y': y_coord, 'driver_type': driver_types}
    df = pd.DataFrame.from_dict(data)
    return df
