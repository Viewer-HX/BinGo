'''
    Get the graph log files.
'''

import os
import re

rootPath = './'
gdatPath = rootPath + '/data_np2/'
logsPath = rootPath + '/logs/'
logsFile = logsPath + '/construct_graphs.txt'

def GetFilePath(line):
    pattern0 = re.compile(r'\[\d+\] .* \[TIME:')
    filepath = pattern0.findall(line)
    filepath = filepath[0]

    pattern1 = re.compile(r'\] .* \[')
    filepath = pattern1.findall(filepath)
    filepath = filepath[0]

    filepath = filepath[2:-2]

    return filepath

def main():
    errSign = 0
    count = 0

    f = open(logsFile)
    line = f.readline()
    while line:
        if (line.lower().startswith('[error]')):
            errSign = 1

        elif (line.lower().startswith('[warning]')):
            # errSign = 1
            pass

        elif (line.startswith('[INFO] <main> save the graph information')):
            if (1 == errSign):
                # print(line, end='')
                filepath = GetFilePath(line)
                # print(filepath)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print('Removing ' + filepath)

                count += 1
            errSign = 0

        elif (line.startswith('=========')):
            errSign = 0

        # print(line, end='')
        line = f.readline()
    f.close()

    print('[INFO] There are totally %d records with errors.' % count)

    return 0

if __name__ == '__main__':
    main()