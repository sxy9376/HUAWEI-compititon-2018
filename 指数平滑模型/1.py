# coding=utf-8
import sys
import os
import predictor


def main():
    print('main function begin.')
#f len(sys.argv) != 4:
        #print('parameter is incorrect!')
        #print('Usage: python esc.py ecsDataPath inputFilePath resultFilePath')
        #exit(1)
    # Read the input files
    ecsDataPath = 'TrainData_2015.12.txt'
    inputFilePath = 'input_3hosttypes_5flavors_1week.txt'
    resultFilePath = 'result.txt'

    ecs_infor_array = read_lines(ecsDataPath)
    input_file_array = read_lines(inputFilePath)
    print(ecs_infor_array)
    print(input_file_array)
    # implementation the function predictVm
    predic_result = predictor.predict_vm(ecs_infor_array, input_file_array)
    # write the result to output file
    if len(predic_result) != 0:
        write_result(predic_result, resultFilePath)
    else:
        predic_result.append("NA")
        write_result(predic_result, resultFilePath)
    print('main function end.')


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print('file not exist: ' + file_path)
        return None


if __name__ == "__main__":
    main()
