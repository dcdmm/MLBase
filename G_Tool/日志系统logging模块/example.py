import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(lineno)d - %(message)s ',
                    filename='example.log')


def division(a, b):
    return (a + 1) / b


number1_list = [34, 20., 34]
number2_list = [80, 3., 0.]

for i in range(len(number1_list)):
    try:
        result = number1_list[i] / number2_list[i]
        if (type(number1_list[i]) or type(number2_list[i])) != float:
            logging.warning(msg="no float type")
        else:
            logging.info(msg="result=" + str(result))
    except ZeroDivisionError as reason:
        logging.error("Error:" + str(reason))
